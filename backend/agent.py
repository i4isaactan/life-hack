"""LangGraph workflow: photo and constraints in, layout and cart out.

The graph is linear and acyclic:

    analyze_room -> build_query -> retrieve_items -> select_items
                 -> solve_layout -> build_cart -> narrate

Nodes stream to the client through get_stream_writer(). That is preferred over
astream_events, which on a graph this small emits only generic on_chain_* frames
with no domain meaning, and over an asyncio.Queue, which would be redundant
plumbing around what the writer already provides.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from . import config
from .models import (
    PLACEMENT_ORDER,
    Alternative,
    Cart,
    CartLine,
    CatalogItem,
    ChatMessage,
    DimensionProposal,
    DimensionSource,
    Intent,
    IntentKind,
    Detection,
    LayoutResult,
    RoleOptions,
    RoomAnalysis,
    Role,
)
from .rag_engine import (
    CatalogIndex,
    DetectionProvider,
    IntentProvider,
    VisionProvider,
)
from .bundles import build_bundles
from .solver import LayoutSolver

log = logging.getLogger(__name__)


class GraphState(TypedDict, total=False):
    messages: list[ChatMessage]
    image_b64: str | None
    budget_cents: int
    aesthetic: str
    prompt: str
    # Structured reading of `prompt`, produced by the parse_intent node. This
    # is what lets "make it cheaper" change the budget instead of becoming
    # three meaningless tokens in an embedding query.
    intent: Intent | None
    # Items the user has rejected this session; excluded from re-selection.
    rejected_ids: list[str]
    # Roles the user asked to drop from the design.
    excluded_roles: list[str]
    # Per-role width ceilings from "the sofa is too big".
    size_caps: dict[str, float]
    # User-supplied room facts that override the vision estimate.
    measurements: dict[str, Any] | None
    # Furniture found in the photo, carried across turns so a follow-up does
    # not pay for a second detection pass over an image that has not changed.
    detections: list[Detection] | None
    room: RoomAnalysis
    query: str
    candidates: dict[str, list[CatalogItem]]
    selected: list[CatalogItem]
    options: list[RoleOptions]
    layout: LayoutResult
    cart: Cart
    errors: Annotated[list[str], lambda a, b: a + b]


# How much of the budget each role may claim. Seating and rugs dominate a real
# room budget; lamps should never crowd out a sofa.
BUDGET_SHARE: dict[Role, float] = {
    Role.SOFA: 0.46,
    Role.RUG: 0.20,
    Role.COFFEE_TABLE: 0.14,
    Role.ACCENT_CHAIR: 0.22,
    Role.FLOOR_LAMP: 0.10,
}


def build_role_options(
    role: Role,
    candidates: list[CatalogItem],
    chosen_ids: set[str],
    budget_cents: int,
    spent_cents: int,
) -> RoleOptions:
    """The alternatives for one role, priced against the current pick.

    `affordable` accounts for the swap being a *replacement*: the outgoing
    item's price comes back before the incoming one is charged, so a like-priced
    alternative stays affordable even when the design is already at budget.
    """
    chosen = next((i for i in candidates if i.id in chosen_ids), None)
    if chosen is None:
        return RoleOptions(role=role, selected_id="", alternatives=[])

    headroom = budget_cents - spent_cents + chosen.price_cents
    return RoleOptions(
        role=role,
        selected_id=chosen.id,
        alternatives=[
            Alternative(
                item_id=item.id,
                name=item.title,
                merchant=item.merchant,
                role=item.role,
                price_cents=item.price_cents,
                price_delta_cents=item.price_cents - chosen.price_cents,
                swatch=item.swatch,
                image_url=item.image_url,
                materials=item.materials,
                primary_color=item.primary_color,
                style_tags=item.style_tags,
                width_cm=item.dimensions.width_cm,
                depth_cm=item.dimensions.depth_cm,
                height_cm=item.dimensions.height_cm,
                affordable=item.price_cents <= headroom,
            )
            for item in candidates
            if item.id != chosen.id
        ],
    )


def build_graph(
    index: CatalogIndex,
    vision: VisionProvider,
    intent_provider: IntentProvider | None = None,
    detector: DetectionProvider | None = None,
):
    """Compile the workflow. No checkpointer: each request is single-shot."""

    intents = intent_provider or IntentProvider()
    detections_for = detector or DetectionProvider()

    def parse_intent(state: GraphState) -> dict[str, Any]:
        """Read the message as structured constraints before anything else runs.

        Constraints accumulate rather than reset: an unmentioned budget or
        aesthetic carries forward from the session, so "make it cheaper" changes
        only the budget and leaves the brief intact.
        """
        writer = get_stream_writer()
        intent = intents.parse(
            message=state.get("prompt", ""),
            history=state.get("messages") or [],
            budget_cents=state.get("budget_cents"),
            current_items=state.get("selected") or [],
        )
        writer({"type": "intent", "intent": intent.model_dump(mode="json")})

        update: dict[str, Any] = {"intent": intent}

        if intent.budget_cents:
            update["budget_cents"] = intent.budget_cents
            writer(
                {
                    "type": "text_delta",
                    "text": f"Adjusting the budget to ${intent.budget_cents / 100:,.0f}. ",
                }
            )
        if intent.aesthetic:
            update["aesthetic"] = intent.aesthetic
        if intent.style_note:
            # Folded into the query rather than replacing the aesthetic, since
            # "warmer" refines a style rather than naming a new one.
            update["prompt"] = f"{state.get('prompt', '')}. {intent.style_note}"

        # Rejections and exclusions persist for the session; a piece the user
        # turned down should not reappear two turns later.
        if intent.reject_item_ids:
            update["rejected_ids"] = list(
                {*(state.get("rejected_ids") or []), *intent.reject_item_ids}
            )
        if intent.remove_roles:
            update["excluded_roles"] = list(
                {
                    *(state.get("excluded_roles") or []),
                    *(r.value for r in intent.remove_roles),
                }
            )
        if intent.max_width_cm:
            update["size_caps"] = {
                **(state.get("size_caps") or {}),
                **intent.max_width_cm,
            }

        # A reroll rejects whatever currently fills that role, so the next
        # search cannot simply hand back the same item.
        if intent.reroll_roles:
            current = state.get("selected") or []
            rerolled = {
                i.id for i in current if i.role in intent.reroll_roles
            }
            if rerolled:
                update["rejected_ids"] = list(
                    {*(update.get("rejected_ids") or state.get("rejected_ids") or []), *rerolled}
                )
        return update

    def answer_directly(state: GraphState) -> dict[str, Any]:
        """Respond from existing state - no retrieval, no solve, no spend.

        Explaining a choice already made needs the design, not a new one.
        """
        writer = get_stream_writer()
        intent = state.get("intent") or Intent()
        layout = state.get("layout")

        if intent.kind is IntentKind.EXPLAIN and layout is not None:
            wanted = intent.explain_role
            placements = [
                p for p in layout.placements if wanted is None or p.role is wanted
            ]
            if placements:
                # Lead with the model's sentence, then the solver's actual
                # rationale and tolerance. The second part is the grounded
                # claim - it comes from geometry, not from the language model.
                if intent.reply:
                    writer({"type": "text_delta", "text": intent.reply})
                for p in placements:
                    writer(
                        {
                            "type": "text_delta",
                            "text": f"\n• {p.name} sits {p.rationale or 'where it fits'} "
                            f"— {p.confidence.value} confidence, ±{p.tolerance_cm:.0f}cm.",
                        }
                    )
                # Withheld pieces are the most interesting thing to explain:
                # they are a deliberate refusal, not a failure.
                for w in layout.withheld:
                    if wanted is None or w.role is wanted:
                        writer(
                            {
                                "type": "text_delta",
                                "text": f"\n• {w.name} is on hold: {w.reason}",
                            }
                        )
                return {}

        writer(
            {
                "type": "text_delta",
                "text": intent.reply
                or "Tell me your room size, budget and the look you want, and "
                "I'll put a design together.",
            }
        )
        return {}

    def route_intent(state: GraphState) -> str:
        """Skip the design pipeline when the turn does not need one."""
        intent = state.get("intent") or Intent()
        if intent.kind in (IntentKind.EXPLAIN, IntentKind.CHITCHAT):
            # Only answer directly if there is something to talk about.
            if state.get("layout") is not None or intent.kind is IntentKind.CHITCHAT:
                return "answer_directly"
        return "analyze_room"

    def analyze_room(state: GraphState) -> dict[str, Any]:
        writer = get_stream_writer()
        writer({"type": "text_delta", "text": "Analyzing your room… "})
        room = vision.analyze(state.get("image_b64"))

        # User-supplied measurements always beat an estimate from a photo, and
        # promote the room to `measured`, which is what unlocks wall-hugging
        # pieces the solver would otherwise withhold.
        supplied = state.get("measurements") or {}
        if supplied:
            room = room.model_copy(update=dict(supplied))

        # What is already in the room. Cached on the state by the caller
        # because this is a second vision call on the critical path, and a
        # follow-up turn ("make it cheaper") re-runs the graph against the
        # same photo - paying for detection again would be pure waste.
        cached = state.get("detections")
        if cached is not None:
            room = room.model_copy(update={"detections": list(cached)})
        elif state.get("image_b64"):
            writer({"type": "text_delta", "text": "and what's already in it… "})
            found = detections_for.detect(state.get("image_b64"))
            room = room.model_copy(update={"detections": found})

        writer({"type": "room_analysis", "room": room.model_dump(mode="json")})
        qualifier = "measured at" if room.measured else "roughly"
        writer(
            {
                "type": "text_delta",
                "text": f"I read it as {qualifier} {room.width_cm/100:.1f}m × "
                f"{room.depth_cm/100:.1f}m with {room.flooring} flooring. ",
            }
        )
        if room.detections:
            names = ", ".join(d.label for d in room.detections[:4])
            writer(
                {
                    "type": "text_delta",
                    "text": f"I can see {names} in there. ",
                }
            )
        return {"room": room, "detections": room.detections}

    def build_query(state: GraphState) -> dict[str, Any]:
        room = state["room"]
        aesthetic = state.get("aesthetic") or "modern"
        parts = [
            aesthetic,
            state.get("prompt", ""),
            f"{room.wall_color} walls",
            f"{room.flooring} floor",
        ]
        # What the user already owns is a stronger style signal than the
        # adjective they typed, so the existing pieces steer retrieval toward
        # things that will sit with them. Appended last and capped, so it
        # colours the query without drowning out an explicit request: someone
        # asking for "something brighter" must not be handed more of the beige
        # they are trying to replace.
        existing = room.existing_style
        if existing:
            parts.append(f"to sit alongside {existing}")
        return {"query": ". ".join(p for p in parts if p)}

    def retrieve_items(state: GraphState) -> dict[str, Any]:
        writer = get_stream_writer()
        writer({"type": "text_delta", "text": "Searching the catalog… "})

        room = state["room"]
        budget = state.get("budget_cents") or config.DEFAULT_BUDGET_CENTS
        # Filtering by room extent here keeps oversized pieces out of the
        # solver entirely, so it never wastes a slot on a guaranteed skip.
        max_w = room.width_cm - 2 * config.WALL_MARGIN_CM
        max_d = room.depth_cm - 2 * config.WALL_MARGIN_CM

        rejected = set(state.get("rejected_ids") or [])
        excluded = set(state.get("excluded_roles") or [])
        size_caps = state.get("size_caps") or {}

        candidates: dict[str, list[CatalogItem]] = {}
        errors: list[str] = []
        for role in PLACEMENT_ORDER:
            # The user asked for this role to be dropped entirely.
            if role.value in excluded:
                candidates[role.value] = []
                continue
            # The share steers balance; it is not a hard ceiling. Applied as a
            # retrieval filter it can empty a whole category (no sofa is under
            # a 42% slice of a small budget), which silently removes seating
            # from the design. So retry unpriced and let select_items enforce
            # the real budget against options that actually exist.
            cap = int(budget * BUDGET_SHARE.get(role, 0.15))
            # "The sofa is too big" narrows this role below what the room
            # alone would allow.
            role_w = min(max_w, size_caps.get(role.value, max_w))
            try:
                hits = index.search(
                    query=state["query"],
                    role=role,
                    max_price_cents=cap,
                    max_width_cm=role_w,
                    max_depth_cm=max_d,
                    limit=6,
                )
                if not hits:
                    hits = index.search(
                        query=state["query"],
                        role=role,
                        max_width_cm=role_w,
                        max_depth_cm=max_d,
                        limit=6,
                    )
            except Exception as exc:
                log.warning("catalog search failed for %s: %s", role.value, exc)
                errors.append(f"search failed for {role.value}")
                hits = []
            # Drop anything the user has turned down. Done after retrieval so
            # a rejection narrows the shortlist rather than the search itself,
            # which would let a rejected item's near-duplicate rank in.
            if rejected:
                hits = [h for h in hits if h.id not in rejected]
            candidates[role.value] = hits

        found = sum(len(v) for v in candidates.values())
        if found == 0:
            writer(
                {
                    "type": "text_delta",
                    "text": "No catalog items matched those constraints. ",
                }
            )
        return {"candidates": candidates, "errors": errors}

    def select_items(state: GraphState) -> dict[str, Any]:
        """Greedy pick by role priority, keeping the running total in budget."""
        budget = state.get("budget_cents") or config.DEFAULT_BUDGET_CENTS
        candidates = state.get("candidates", {})

        available = [r for r in PLACEMENT_ORDER if candidates.get(r.value)]

        # Buy in order of how much each piece makes the room usable, not in
        # placement order: at a tight budget a sofa and nothing else beats a rug
        # and a lamp with nowhere to sit. Only seating is reserved for.
        priority = [
            Role.SOFA,
            Role.RUG,
            Role.COFFEE_TABLE,
            Role.ACCENT_CHAIR,
            Role.FLOOR_LAMP,
        ]
        essential = [Role.SOFA]
        order = [r for r in priority if r in available]

        selected: list[CatalogItem] = []
        spent = 0
        for idx, role in enumerate(order):
            options = candidates[role.value]

            # Reserve the cheapest option for each *essential* role still to
            # come, so a premium rug cannot price out the seating behind it.
            reserve = sum(
                min(candidates[r.value], key=lambda i: i.price_cents).price_cents
                for r in order[idx + 1 :]
                if r in essential
            )
            affordable = budget - spent - reserve

            # Retrieval ordered these by relevance. Prefer the best-ranked item
            # within reach, but consider every affordable option rather than
            # only the top one - the best fit for a tight budget is often
            # further down the list.
            in_reach = [i for i in options if i.price_cents <= affordable]
            pick = in_reach[0] if in_reach else None
            if pick is None:
                cheapest = min(options, key=lambda i: i.price_cents)
                if spent + cheapest.price_cents <= budget:
                    pick = cheapest
            if pick is not None:
                selected.append(pick)
                spent += pick.price_cents

        # Retrieval found several candidates per role and we kept one. The rest
        # are equally valid picks, so offer them rather than presenting a single
        # choice as though it were the only one.
        chosen_ids = {i.id for i in selected}
        options = [
            build_role_options(role, candidates[role.value], chosen_ids, budget, spent)
            for role in order
            if any(i.id in chosen_ids for i in candidates[role.value])
        ]
        options = [o for o in options if o.alternatives]

        if options:
            writer = get_stream_writer()
            writer(
                {
                    "type": "alternatives",
                    "options": [o.model_dump(mode="json") for o in options],
                }
            )

        return {"selected": selected, "options": options}

    def solve_layout(state: GraphState) -> dict[str, Any]:
        writer = get_stream_writer()
        writer({"type": "text_delta", "text": "Laying out the floor plan… "})
        layout = LayoutSolver(state["room"]).solve(state.get("selected", []))
        writer({"type": "layout_update", "layout": layout.model_dump(mode="json")})

        # Anything withheld for want of measurements becomes an explicit ask.
        # Questions are de-duplicated by field: one "what are the dimensions?"
        # covers every piece waiting on it.
        if layout.withheld:
            asks: dict[str, dict[str, Any]] = {}
            for item in layout.withheld:
                for need in item.needs:
                    entry = asks.setdefault(
                        need.field,
                        {"field": need.field, "question": need.question, "affects": []},
                    )
                    entry["affects"].append(item.name)
            writer(
                {
                    "type": "clarification_needed",
                    "questions": list(asks.values()),
                    "withheld": [w.model_dump(mode="json") for w in layout.withheld],
                }
            )

            # Offer the estimate back for confirmation rather than demanding a
            # measurement. A client renders this as a prefilled, editable field:
            # accepting costs one tap, and people are much better at spotting a
            # wrong number than producing a right one.
            room = state["room"]
            if room.dimension_source is DimensionSource.ESTIMATED:
                proposal = DimensionProposal(
                    width_cm=room.width_cm,
                    depth_cm=room.depth_cm,
                    source=room.dimension_source,
                    unlocks=sorted({w.role.value for w in layout.withheld}),
                    question=(
                        f"I read your room as about {room.width_cm / 100:.1f}m × "
                        f"{room.depth_cm / 100:.1f}m. Is that close? Confirming "
                        "lets me place the pieces I'm holding back."
                    ),
                )
                writer(
                    {
                        "type": "dimension_proposal",
                        "proposal": proposal.model_dump(mode="json"),
                    }
                )
                writer({"type": "text_delta", "text": f"\n\n{proposal.question}"})
        return {"layout": layout}

    def build_cart(state: GraphState) -> dict[str, Any]:
        writer = get_stream_writer()
        budget = state.get("budget_cents") or config.DEFAULT_BUDGET_CENTS
        layout = state["layout"]

        # Bill only for what actually made it into the plan. Billing a skipped
        # item would charge for furniture the customer cannot place.
        placed_ids = {p.item_id for p in layout.placements}
        lines = [
            CartLine(
                item_id=item.id,
                name=item.title,
                merchant=item.merchant,
                role=item.role,
                price_cents=item.price_cents,
                checkout_url=item.checkout_url,
                image_url=item.image_url,
            )
            for item in state.get("selected", [])
            if item.id in placed_ids
        ]
        cart = Cart(
            lines=lines,
            subtotal_cents=sum(line.line_total_cents for line in lines),
            budget_cents=budget,
        )
        writer(
            {
                "type": "cart_update",
                "cart": cart.model_dump(mode="json"),
                "subtotal_cents": cart.subtotal_cents,
                "budget_cents": cart.budget_cents,
                "over_budget": cart.over_budget,
            }
        )

        # Bundles extend the design the user just got, so they are built from
        # the placed items and the solved layout. Failure here must not take
        # the turn down with it: a missing suggestion is a smaller loss than a
        # lost design.
        try:
            placed_items = [i for i in state.get("selected", []) if i.id in placed_ids]
            bundles = build_bundles(
                placed_items,
                list(index.all_items()),
                budget_cents=budget,
                spent_cents=cart.subtotal_cents,
                room=state.get("room"),
                layout=layout,
            )
            if bundles:
                writer(
                    {
                        "type": "bundles",
                        "bundles": [b.model_dump(mode="json") for b in bundles],
                    }
                )
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("bundle suggestions unavailable: %s", exc)

        return {"cart": cart}

    def narrate(state: GraphState) -> dict[str, Any]:
        writer = get_stream_writer()
        layout = state["layout"]
        cart = state["cart"]

        n = len(layout.placements)
        if n == 0:
            writer(
                {
                    "type": "text_delta",
                    "text": "I could not fit anything into this room within "
                    "those constraints. Try a larger room or a higher budget.",
                }
            )
            return {}

        total = cart.subtotal_cents / 100
        budget = cart.budget_cents / 100
        writer(
            {
                "type": "text_delta",
                "text": f"\n\nI placed {n} piece{'s' if n != 1 else ''} "
                f"totalling ${total:,.2f} against your ${budget:,.0f} budget. ",
            }
        )
        for p in sorted(layout.placements, key=lambda p: PLACEMENT_ORDER.index(p.role)):
            writer(
                {
                    "type": "text_delta",
                    "text": f"\n• {p.name} ({p.merchant}) — ${p.price_cents/100:,.2f}",
                }
            )
        if layout.skipped:
            # Report each item's actual reason; collapsing them all into "no
            # space" misexplains a table dropped for want of seating.
            lines = "".join(
                f"\n• {s.name} — {s.detail or s.reason.replace('_', ' ')}"
                for s in layout.skipped
            )
            writer({"type": "text_delta", "text": f"\n\nCouldn't include:{lines}"})

        # Withheld is a different claim from skipped: these fit, but placing
        # them well needs measurements we do not have. Say so plainly rather
        # than showing a confident-looking guess.
        if layout.withheld:
            # Each item carries its own reason - a console is withheld because
            # it hugs a wall, a table because the sofa it serves is withheld -
            # so quote them rather than asserting one reason for all.
            items = "".join(f"\n• {w.name} — {w.reason}" for w in layout.withheld)
            questions = {n.question for w in layout.withheld for n in w.needs}
            writer(
                {
                    "type": "text_delta",
                    "text": "\n\nI've held these back rather than guess at "
                    f"positions that need to be right:{items}\n\n"
                    + " ".join(sorted(questions)),
                }
            )
        return {}

    graph = StateGraph(GraphState)
    for name, fn in [
        ("parse_intent", parse_intent),
        ("answer_directly", answer_directly),
        ("analyze_room", analyze_room),
        ("build_query", build_query),
        ("retrieve_items", retrieve_items),
        ("select_items", select_items),
        ("solve_layout", solve_layout),
        ("build_cart", build_cart),
        ("narrate", narrate),
    ]:
        graph.add_node(name, fn)

    graph.add_edge(START, "parse_intent")
    # Explain and chitchat answer from existing state, skipping retrieval,
    # solving and any API spend.
    graph.add_conditional_edges(
        "parse_intent",
        route_intent,
        {"answer_directly": "answer_directly", "analyze_room": "analyze_room"},
    )
    graph.add_edge("answer_directly", END)
    graph.add_edge("analyze_room", "build_query")
    graph.add_edge("build_query", "retrieve_items")
    graph.add_edge("retrieve_items", "select_items")
    graph.add_edge("select_items", "solve_layout")
    graph.add_edge("solve_layout", "build_cart")
    graph.add_edge("build_cart", "narrate")
    graph.add_edge("narrate", END)

    return graph.compile()


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.WARNING)

    async def main() -> None:
        index = CatalogIndex()
        index.seed()
        app = build_graph(index, VisionProvider())
        state: GraphState = {
            "messages": [],
            "image_b64": None,
            "budget_cents": 250_000,
            "aesthetic": "Japandi",
            "prompt": "calm living room with natural materials",
        }
        async for chunk in app.astream(state, stream_mode="custom"):
            kind = chunk.get("type")
            if kind == "text_delta":
                print(chunk["text"], end="", flush=True)
            else:
                print(f"\n[{kind}]", flush=True)
        print()

    asyncio.run(main())
