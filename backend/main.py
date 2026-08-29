"""FastAPI server: multipart in, Server-Sent Events out.

The SSE contract is the product here, since the chat client is assumed rather
than built. It is documented in README.md and mirrored by test_client.html.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from . import config
from .agent import GraphState, build_graph
from .models import (
    AuthorizationReceipt,
    Detection,
    AuthorizeRequest,
    Cart,
    CartLine,
    ChatMessage,
    CheckoutRequest,
    CheckoutResult,
    DimensionSource,
    MerchantGroup,
    Opening,
    PaymentIntent,
    PaymentIntentRequest,
    PaymentMethod,
    RenderFailure,
    RenderRequest,
    RenderResult,
    RoomRender,
    LayoutResult,
    RoomAnalysis,
    SwapRequest,
    VerificationChallenge,
    VerifyRequest,
)
from . import payments
from .rag_engine import (
    CatalogIndex,
    DetectionProvider,
    IntentProvider,
    VisionProvider,
    prepare_image,
)
from .render_engine import RenderProvider, render_layout, render_room
from .solver import LayoutSolver
from .seed_data import SEED_ITEMS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("roomhack")

# Populated in the lifespan handler. Seeding at import would run twice under
# uvicorn --reload and rebuild the in-memory collection needlessly.
STATE: dict[str, object] = {}

# In-process chat history. The client sends a session_id rather than replaying
# the transcript, which would mean re-uploading the room photo every turn.
SESSIONS: dict[str, list[ChatMessage]] = {}

# Room measurements accumulated per session. Kept separate from chat history
# because they are structured facts that refine across turns: a user may give
# dimensions on one turn and door positions on the next.
MEASUREMENTS: dict[str, dict[str, object]] = {}

# Conversational preferences that accumulate across turns: budget and aesthetic
# as last stated, items the user rejected, roles they dropped, size ceilings.
# Separate from MEASUREMENTS because these are taste, not facts about the room.
PREFERENCES: dict[str, dict[str, object]] = {}

# What /api/render needs from the chat turn that produced the design: the photo
# to paint into, the analysed room, and the solved layout. Rendering is a
# separate request because it takes tens of seconds per item and must not hold
# up the floor plan and cart, which arrive in under a second.
RENDER_CONTEXT: dict[str, dict[str, object]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    index = CatalogIndex()
    count = index.seed()
    vision = VisionProvider()
    renderer = RenderProvider()
    detector = DetectionProvider()
    # Constructed here rather than inside build_graph so its provider status is
    # observable on /api/health; the graph is handed the same instance.
    intent = IntentProvider()
    STATE["index"] = index
    STATE["vision"] = vision
    STATE["renderer"] = renderer
    STATE["intent"] = intent
    STATE["detector"] = detector
    STATE["graph"] = build_graph(index, vision, intent, detector)
    mode = "OpenAI" if config.HAS_OPENAI else "offline mock"
    log.info("catalog seeded with %d items | providers: %s", count, mode)
    # Per-subsystem, so a partial fallback is visible at boot rather than
    # showing up later as a mysteriously generic design.
    log.info(
        "vision: %s | intent: %s | embeddings: %s | detection: %s",
        vision.source,
        intent.source,
        index.embedder.source,
        detector.source,
    )
    if config.HAS_OPENAI:
        degraded = [
            name
            for name, provider in (
                ("vision", vision.source),
                ("intent", intent.source),
                ("embeddings", index.embedder.source),
            )
            if provider != "openai"
        ]
        if degraded:
            log.warning(
                "OPENAI_API_KEY is set but these fell back to offline: %s",
                ", ".join(degraded),
            )
    log.info(
        "renderer: %s | per-item: %s | compose: %s",
        renderer.source,
        renderer.method.value,
        config.GEMINI_IMAGE_MODEL if renderer.can_compose else "unavailable",
    )
    if not config.HAS_OPENAI:
        log.info("No OPENAI_API_KEY set - running fully offline with mock providers.")
    if not (config.HAS_GEMINI or config.HAS_REPLICATE):
        log.info(
            "No GEMINI_API_KEY or REPLICATE_API_TOKEN set - /api/render returns "
            "schematic previews."
        )
    yield
    STATE.clear()
    RENDER_CONTEXT.clear()


app = FastAPI(title="Room Hack", version="1.0.0", lifespan=lifespan)

# Credentials stay off, which is what permits this explicit origin list.
# "null" covers test_client.html opened directly from disk over file://.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Outbound image URLs ---------------------------------------------------
# Catalog image_urls are IKEA's own https:// product photos, which a browser
# loads directly, so nothing needs rewriting or re-hosting. This hook stays as
# the single place an outbound image_url is normalised, because renders arrive
# as data: URIs and any future local imagery would need mapping to a route
# rather than leaking a filesystem path to the client.


def _image_key(image_b64: str) -> str:
    """A short content hash, used to tell whether a photo is the same one.

    Hashing the base64 rather than storing it twice keeps the session dict
    small; a collision would only mean reusing detections for a different
    photo, which blake2b at this width makes irrelevant.
    """
    return hashlib.blake2b(image_b64.encode(), digest_size=16).hexdigest()


def _http_image_url(url: str) -> str:
    """An image_url the browser can load, or "" if it could not be made one."""
    if not isinstance(url, str):
        return ""
    if url.startswith(("http://", "https://", "data:", "/assets/")):
        return url
    # A file:// path would 404 in the browser and leak a local path in the
    # payload, so it is dropped rather than sent. Clients already handle a
    # missing image; they cannot handle a broken one.
    log.warning("dropping non-browser-loadable image_url: %s", url[:80])
    return ""


def _rewrite_image_urls(payload: object) -> object:
    """Recursively normalise every image_url in an outbound SSE/JSON payload."""
    if isinstance(payload, dict):
        return {
            key: (
                _http_image_url(value)
                if key == "image_url" and isinstance(value, str)
                else _rewrite_image_urls(value)
            )
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [_rewrite_image_urls(entry) for entry in payload]
    return payload


# --- SSE framing -----------------------------------------------------------


def sse_frame(event: str, data: dict) -> str:
    """Encode one SSE frame.

    Separators are compact and the JSON must stay on a single line: a newline
    inside a data: field silently truncates the frame at the client.
    """
    payload = json.dumps(
        _rewrite_image_urls(data), separators=(",", ":"), default=str
    )
    return f"event: {event}\ndata: {payload}\n\n"


# --- Chat ------------------------------------------------------------------


@app.post("/api/chat")
async def chat(
    message: str = Form(""),
    budget: float | None = Form(None),
    aesthetic: str = Form(""),
    session_id: str | None = Form(None),
    image: UploadFile | None = File(None),
    # Measurements. Supplying width and depth promotes the room from an
    # estimate to measured, which unlocks wall-hugging pieces the solver
    # otherwise withholds rather than guessing at.
    room_width_cm: float | None = Form(None),
    room_depth_cm: float | None = Form(None),
    # "Your estimate looks right." Promotes the room to CONFIRMED, which is
    # enough to release wall-hugging pieces without demanding a tape measure.
    # Send alone to accept the proposal as-is, or with edited dimensions.
    confirm_dimensions: bool = Form(False),
    irregular: bool = Form(False),
    # JSON array of openings, e.g.
    # [{"kind":"door","wall":"north","offset_cm":10,"width_cm":80,"swing_cm":80}]
    openings: str | None = Form(None),
) -> StreamingResponse:
    graph = STATE.get("graph")
    if graph is None:  # pragma: no cover - only before lifespan completes
        raise HTTPException(503, "server still starting")

    sid = session_id or f"s_{secrets.token_hex(6)}"
    history = SESSIONS.setdefault(sid, [])

    # Read and downscale the upload before streaming starts: an oversized or
    # undecodable file should fail as a normal HTTP error, not mid-stream where
    # the status line has already been sent.
    image_b64: str | None = None
    if image is not None:
        raw = await image.read()
        if len(raw) > config.MAX_IMAGE_BYTES:
            raise HTTPException(
                413,
                f"image exceeds {config.MAX_IMAGE_BYTES // (1024 * 1024)}MB limit",
            )
        if raw:
            image_b64 = prepare_image(raw)
            if image_b64 is None:
                raise HTTPException(400, "could not decode image file")

    budget_cents = (
        int(budget * 100) if budget and budget > 0 else config.DEFAULT_BUDGET_CENTS
    )

    # Parse measurements up front so malformed input fails as a normal HTTP
    # error rather than mid-stream.
    parsed_openings: list[Opening] = []
    if openings:
        try:
            raw_openings = json.loads(openings)
            parsed_openings = [Opening(**o) for o in raw_openings]
        except Exception as exc:
            raise HTTPException(400, f"invalid openings: {exc}") from exc

    measurements: dict[str, object] | None = None
    if room_width_cm and room_depth_cm:
        if room_width_cm <= 0 or room_depth_cm <= 0:
            raise HTTPException(400, "room dimensions must be positive")
        # confirm_dimensions marks "I looked at your estimate and it's about
        # right" - the same numbers, but now vouched for. Typed-in numbers
        # without that flag are a real measurement.
        measurements = {
            "width_cm": room_width_cm,
            "depth_cm": room_depth_cm,
            "dimension_source": (
                DimensionSource.CONFIRMED
                if confirm_dimensions
                else DimensionSource.MEASURED
            ),
            "irregular": irregular,
            "openings": parsed_openings,
        }
    elif confirm_dimensions:
        # Accepting the estimate without editing it: no numbers come back, so
        # promote whatever the last turn proposed.
        prior_dims = MEASUREMENTS.get(sid) or {}
        proposed = PREFERENCES.get(sid, {}).get("proposed_dimensions")
        if proposed:
            measurements = {
                **{k: v for k, v in proposed.items() if k in ("width_cm", "depth_cm")},
                "dimension_source": DimensionSource.CONFIRMED,
                "irregular": irregular or bool(prior_dims.get("irregular")),
                "openings": parsed_openings or prior_dims.get("openings") or [],
            }
        else:
            raise HTTPException(
                409, "nothing to confirm - no dimensions have been proposed yet"
            )
    elif parsed_openings or irregular:
        # Openings without dimensions still refine the room we estimated.
        measurements = {"irregular": irregular, "openings": parsed_openings}

    # Measurements persist for the session: having stated the room is 4.2×3.3m,
    # the user should not have to repeat it on every follow-up turn.
    if measurements:
        MEASUREMENTS[sid] = {**MEASUREMENTS.get(sid, {}), **measurements}

    history.append(ChatMessage(role="user", content=message))
    del history[: -config.MAX_HISTORY_MESSAGES]

    # Preferences carry forward. The form fields are authoritative only when
    # the caller actually sets them: a follow-up turn that omits budget or
    # aesthetic should keep what the conversation established, not silently
    # reset to the default.
    prefs = PREFERENCES.setdefault(sid, {})
    if budget is not None and budget > 0:
        prefs["budget_cents"] = budget_cents
    if aesthetic:
        prefs["aesthetic"] = aesthetic

    # The previous turn's design, so parse_intent can resolve "cheaper than
    # what" and "another one" rather than "which one".
    prior_ctx = RENDER_CONTEXT.get(sid) or {}
    prior_ids = prior_ctx.get("selected_ids") or []
    index = STATE.get("index")
    prior_items = [
        it
        for it in (index.get(i) for i in prior_ids)  # type: ignore[attr-defined]
        if it is not None
    ]
    prior_layout_raw = prior_ctx.get("layout")
    prior_layout = (
        LayoutResult(**prior_layout_raw)
        if isinstance(prior_layout_raw, dict)
        else prior_layout_raw
    )

    # Detections cached from a previous turn on this session, reusable only
    # while the photo they describe is still the current one. Detection is a
    # second vision call, and a follow-up like "make it cheaper" re-runs the
    # whole graph over an unchanged photo - paying for it again buys nothing.
    cached_detections = None
    if image_b64 and prior_ctx.get("detected_for") == _image_key(image_b64):
        raw_dets = prior_ctx.get("detections")
        if isinstance(raw_dets, list):
            cached_detections = [Detection(**d) for d in raw_dets]

    state: GraphState = {
        "messages": list(history),
        "image_b64": image_b64,
        "detections": cached_detections,
        "budget_cents": prefs.get("budget_cents", budget_cents),
        "aesthetic": prefs.get("aesthetic", aesthetic),
        "prompt": message,
        "measurements": MEASUREMENTS.get(sid) or None,
        "selected": prior_items,
        "layout": prior_layout,
        "rejected_ids": list(prefs.get("rejected_ids") or []),
        "excluded_roles": list(prefs.get("excluded_roles") or []),
        "size_caps": dict(prefs.get("size_caps") or {}),
    }

    # The photo is kept for rendering. A later turn with no new photo reuses
    # the one already on the session, so a user can refine a design without
    # re-uploading it every message.
    ctx = RENDER_CONTEXT
    if image_b64:
        ctx.setdefault(sid, {})["image_b64"] = image_b64

    async def stream() -> AsyncIterator[str]:
        started = time.perf_counter()
        transcript: list[str] = []
        # Flush something immediately so proxies commit the response headers
        # and the client can distinguish "connected" from "still waiting".
        yield ": connected\n\n"

        queue: asyncio.Queue[tuple[str, dict] | None] = asyncio.Queue()

        async def run() -> None:
            try:
                async for chunk in graph.astream(state, stream_mode="custom"):
                    kind = chunk.get("type", "text_delta")
                    await queue.put((kind, chunk))
            except Exception as exc:
                log.exception("graph failed")
                # Headers are long gone by now, so an error must travel as a
                # frame; raising here would truncate the stream instead.
                await queue.put(
                    ("error", {"message": str(exc), "code": "graph_error", "fatal": True})
                )
            finally:
                await queue.put(None)

        task = asyncio.create_task(run())
        try:
            while True:
                try:
                    item = await asyncio.wait_for(
                        queue.get(), timeout=config.SSE_HEARTBEAT_SECONDS
                    )
                except asyncio.TimeoutError:
                    # Keeps idle intermediaries from dropping the connection.
                    yield ": ping\n\n"
                    continue

                if item is None:
                    break
                kind, payload = item
                if kind == "text_delta":
                    transcript.append(payload.get("text", ""))
                # Capture what /api/render will need. Taken from the frames
                # rather than the graph's final state because stream_mode
                # "custom" yields writer output, not state, and these are the
                # same objects the client is being told about.
                elif kind == "room_analysis":
                    room_payload = payload.get("room") or {}
                    ctx.setdefault(sid, {})["room"] = room_payload
                    # Cache the detections against the photo they came from, so
                    # a follow-up turn over the same image skips the second
                    # vision call. Keyed by content, not by session, because a
                    # new photo must invalidate them.
                    if image_b64:
                        ctx[sid]["detections"] = room_payload.get("detections") or []
                        ctx[sid]["detected_for"] = _image_key(image_b64)
                elif kind == "layout_update":
                    ctx.setdefault(sid, {})["layout"] = payload.get("layout")
                elif kind == "alternatives":
                    # Kept so /api/swap can validate a chosen item against what
                    # was actually offered, rather than trusting a client id.
                    ctx.setdefault(sid, {})["options"] = payload.get("options")
                elif kind == "cart_update":
                    ctx.setdefault(sid, {})["selected_ids"] = [
                        line["item_id"] for line in payload.get("cart", {}).get("lines", [])
                    ]
                    ctx.setdefault(sid, {})["budget_cents"] = payload.get("budget_cents")
                elif kind == "dimension_proposal":
                    # Remember what we offered, so a bare confirm_dimensions on
                    # the next turn can promote these numbers without the client
                    # having to echo them back.
                    prefs["proposed_dimensions"] = payload.get("proposal") or {}
                elif kind == "intent":
                    # Persist what the turn established so the next one inherits
                    # it: a rejected sofa stays rejected, a lowered budget stays
                    # lowered, without the client having to resend either.
                    parsed = payload.get("intent") or {}
                    if parsed.get("budget_cents"):
                        prefs["budget_cents"] = parsed["budget_cents"]
                    if parsed.get("aesthetic"):
                        prefs["aesthetic"] = parsed["aesthetic"]
                    if parsed.get("reject_item_ids"):
                        prefs["rejected_ids"] = sorted(
                            {*(prefs.get("rejected_ids") or []), *parsed["reject_item_ids"]}
                        )
                    if parsed.get("remove_roles"):
                        prefs["excluded_roles"] = sorted(
                            {*(prefs.get("excluded_roles") or []), *parsed["remove_roles"]}
                        )
                    if parsed.get("max_width_cm"):
                        prefs["size_caps"] = {
                            **(prefs.get("size_caps") or {}),
                            **parsed["max_width_cm"],
                        }
                    # A reroll rejects the item currently in that role, so the
                    # next search cannot return the same piece.
                    if parsed.get("reroll_roles"):
                        rolled = {
                            it.id
                            for it in prior_items
                            if it.role.value in parsed["reroll_roles"]
                        }
                        if rolled:
                            prefs["rejected_ids"] = sorted(
                                {*(prefs.get("rejected_ids") or []), *rolled}
                            )
                yield sse_frame(kind, payload)
        finally:
            task.cancel()

        if transcript:
            history.append(ChatMessage(role="assistant", content="".join(transcript)))
            del history[: -config.MAX_HISTORY_MESSAGES]

        # `done` always fires, including after a fatal error, so a client can
        # unconditionally re-enable its composer when the stream ends.
        yield sse_frame(
            "done",
            {
                "session_id": sid,
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
            },
        )

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            # Tells nginx and friends not to buffer, which would otherwise
            # withhold every frame until the response completed.
            "X-Accel-Buffering": "no",
        },
    )


# --- Render ----------------------------------------------------------------


@app.post("/api/render")
async def render(req: RenderRequest) -> StreamingResponse:
    """Visualize the placed items in the user's own room photo.

    A separate endpoint rather than a graph node: a generative render is tens
    of seconds per item, and folding that into /api/chat would delay the floor
    plan and cart by minutes for a picture the user may not have asked for.

    Streams the same SSE framing as /api/chat, one `render_update` per item as
    it completes, in painter's order so a client can composite on arrival.
    """
    renderer = STATE.get("renderer")
    index = STATE.get("index")
    if renderer is None or index is None:  # pragma: no cover
        raise HTTPException(503, "server still starting")

    ctx = RENDER_CONTEXT.get(req.session_id)
    if not ctx:
        raise HTTPException(
            404,
            "unknown session; POST /api/chat first to produce a design to render",
        )

    room_raw, layout_raw = ctx.get("room"), ctx.get("layout")
    if not room_raw or not layout_raw:
        raise HTTPException(409, "this session has no completed design yet")

    # Rehydrate through the models so a malformed cache entry fails here as a
    # 4xx rather than deep inside the renderer mid-stream.
    try:
        room = RoomAnalysis(**room_raw)
        layout = LayoutResult(**layout_raw)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(500, f"corrupt session design: {exc}") from exc

    image_b64 = ctx.get("image_b64")
    items = {i.id: i for i in SEED_ITEMS}

    # A composing backend renders the whole room in one call, which is both
    # cheaper and more coherent than six masked inpaints - and every piece
    # comes from its real product photo. Fall back to per-item when there is
    # no composer, or when the caller explicitly asks for it.
    compose = renderer.can_compose and not req.per_item

    async def stream() -> AsyncIterator[str]:
        started = time.perf_counter()
        yield ": connected\n\n"

        total = (
            1
            if compose
            else len(
                [
                    p
                    for p in layout.placements
                    if not req.item_ids or p.item_id in req.item_ids
                ]
            )
        )
        yield sse_frame(
            "render_started",
            {
                "total": total,
                "method": "composed" if compose else renderer.method.value,
                "erased": bool(image_b64) and renderer.method.value == "generative",
            },
        )

        if compose:
            result = await render_room(
                renderer,
                image_b64,
                room,
                layout,
                items,
                only=req.item_ids or None,
            )
            event = "room_render" if isinstance(result, RoomRender) else "render_failed"
            payload = result.model_dump(mode="json")
            if isinstance(result, RoomRender):
                payload["progress"] = {"done": 1, "total": 1}
            yield sse_frame(event, payload)
            yield sse_frame(
                "done",
                {
                    "session_id": req.session_id,
                    "rendered": 1 if isinstance(result, RoomRender) else 0,
                    "total": 1,
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                },
            )
            return

        queue: asyncio.Queue[tuple[str, dict] | None] = asyncio.Queue()

        async def run() -> None:
            try:
                async for result in render_layout(
                    renderer,
                    image_b64,
                    room,
                    layout,
                    items,
                    only=req.item_ids or None,
                ):
                    event = (
                        "render_update"
                        if isinstance(result, RenderResult)
                        else "render_failed"
                    )
                    await queue.put((event, result.model_dump(mode="json")))
            except Exception as exc:
                log.exception("render failed")
                await queue.put(
                    ("error", {"message": str(exc), "code": "render_error", "fatal": True})
                )
            finally:
                await queue.put(None)

        task = asyncio.create_task(run())
        done_count = 0
        try:
            while True:
                try:
                    item = await asyncio.wait_for(
                        queue.get(), timeout=config.SSE_HEARTBEAT_SECONDS
                    )
                except asyncio.TimeoutError:
                    # Renders are slow enough that heartbeats are load-bearing
                    # here, not just defensive as they are on /api/chat.
                    yield ": ping\n\n"
                    continue
                if item is None:
                    break
                kind, payload = item
                if kind == "render_update":
                    done_count += 1
                    payload["progress"] = {"done": done_count, "total": total}
                yield sse_frame(kind, payload)
        finally:
            task.cancel()

        yield sse_frame(
            "done",
            {
                "session_id": req.session_id,
                "rendered": done_count,
                "total": total,
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
            },
        )

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# --- Swap ------------------------------------------------------------------


@app.post("/api/swap")
async def swap(req: SwapRequest) -> StreamingResponse:
    """Replace one item with an alternative, re-solve, and re-render.

    A swap is never merely a re-render. A different sofa has different
    dimensions, so the layout must be re-solved - the replacement may not fit,
    or may displace the coffee table that was positioned against the old one -
    and the cart re-billed. Returning a new picture over a stale plan would
    show the user something the solver never agreed to.
    """
    renderer = STATE.get("renderer")
    index = STATE.get("index")
    if renderer is None or index is None:  # pragma: no cover
        raise HTTPException(503, "server still starting")

    ctx = RENDER_CONTEXT.get(req.session_id)
    if not ctx:
        raise HTTPException(404, "unknown session; POST /api/chat first")

    room_raw = ctx.get("room")
    selected_ids = ctx.get("selected_ids")
    if not room_raw or selected_ids is None:
        raise HTTPException(409, "this session has no completed design yet")

    replacement = index.get(req.item_id)  # type: ignore[attr-defined]
    if replacement is None:
        raise HTTPException(404, f"unknown item: {req.item_id}")
    if replacement.role is not req.role:
        raise HTTPException(
            400,
            f"{req.item_id} is a {replacement.role.value}, not a {req.role.value}",
        )

    try:
        room = RoomAnalysis(**room_raw)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(500, f"corrupt session design: {exc}") from exc

    # Substitute by role. The outgoing item leaves the design entirely, so a
    # swap never grows the piece count - it exchanges one for one.
    items = {i.id: i for i in SEED_ITEMS}
    current = [items[i] for i in selected_ids if i in items]
    outgoing = next((i for i in current if i.role is req.role), None)
    if outgoing is None:
        raise HTTPException(
            409, f"nothing of role {req.role.value} is currently in the design"
        )
    if outgoing.id == replacement.id:
        raise HTTPException(400, "that item is already selected")

    updated = [replacement if i.role is req.role else i for i in current]

    budget_cents = int(ctx.get("budget_cents") or config.DEFAULT_BUDGET_CENTS)
    image_b64 = ctx.get("image_b64")

    async def stream() -> AsyncIterator[str]:
        started = time.perf_counter()
        yield ": connected\n\n"
        yield sse_frame(
            "swap_started",
            {
                "role": req.role.value,
                "from": {"item_id": outgoing.id, "name": outgoing.title},
                "to": {"item_id": replacement.id, "name": replacement.title},
            },
        )

        # Re-solve. The replacement's footprint differs, so positions move and
        # a piece that no longer fits is reported rather than forced.
        layout = LayoutSolver(room).solve(updated)
        yield sse_frame(
            "layout_update", {"type": "layout_update", "layout": layout.model_dump(mode="json")}
        )

        placed_ids = {p.item_id for p in layout.placements}
        lines = [
            CartLine(
                item_id=i.id,
                name=i.title,
                merchant=i.merchant,
                role=i.role,
                price_cents=i.price_cents,
                checkout_url=i.checkout_url,
                image_url=i.image_url,
            )
            for i in updated
            if i.id in placed_ids
        ]
        cart = Cart(
            lines=lines,
            subtotal_cents=sum(line.line_total_cents for line in lines),
            budget_cents=budget_cents,
        )
        yield sse_frame(
            "cart_update",
            {
                "type": "cart_update",
                "cart": cart.model_dump(mode="json"),
                "subtotal_cents": cart.subtotal_cents,
                "budget_cents": cart.budget_cents,
                "over_budget": cart.over_budget,
            },
        )

        # The swap is committed to the session before rendering, so a failed or
        # skipped render still leaves the design in its new state.
        ctx["selected_ids"] = [i.id for i in updated]
        ctx["layout"] = layout.model_dump(mode="json")

        # Offer the alternatives again, now priced against the new pick.
        alternatives = _reprice_options(ctx.get("options"), updated, budget_cents, cart)
        if alternatives:
            yield sse_frame("alternatives", {"options": alternatives})

        if not req.layout_only:
            result = await render_room(renderer, image_b64, room, layout, items)
            event = "room_render" if isinstance(result, RoomRender) else "render_failed"
            yield sse_frame(event, result.model_dump(mode="json"))

        yield sse_frame(
            "done",
            {
                "session_id": req.session_id,
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
            },
        )

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _reprice_options(
    stored: object, selected: list, budget_cents: int, cart: Cart
) -> list[dict]:
    """Re-point the stored alternatives at the new selection.

    Deltas and affordability were computed against the item that just left the
    design, so leaving them untouched would price every option against a piece
    the user no longer has.
    """
    if not isinstance(stored, list):
        return []

    by_role = {i.role.value: i for i in selected}
    spent = cart.subtotal_cents
    out: list[dict] = []
    for entry in stored:
        if not isinstance(entry, dict):
            continue
        chosen = by_role.get(entry.get("role"))
        if chosen is None:
            continue
        headroom = budget_cents - spent + chosen.price_cents
        alts = [
            {
                **alt,
                "price_delta_cents": alt["price_cents"] - chosen.price_cents,
                "affordable": alt["price_cents"] <= headroom,
            }
            for alt in entry.get("alternatives", [])
            if alt.get("item_id") != chosen.id
        ]
        # The outgoing item becomes an alternative itself - a user who dislikes
        # the swap needs a way back without replaying the conversation. It is
        # listed first, because undo is the most likely next click.
        outgoing_id = entry.get("selected_id")
        if outgoing_id and outgoing_id != chosen.id:
            if not any(a["item_id"] == outgoing_id for a in alts):
                previous = next(
                    (i for i in SEED_ITEMS if i.id == outgoing_id), None
                )
                if previous is not None:
                    alts.insert(
                        0,
                        {
                            "item_id": previous.id,
                            "name": previous.title,
                            "merchant": previous.merchant,
                            "role": previous.role.value,
                            "price_cents": previous.price_cents,
                            "price_delta_cents": previous.price_cents - chosen.price_cents,
                            "swatch": previous.swatch,
                            "image_url": previous.image_url,
                            "materials": previous.materials,
                            "primary_color": previous.primary_color,
                            "style_tags": previous.style_tags,
                            "width_cm": previous.dimensions.width_cm,
                            "depth_cm": previous.dimensions.depth_cm,
                            "height_cm": previous.dimensions.height_cm,
                            "affordable": previous.price_cents <= headroom,
                        },
                    )
        entry["selected_id"] = chosen.id
        entry["alternatives"] = alts
        out.append(
            {"role": entry["role"], "selected_id": chosen.id, "alternatives": alts}
        )
    return out


# --- Checkout (simulated) --------------------------------------------------


@app.post("/api/checkout/simulate", response_model=CheckoutResult)
async def checkout_simulate(req: CheckoutRequest) -> CheckoutResult:
    """Fabricate a multi-merchant confirmation. No payment is processed."""
    index = STATE.get("index")
    if index is None:  # pragma: no cover
        raise HTTPException(503, "server still starting")

    if not req.item_ids:
        raise HTTPException(400, "no items to check out")

    lines: list[CartLine] = []
    for item_id in req.item_ids:
        item = index.get(item_id)  # type: ignore[attr-defined]
        if item is None:
            raise HTTPException(404, f"unknown item: {item_id}")
        lines.append(
            CartLine(
                item_id=item.id,
                name=item.title,
                merchant=item.merchant,
                role=item.role,
                price_cents=item.price_cents,
                checkout_url=item.checkout_url,
                # Browser-loadable; this response is rendered directly by the
                # client rather than passing through sse_frame.
                image_url=_http_image_url(item.image_url),
            )
        )

    grouped: dict[str, list[CartLine]] = {}
    for line in lines:
        grouped.setdefault(line.merchant, []).append(line)

    groups = [
        MerchantGroup(
            merchant=merchant,
            lines=merchant_lines,
            subtotal_cents=sum(line.line_total_cents for line in merchant_lines),
        )
        for merchant, merchant_lines in sorted(grouped.items())
    ]

    return CheckoutResult(
        order_id=f"SIM-{secrets.token_hex(4).upper()}",
        payment_token=f"tok_sim_{secrets.token_hex(8)}",
        groups=groups,
        total_cents=sum(g.subtotal_cents for g in groups),
    )


# --- Payment (simulated Visa rail) -----------------------------------------
#
# The authorization model, in one line: the agent may PRICE a purchase; only
# the user may AUTHORIZE one. Everything below enforces that split.
#
#   GET  /api/payment/methods    the user's stored cards
#   POST /api/payment/intent     agent prices it -> preview, charges nothing
#   POST /api/payment/verify/start   issue a step-up challenge
#   POST /api/payment/verify     answer it
#   POST /api/payment/authorize  the user releases the charge
#   POST /api/payment/cancel     the user declines
#   GET  /api/payment/intent/{id}  re-read a preview


def _payment_error(exc: payments.PaymentError) -> HTTPException:
    return HTTPException(exc.status, exc.detail)


@app.get("/api/payment/methods", response_model=list[PaymentMethod])
async def payment_methods() -> list[PaymentMethod]:
    """The cards on file. Fictional; only last four digits ever exist."""
    return payments.wallet()


@app.post("/api/payment/intent", response_model=PaymentIntent)
async def payment_intent(req: PaymentIntentRequest) -> PaymentIntent:
    """Price a purchase into a preview. NOTHING IS CHARGED HERE.

    This is the boundary of the agent's authority. It resolves every item
    against the live catalog rather than trusting prices the client sends,
    because a client-supplied total is a total the user could be shown while a
    different one is charged.
    """
    index = STATE.get("index")
    if index is None:  # pragma: no cover
        raise HTTPException(503, "server still starting")
    if not req.item_ids:
        raise HTTPException(400, "no items to pay for")

    lines: list[CartLine] = []
    for item_id in req.item_ids:
        item = index.get(item_id)  # type: ignore[attr-defined]
        if item is None:
            raise HTTPException(404, f"unknown item: {item_id}")
        if not item.in_stock:
            raise HTTPException(409, f"out of stock: {item.title}")
        lines.append(
            CartLine(
                item_id=item.id,
                name=item.title,
                merchant=item.merchant,
                role=item.role,
                price_cents=item.price_cents,
                checkout_url=item.checkout_url,
                image_url=_http_image_url(item.image_url),
            )
        )

    # The budget the design was actually solved against, so the preview can say
    # whether this purchase honours the constraint the user set.
    budget_cents = 0
    if req.session_id:
        prefs = PREFERENCES.get(req.session_id) or {}
        budget_cents = int(prefs.get("budget_cents") or 0)

    try:
        return payments.create_intent(
            lines,
            session_id=req.session_id,
            budget_cents=budget_cents,
            payment_method_ids=req.payment_method_ids,
            initiated_by="agent",
        )
    except payments.PaymentError as exc:
        raise _payment_error(exc) from exc


@app.get("/api/payment/intent/{intent_id}", response_model=PaymentIntent)
async def payment_intent_read(intent_id: str) -> PaymentIntent:
    try:
        return payments.get_intent(intent_id)
    except payments.PaymentError as exc:
        raise _payment_error(exc) from exc


@app.post("/api/payment/verify/start", response_model=VerificationChallenge)
async def payment_verify_start(body: dict) -> VerificationChallenge:
    """Issue a step-up challenge, in the shape of a 3-D Secure prompt."""
    intent_id = str(body.get("intent_id") or "")
    if not intent_id:
        raise HTTPException(400, "intent_id required")
    try:
        return payments.start_verification(intent_id)
    except payments.PaymentError as exc:
        raise _payment_error(exc) from exc


@app.post("/api/payment/verify", response_model=PaymentIntent)
async def payment_verify(req: VerifyRequest) -> PaymentIntent:
    """Answer a step-up challenge. Proves a human is present."""
    try:
        return payments.verify(req.intent_id, req.challenge_id, req.code)
    except payments.PaymentError as exc:
        raise _payment_error(exc) from exc


@app.post("/api/payment/authorize", response_model=AuthorizationReceipt)
async def payment_authorize(req: AuthorizeRequest) -> AuthorizationReceipt:
    """The user releases the charge. The only endpoint that moves money.

    Refuses if the total has drifted from what the user reviewed, if step-up
    was required and not completed, or if the preview has expired. Replaying
    an idempotency key returns the original receipt rather than charging twice.
    """
    if not req.idempotency_key:
        raise HTTPException(400, "idempotency_key required")
    try:
        return payments.authorize(
            req.intent_id, req.idempotency_key, req.confirmed_total_cents
        )
    except payments.PaymentError as exc:
        raise _payment_error(exc) from exc


@app.post("/api/payment/cancel", response_model=PaymentIntent)
async def payment_cancel(body: dict) -> PaymentIntent:
    """Decline the purchase. Always available while nothing is charged."""
    intent_id = str(body.get("intent_id") or "")
    if not intent_id:
        raise HTTPException(400, "intent_id required")
    try:
        return payments.cancel_intent(intent_id)
    except payments.PaymentError as exc:
        raise _payment_error(exc) from exc



# --- Health ----------------------------------------------------------------


@app.get("/api/health")
async def health() -> dict:
    index = STATE.get("index")
    renderer = STATE.get("renderer")
    vision = STATE.get("vision")
    return {
        "status": "ok" if index is not None else "starting",
        "providers": "openai" if config.HAS_OPENAI else "mock",
        "offline_mode": not config.HAS_OPENAI,
        "catalog_seeded": index is not None,
        # Broken out per provider, because "providers: openai" only says a key
        # was present - it does not say which subsystems actually built a
        # client. A single rolled-up flag sends you to the wrong place when one
        # of them silently falls back.
        "vision": {
            "source": getattr(vision, "source", "unavailable"),
            "model": config.VISION_MODEL if config.HAS_OPENAI else None,
            # Vision only runs when a photo is supplied. Without one the room
            # comes back with source "default", which is not a fallback.
            "requires_photo": True,
        },
        "intent": {
            "source": getattr(STATE.get("intent"), "source", "unavailable"),
            "model": config.INTENT_MODEL if config.HAS_OPENAI else None,
        },
        "embeddings": {
            "source": getattr(getattr(index, "embedder", None), "source", "unavailable"),
            "model": config.EMBEDDING_MODEL if config.HAS_OPENAI else None,
        },
        "renderer": {
            "source": getattr(renderer, "source", "unavailable"),
            "method": getattr(getattr(renderer, "method", None), "value", "unavailable"),
            "can_compose": bool(getattr(renderer, "can_compose", False)),
            "compose_model": config.GEMINI_IMAGE_MODEL if config.HAS_GEMINI else None,
        },
    }


@app.post("/api/detect")
async def detect(
    image: UploadFile = File(...),
    # Off by default: detection alone is fast and cheap, while matching costs
    # one embedding call per object found.
    match: bool = Form(False),
    limit: int = Form(config.REVERSE_SEARCH_LIMIT),
) -> dict:
    """Identify the furniture in a room photo, and optionally price it.

    Two things in one call because they answer one question: "what is in this
    room, and can I buy it here?" The boxes are normalized [0,1] so a client
    can draw them over the photo at any display size.

    Reverse search is a *suggestion*, not an identification: see `confident` on
    each result for whether the top match actually resembles the object or is
    merely the closest thing in a small catalog.
    """
    index = STATE.get("index")
    detector = STATE.get("detector")
    if index is None or detector is None:  # pragma: no cover - pre-lifespan
        raise HTTPException(503, "server still starting")

    raw = await image.read()
    if len(raw) > config.MAX_IMAGE_BYTES:
        raise HTTPException(
            413, f"image exceeds {config.MAX_IMAGE_BYTES // (1024 * 1024)}MB limit"
        )
    if not raw:
        raise HTTPException(400, "empty image file")
    image_b64 = prepare_image(raw)
    if image_b64 is None:
        raise HTTPException(400, "could not decode image file")

    # 503 rather than an empty list: "we cannot look" and "we looked and found
    # nothing" are different answers, and a client that conflated them would
    # tell the user their room is empty.
    if not detector.available:  # type: ignore[attr-defined]
        raise HTTPException(
            503,
            "furniture detection needs OPENAI_API_KEY; the server is running "
            "with offline providers",
        )

    found = await asyncio.to_thread(detector.detect, image_b64)  # type: ignore[attr-defined]
    if not match:
        return {
            "count": len(found),
            "detections": [d.model_dump(mode="json") for d in found],
        }

    limit = max(1, min(limit, 20))
    results = await asyncio.to_thread(index.identify_room, found, limit)  # type: ignore[attr-defined]
    return {
        "count": len(results),
        "results": [
            _rewrite_image_urls(r.model_dump(mode="json")) for r in results
        ],
    }


@app.get("/api/catalog")
async def catalog(limit: int = 100) -> dict:
    """Inspect the seeded catalog. Useful when debugging retrieval."""
    from .seed_data import SEED_ITEMS

    return {
        "count": len(SEED_ITEMS),
        "items": [
            _rewrite_image_urls(i.model_dump(mode="json")) for i in SEED_ITEMS[:limit]
        ],
    }
