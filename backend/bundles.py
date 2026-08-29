"""Bundle and set recommendations, built only from what the catalog knows.

WHAT THIS IS NOT. There is no "customers also bought" here. The catalog is a
product scrape - attributes, prices, dimensions - with no order history, no
baskets and no view logs, so no co-purchase signal exists to compute. Any
number claiming otherwise would be fabricated, and fabricated social proof is
worse than none: it pushes real spending decisions with invented evidence.

WHAT IT IS INSTEAD. Three bases, each a checkable property of the products:

  SAME_SERIES     Both pieces belong to one IKEA product line (LANDSKRONA,
                  SODERHAMN, ...). Same frame, fabric and proportions - a
                  matching set in the literal sense. This is the strongest
                  signal in the data and is ranked first.

  STYLE_MATCH     Different lines that share style tags and sit in a
                  compatible colour family. Weaker, honest, and the only way
                  to suggest across roles the series do not cover.

  COMPLETES_ROOM  A role missing from the design entirely, filled by the piece
                  that best matches what is already chosen.

Every bundle is also checked against the room: a suggestion that cannot
physically fit next to what is already placed is flagged rather than silently
offered as a drop-in.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from . import config
from .models import (
    Bundle,
    BundleBasis,
    BundleItem,
    CatalogItem,
    LayoutResult,
    PLACEMENT_ORDER,
    ROLE_COUNTS,
    Role,
    RoomAnalysis,
    _BASIS_LABEL,
)

log = logging.getLogger(__name__)

# Roles that read as a set when they share a series. A sofa and its matching
# armchair are a recognisable pairing; a sofa and a lamp from the same line
# are not, even when the series technically spans both.
_SET_PAIRS: dict[Role, tuple[Role, ...]] = {
    Role.SOFA: (Role.ACCENT_CHAIR,),
    Role.ACCENT_CHAIR: (Role.SOFA,),
}

# Colour families, so "goes with" is not decided by string equality. Colours
# inside one group sit together without clashing; this is a design judgement,
# deliberately coarse, and only ever used to *rank* suggestions.
_COLOUR_FAMILIES: tuple[frozenset[str], ...] = (
    frozenset({"white", "off-white", "beige", "grey-beige", "greige", "natural",
               "birch", "oak", "light grey", "cream"}),
    frozenset({"grey", "dark grey", "anthracite", "black", "black-blue",
               "silver-colour", "gunmetal"}),
    frozenset({"brown", "dark brown", "light brown", "walnut", "pine",
               "red-brown", "brown-red", "rattan", "bamboo"}),
    frozenset({"blue", "dark blue", "light blue", "turquoise", "green",
               "dark green", "light green"}),
    frozenset({"red", "dark red", "orange", "yellow", "dark yellow", "pink",
               "light pink", "purple", "violet"}),
)

MAX_BUNDLES = 4


def _style_agrees(a: CatalogItem, b: CatalogItem) -> bool:
    """Whether two pieces share enough styling to suggest together.

    The bar adapts to how much vocabulary each piece actually has. Style tags
    are inferred and deliberately sparse - 96 of 175 items carry exactly one -
    so a flat "two shared tags" rule is unsatisfiable for most of the catalog
    and silently produces no bundles at all. Instead: require two shared tags
    when both pieces have two or more to give, and one when either is
    single-tagged, which is the most agreement that piece can express.
    """
    shared = set(a.style_tags) & set(b.style_tags)
    if not shared:
        return False
    need = 2 if min(len(a.style_tags), len(b.style_tags)) >= 2 else 1
    return len(shared) >= need


def _colour_family(colour: str) -> frozenset[str] | None:
    c = (colour or "").strip().lower()
    for family in _COLOUR_FAMILIES:
        if c in family:
            return family
    return None


def _colours_harmonise(a: str, b: str) -> bool:
    """Whether two colours sit in the same family.

    Neutrals are handled by membership rather than a special case: the first
    family is the neutral one, so a beige sofa and an oak table match through
    it without needing an exception.
    """
    fa, fb = _colour_family(a), _colour_family(b)
    return fa is not None and fa is fb


def _shared_tags(a: CatalogItem, b: CatalogItem) -> int:
    return len(set(a.style_tags) & set(b.style_tags))


def _fits_alongside(
    item: CatalogItem, room: RoomAnalysis, layout: LayoutResult | None
) -> bool:
    """Whether the room plausibly has floor area left for this piece.

    Deliberately an area check rather than a re-solve. A real answer requires
    running the solver, which the swap endpoint already does when the user
    commits; doing it for every candidate bundle would cost far more than the
    suggestion is worth. So this is a cheap necessary condition - it rules out
    the obviously impossible and never claims a placement is guaranteed.
    """
    if room is None:
        return True
    usable = max(0.0, room.width_cm - 2 * config.WALL_MARGIN_CM) * max(
        0.0, room.depth_cm - 2 * config.WALL_MARGIN_CM
    )
    if usable <= 0:
        return False
    taken = 0.0
    if layout is not None:
        for p in layout.placements:
            taken += p.w_cm * p.d_cm
    footprint = item.dimensions.width_cm * item.dimensions.depth_cm
    # Furniture never fills a room wall to wall; past roughly two thirds the
    # room stops being walkable, so that is treated as full.
    return (taken + footprint) <= usable * 0.66


def _bundle_item(item: CatalogItem, owned: set[str]) -> BundleItem:
    return BundleItem(
        item_id=item.id,
        name=item.title,
        role=item.role,
        price_cents=item.price_cents,
        image_url=item.image_url,
        swatch=item.swatch,
        series=item.series,
        is_new=item.id not in owned,
    )


def _make(
    bundle_id: str,
    basis: BundleBasis,
    reason: str,
    members: list[CatalogItem],
    owned: set[str],
    budget_cents: int,
    spent_cents: int,
    room: RoomAnalysis | None,
    layout: LayoutResult | None,
) -> Bundle:
    items = [_bundle_item(m, owned) for m in members]
    total = sum(m.price_cents for m in members)
    added = sum(m.price_cents for m in members if m.id not in owned)
    fits = all(
        _fits_alongside(m, room, layout) for m in members if m.id not in owned
    ) if room is not None else True
    return Bundle(
        id=bundle_id,
        basis=basis,
        label=_BASIS_LABEL[basis.value],
        reason=reason,
        items=items,
        total_cents=total,
        added_cents=added,
        currency=members[0].currency if members else "SGD",
        # Budget is judged on the *added* spend: the pieces already in the cart
        # are paid for in the running total and must not be charged twice.
        affordable=(spent_cents + added) <= budget_cents if budget_cents else True,
        fits_room=fits,
    )


def build_bundles(
    selected: list[CatalogItem],
    catalog: list[CatalogItem],
    *,
    budget_cents: int = 0,
    spent_cents: int = 0,
    room: RoomAnalysis | None = None,
    layout: LayoutResult | None = None,
    limit: int = MAX_BUNDLES,
) -> list[Bundle]:
    """Suggest sets that extend the current design.

    Ranked strongest-basis-first: a real matching series beats a style
    inference, and both beat merely filling an empty role.
    """
    if not selected:
        return []

    owned = {i.id for i in selected}
    by_id = {i.id: i for i in catalog}
    in_stock = [i for i in catalog if i.in_stock and i.id not in owned]

    by_series: dict[str, list[CatalogItem]] = defaultdict(list)
    for item in in_stock:
        if item.series:
            by_series[item.series].append(item)

    bundles: list[Bundle] = []
    seen: set[frozenset[str]] = set()

    # Roles that must not be suggested again. A role is closed once the design
    # holds as many as the role allows: offering a second sofa as an "addition"
    # is wrong - that is a swap, which /api/swap already does - but a second
    # armchair in a room with four seats' worth of floor is a real suggestion,
    # so roles that scale stay open until their ceiling.
    #
    # Extended as suggestions are accepted, so a matching ROCKSJÖN armchair and
    # a generic style-matched armchair do not both appear.
    role_counts: dict[Role, int] = defaultdict(int)
    for item in selected:
        role_counts[item.role] += 1
    suggested_roles: set[Role] = {
        role for role, n in role_counts.items()
        if n >= ROLE_COUNTS.get(role, (0, 1))[1]
    }

    def add(bundle: Bundle) -> None:
        key = frozenset(i.item_id for i in bundle.items)
        if key in seen or len(bundle.items) < 2:
            return
        new_roles = {i.role for i in bundle.items if i.is_new}
        if new_roles & suggested_roles:
            return
        seen.add(key)
        suggested_roles.update(new_roles)
        bundles.append(bundle)

    # 1. Matching sets: a chosen piece plus another from the same series.
    for chosen in selected:
        if not chosen.series:
            continue
        for partner_role in _SET_PAIRS.get(chosen.role, ()):
            partners = [
                c for c in by_series.get(chosen.series, ())
                if c.role is partner_role
            ]
            if not partners:
                continue
            partner = min(partners, key=lambda c: c.price_cents)
            add(_make(
                f"set-{chosen.id}-{partner.id}",
                BundleBasis.SAME_SERIES,
                f"{partner.title.split(' - ')[0]} is from the same "
                f"{chosen.series} range as your {chosen.role.value.replace('_', ' ')}, "
                f"so the frame and fabric match.",
                [chosen, partner],
                owned, budget_cents, spent_cents, room, layout,
            ))

    # 2. Style matches: fill a missing role with something that suits the
    #    anchor piece. The sofa anchors when there is one - it is the piece a
    #    room is built around.
    anchor = next((i for i in selected if i.role is Role.SOFA), selected[0])
    # Roles with room left, not merely roles at zero: a design with one chair
    # in a room that seats four still has a gap worth filling.
    missing = [
        r for r in PLACEMENT_ORDER
        if role_counts[r] < ROLE_COUNTS.get(r, (0, 1))[1]
    ]
    for role in missing:
        candidates = [
            c for c in in_stock
            if c.role is role
            and _style_agrees(c, anchor)
            and _colours_harmonise(c.primary_color, anchor.primary_color)
        ]
        if not candidates:
            continue
        # Cheapest qualifying piece: a suggestion the user has not asked for
        # should not be the most expensive thing on the page.
        pick = min(candidates, key=lambda c: c.price_cents)
        shared = sorted(set(pick.style_tags) & set(anchor.style_tags))
        label = role.value.replace("_", " ")
        # A role at zero is a gap; a role below its ceiling is an addition.
        # Telling someone their room "has no armchair" when it has one reads
        # as a bug in the recommendation, not a suggestion.
        opener = (
            f"Your room has no {label}."
            if role_counts[role] == 0
            else f"There is floor for another {label}."
        )
        add(_make(
            f"style-{anchor.id}-{pick.id}",
            BundleBasis.COMPLETES_ROOM,
            f"{opener} This one shares "
            f"{' and '.join(shared)} styling with your "
            f"{anchor.role.value.replace('_', ' ')} and sits in the same colour family.",
            [anchor, pick],
            owned, budget_cents, spent_cents, room, layout,
        ))

    # 3. Style pairs across roles already filled, for users who want an
    #    alternative pairing rather than a gap filled.
    if len(bundles) < limit:
        for chosen in selected:
            for cand in in_stock:
                if cand.role is chosen.role:
                    continue
                if not _style_agrees(cand, chosen):
                    continue
                if not _colours_harmonise(cand.primary_color, chosen.primary_color):
                    continue
                shared = sorted(set(cand.style_tags) & set(chosen.style_tags))
                add(_make(
                    f"pair-{chosen.id}-{cand.id}",
                    BundleBasis.STYLE_MATCH,
                    f"Shares {' and '.join(shared)} styling with your "
                    f"{chosen.role.value.replace('_', ' ')}, in a matching colour family.",
                    [chosen, cand],
                    owned, budget_cents, spent_cents, room, layout,
                ))
                break

    # Strongest basis first, then affordable and fitting ones, then cheapest.
    rank = {
        BundleBasis.SAME_SERIES: 0,
        BundleBasis.COMPLETES_ROOM: 1,
        BundleBasis.STYLE_MATCH: 2,
    }
    bundles.sort(
        key=lambda b: (rank[b.basis], not b.affordable, not b.fits_room, b.added_cents)
    )
    return bundles[:limit]


if __name__ == "__main__":
    from .seed_data import SEED_ITEMS

    by_role: dict[Role, list[CatalogItem]] = defaultdict(list)
    for it in SEED_ITEMS:
        by_role[it.role].append(it)

    # A partial design, so the "missing role" path is exercised too.
    chosen = [
        min(by_role[Role.SOFA], key=lambda i: i.price_cents),
        min(by_role[Role.RUG], key=lambda i: i.price_cents),
    ]
    print("design:")
    for c in chosen:
        print(f"  {c.role.value:<14} {c.title[:44]}  S${c.price_cents/100:,.2f}")

    out = build_bundles(
        chosen, SEED_ITEMS, budget_cents=200_000,
        spent_cents=sum(c.price_cents for c in chosen),
    )
    print(f"\n{len(out)} bundles:")
    for b in out:
        flags = "" if b.affordable else "  [over budget]"
        flags += "" if b.fits_room else "  [may not fit]"
        print(f"\n  [{b.label}] +S${b.added_cents/100:,.2f}{flags}")
        print(f"    {b.reason}")
        for i in b.items:
            mark = "+" if i.is_new else " "
            print(f"    {mark} {i.role.value:<14} {i.name[:46]}")
