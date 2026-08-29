"""Deterministic spatial layout solver.

All geometry is in centimetres with the origin at the room's top-left corner,
x increasing right (width) and y increasing down (depth). No LLM produces
coordinates: placement is rule-based so it always terminates and never emits an
overlapping or out-of-bounds layout.

The core invariants, asserted by selftest():
  1. Every placement lies fully inside the room.
  2. No two colliding placements overlap.
  3. An item that cannot satisfy both is skipped with a reason, never forced.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from . import config
from .models import (
    NON_COLLIDING_ROLES,
    PLACEMENT_ORDER,
    ROLE_PRECISION,
    CatalogItem,
    Confidence,
    DimensionSource,
    LayoutResult,
    MeasurementRequest,
    Opening,
    Placement,
    Precision,
    RoomAnalysis,
    Role,
    SkippedItem,
    Wall,
    WithheldItem,
)

EPS = 1e-6

Rotation = int  # one of 0, 90, 180, 270


class Arrangement(str, Enum):
    """A named spatial intent the solver can lay a room out under.

    The solver is deterministic, so asking it twice for the same wishlist
    returns the same coordinates. That is the right behaviour for a single
    answer and the wrong one for offering a choice: three renders of one
    layout are three samples of the compositor's noise, not three designs.

    Each arrangement changes only where pieces WANT to go - the anchors. Every
    invariant downstream (in bounds, no overlap, door swings clear) is enforced
    by the same code for all of them, so a variant can look different without
    being able to be wrong.
    """

    # Sofa flat against the focal wall, everything squared to it. The safe,
    # conventional reading of a room and the one most photos already show.
    WALL_ANCHORED = "wall_anchored"
    # Sofa pulled off the wall with the rug defining a conversation area.
    # Opens circulation behind the seating; wants floor area to work.
    FLOATING = "floating"
    # Seating turned into a corner, chairs closing the fourth side. Frees the
    # largest contiguous stretch of floor, which suits narrow rooms.
    CORNER = "corner"


# The arrangements the solver knows how to lay a room out under. Only the
# first is used today: the product settled on a single design showing the whole
# selected wishlist, rather than a set of alternatives. The others stay because
# they cost nothing to keep and are how a "show me another way" feature would
# be built - LayoutSolver takes the arrangement as a parameter already.
ARRANGEMENTS: list[Arrangement] = [
    Arrangement.WALL_ANCHORED,
    Arrangement.FLOATING,
    Arrangement.CORNER,
]

@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    d: float

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.d

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.d / 2


def overlaps(a: Rect, b: Rect, eps: float = EPS) -> bool:
    """Axis-aligned overlap test.

    Strict inequality matters: furniture pushed flush against a wall or against
    another piece shares an edge exactly, and touching is legal. A non-strict
    test would reject correct layouts.
    """
    return (
        a.x < b.x2 - eps
        and b.x < a.x2 - eps
        and a.y < b.y2 - eps
        and b.y < a.y2 - eps
    )


def effective_extents(w: float, d: float, rotation: Rotation) -> tuple[float, float]:
    """Rotation only swaps extents; the top-left anchor is unchanged.

    Keeping the anchor at top-left avoids trigonometry and the origin drift
    that comes with rotating about a centre.
    """
    return (d, w) if rotation in (90, 270) else (w, d)


def inflate(r: Rect, margin: float) -> Rect:
    """Clearance expressed as an inflated box, so one overlap test covers it."""
    return Rect(r.x - margin, r.y - margin, r.w + 2 * margin, r.d + 2 * margin)


@dataclass
class _Placed:
    rect: Rect
    rotation: Rotation
    item: CatalogItem
    collides: bool
    confidence: Confidence = Confidence.HIGH
    tolerance_cm: float = 0.0
    rationale: str = ""


# --- confidence ------------------------------------------------------------

# What a user could tell us to unlock each withheld role.
_MEASUREMENT_QUESTIONS: dict[str, str] = {
    "measured": "What are the room's actual wall-to-wall dimensions? "
    "A photo estimate can be off by a metre, which matters for anything "
    "that sits against a wall.",
    "openings": "Where are the doors and windows? A door's swing has to stay "
    "clear, and a console under a window needs to sit below the sill.",
    "irregular": "Is the room a plain rectangle? Alcoves, chimney breasts or "
    "an L-shape change where furniture can actually go.",
}


def _measurement_requests(room: RoomAnalysis, role: Role) -> list[MeasurementRequest]:
    """The specific facts that would let this role be placed confidently."""
    reqs: list[MeasurementRequest] = []
    if not room.measured:
        reqs.append(
            MeasurementRequest(
                field="dimensions",
                question=_MEASUREMENT_QUESTIONS["measured"],
                affects=[role.value],
            )
        )
    if not room.openings:
        reqs.append(
            MeasurementRequest(
                field="openings",
                question=_MEASUREMENT_QUESTIONS["openings"],
                affects=[role.value],
            )
        )
    if room.irregular:
        reqs.append(
            MeasurementRequest(
                field="shape",
                question=_MEASUREMENT_QUESTIONS["irregular"],
                affects=[role.value],
            )
        )
    return reqs


def blocked_zones(room: RoomAnalysis) -> list[Rect]:
    """Floor area kept clear for door swings and window access.

    Only derivable when openings are known; an empty list means "unknown",
    which is exactly why EXACT-tier pieces are withheld rather than guessed.
    """
    zones: list[Rect] = []
    for op in room.openings:
        depth = op.swing_cm if op.kind == "door" else 0.0
        if depth <= 0:
            continue
        if op.wall is Wall.NORTH:
            zones.append(Rect(op.offset_cm, 0.0, op.width_cm, depth))
        elif op.wall is Wall.SOUTH:
            zones.append(Rect(op.offset_cm, room.depth_cm - depth, op.width_cm, depth))
        elif op.wall is Wall.WEST:
            zones.append(Rect(0.0, op.offset_cm, depth, op.width_cm))
        else:  # EAST
            zones.append(Rect(room.width_cm - depth, op.offset_cm, depth, op.width_cm))
    return zones


class LayoutSolver:
    def __init__(
        self,
        room: RoomAnalysis,
        arrangement: Arrangement = Arrangement.WALL_ANCHORED,
    ) -> None:
        self.room = room
        self.arrangement = arrangement
        self.W = room.width_cm
        self.D = room.depth_cm
        self.margin = config.WALL_MARGIN_CM
        # Door swings are hard exclusions: blocking a door is not a matter of
        # taste. Empty when openings are unknown, which is why EXACT-tier
        # pieces are withheld in that case rather than placed blind.
        self.blocked = blocked_zones(room)

    # --- geometry helpers -------------------------------------------------

    def _inside(self, r: Rect) -> bool:
        m = self.margin
        return (
            r.x >= m - EPS
            and r.y >= m - EPS
            and r.x2 <= self.W - m + EPS
            and r.y2 <= self.D - m + EPS
        )

    def _clamp(self, x: float, y: float, w: float, d: float) -> tuple[float, float]:
        m = self.margin
        return (
            min(max(x, m), max(m, self.W - m - w)),
            min(max(y, m), max(m, self.D - m - d)),
        )

    def _feasible(self, cand: Rect, placed: list[_Placed], clearance: float) -> bool:
        if not self._inside(cand):
            return False
        if any(overlaps(cand, z) for z in self.blocked):
            return False
        zone = inflate(cand, clearance)
        return not any(overlaps(zone, p.rect) for p in placed if p.collides)

    # --- anchors ----------------------------------------------------------

    def _anchors(
        self, role: Role, w: float, d: float, placed: list[_Placed]
    ) -> list[tuple[float, float]]:
        """Ideal top-left positions for a role, best first.

        Encodes the interior-design intent: sofas sit against the focal wall,
        coffee tables centre on the sofa, chairs flank it, lamps take corners.
        The active arrangement varies that intent; what it cannot vary is the
        feasibility checking applied to every candidate it returns.

        Fallbacks are appended rather than substituted, so an arrangement that
        does not suit this room degrades to the conventional layout instead of
        dropping the piece.
        """
        m = self.margin
        cx = (self.W - w) / 2
        cy = (self.D - d) / 2
        sofa = self._find(placed, Role.SOFA)

        if role is Role.RUG:
            return self._rug_anchors(w, d, cx, cy, sofa)
        if role is Role.SOFA:
            return self._sofa_anchors(w, d, cx, cy)
        if role is Role.COFFEE_TABLE:
            return self._table_anchors(w, d, cx, cy, sofa)
        if role is Role.ACCENT_CHAIR:
            return self._chair_anchors(w, d, cx, cy, placed, sofa)
        if role is Role.FLOOR_LAMP:
            return self._lamp_anchors(w, d, m, sofa, placed)
        return [(cx, cy)]

    def _rug_anchors(
        self, w: float, d: float, cx: float, cy: float, sofa: _Placed | None
    ) -> list[tuple[float, float]]:
        """Rugs centre on the seating, falling back to the room's centre.

        A rug centred in the room only lands under the furniture when the
        furniture happens to be centred too - and a sofa on a wall never is.
        That produces the layout this codebase's own prompt tries to describe
        away: a rug in open floor with the seating beside it. Anchoring on the
        sofa instead is what makes "front legs on the rug" true rather than
        merely requested.

        Under FLOATING the rug goes further and defines the whole island, so
        it is centred on the seating group rather than tucked under its front
        edge.
        """
        if sofa is None:
            return [(cx, cy)]

        # Reach forward from the sofa's front edge, into the room. The rug
        # wants to cover the sofa's front legs and run under the coffee table,
        # so it starts a little way under the sofa rather than at its face.
        under = min(d * 0.3, sofa.rect.d * 0.5)
        toward_camera = sofa.rect.cy <= self.D / 2
        if self.arrangement is Arrangement.FLOATING:
            # Centred on the seating island: the rug is the thing that makes
            # a floated group read as deliberate.
            y = sofa.rect.cy - d / 2 + (d * 0.2 if toward_camera else -d * 0.2)
        else:
            y = sofa.rect.y2 - under if toward_camera else sofa.rect.y - d + under

        return [(sofa.rect.cx - w / 2, y), (cx, cy)]

    def _sofa_anchors(
        self, w: float, d: float, cx: float, cy: float
    ) -> list[tuple[float, float]]:
        """Where the sofa wants to sit - the decision the room is built around."""
        focal = self._focal_wall()
        # The focal wall is the intent, but a long sofa may only fit along a
        # different axis; try the alternatives before giving up on walls
        # entirely, since a floating sofa is a much worse outcome.
        walls = [focal, _opposite(focal)] + [
            wl for wl in (Wall.NORTH, Wall.SOUTH, Wall.EAST, Wall.WEST)
            if wl not in (focal, _opposite(focal))
        ]
        wall_anchors = [self._against(wl, w, d) for wl in walls]

        if self.arrangement is Arrangement.FLOATING:
            # Off the wall far enough to read as floating.
            #
            # A fixed 40cm walkway is a real walkway but not a visible design
            # choice: the composing model is told positions in thirds of the
            # room, so a sofa 40cm off the wall is described with the same
            # words as one against it, and the "alternative" renders as a copy
            # of the first. Floating a fifth of the room's depth actually
            # crosses that boundary, which is what makes it a different layout
            # rather than a different number.
            #
            # Still floored at a walkway's width, so a small room floats by a
            # usable amount or not at all.
            gap = max(config.CLEARANCE_LADDER_CM[0], self.D * 0.2)
            floated = self._float_off(self._focal_wall(), w, d, gap)
            if floated is not None:
                return [floated] + wall_anchors
            return wall_anchors

        if self.arrangement is Arrangement.CORNER:
            # Into a corner along the focal wall, freeing the opposite half of
            # the room as one contiguous stretch of floor.
            return self._corner_anchors(w, d) + wall_anchors

        return wall_anchors

    def _float_off(
        self, wall: Wall, w: float, d: float, gap: float
    ) -> tuple[float, float] | None:
        """The wall-anchored position pulled `gap` into the room, or None.

        None when the room cannot spare the walkway: floating a sofa in a room
        with no room to walk behind it is worse than leaving it on the wall.
        """
        x, y = self._against(wall, w, d)
        if wall in (Wall.NORTH, Wall.SOUTH):
            if self.D - 2 * self.margin < d + gap + config.CLEARANCE_LADDER_CM[0]:
                return None
            return (x, y + gap) if wall is Wall.NORTH else (x, y - gap)
        if self.W - 2 * self.margin < w + gap + config.CLEARANCE_LADDER_CM[0]:
            return None
        return (x + gap, y) if wall is Wall.WEST else (x - gap, y)

    def _corner_anchors(self, w: float, d: float) -> list[tuple[float, float]]:
        """The two corners along the focal wall, nearest corner first."""
        m = self.margin
        focal = self._focal_wall()
        if focal in (Wall.NORTH, Wall.SOUTH):
            y = m if focal is Wall.NORTH else self.D - m - d
            return [(m, y), (self.W - m - w, y)]
        x = m if focal is Wall.WEST else self.W - m - w
        return [(x, m), (x, self.D - m - d)]

    def _table_anchors(
        self, w: float, d: float, cx: float, cy: float, sofa: _Placed | None
    ) -> list[tuple[float, float]]:
        """Directly in front of the sofa, offset into the room."""
        if sofa is None:
            return [(cx, cy)]
        gap = config.CLEARANCE_LADDER_CM[0]
        # Toward the room's centre first: in front of the sofa means between it
        # and the rest of the room, which is not always +y.
        primary = (
            (sofa.rect.cx - w / 2, sofa.rect.y2 + gap)
            if sofa.rect.cy <= self.D / 2
            else (sofa.rect.cx - w / 2, sofa.rect.y - d - gap)
        )
        return [
            primary,
            (sofa.rect.cx - w / 2, sofa.rect.y2 + gap),
            (sofa.rect.cx - w / 2, sofa.rect.y - d - gap),
            (sofa.rect.x2 + gap, sofa.rect.cy - d / 2),
            (sofa.rect.x - w - gap, sofa.rect.cy - d / 2),
        ]

    def _chair_anchors(
        self,
        w: float,
        d: float,
        cx: float,
        cy: float,
        placed: list[_Placed],
        sofa: _Placed | None,
    ) -> list[tuple[float, float]]:
        """Chairs flank the sofa, alternating sides as more are added.

        With several chairs in the design, every one taking the same anchor
        list would stack them all on the sofa's right and let the ring search
        scatter the overflow. Counting the chairs already placed and flipping
        the preferred side distributes them the way a room actually seats
        people - and under CORNER they close the fourth side of the group.
        """
        if sofa is None:
            return [(cx, cy)]

        gap = config.CLEARANCE_LADDER_CM[-1]
        n = sum(1 for p in placed if p.item.role is Role.ACCENT_CHAIR)

        right = (sofa.rect.x2 + gap, sofa.rect.y)
        left = (sofa.rect.x - w - gap, sofa.rect.y)

        # Facing the sofa across the coffee table: the second pair of seats in
        # a conversation group, and the only anchor that reads as a circle.
        #
        # Measured from the coffee table when there is one, not from the sofa.
        # Deriving it from the sofa meant guessing at the gap, and the guess
        # was far too generous - a chair 194cm from a sofa is not part of the
        # conversation, it is against the opposite wall, which is exactly what
        # it rendered as. The table is the centre of the group, so a chair on
        # its far side is at a believable distance by construction.
        table = self._find(placed, Role.COFFEE_TABLE)
        anchor = table.rect if table is not None else sofa.rect
        toward_camera = sofa.rect.cy <= self.D / 2
        facing = (
            (anchor.cx - w / 2, anchor.y2 + gap)
            if toward_camera
            else (anchor.cx - w / 2, anchor.y - d - gap)
        )

        sides = [left, right] if n % 2 else [right, left]
        if self.arrangement is Arrangement.CORNER:
            # The group is already against a corner, so the open side is where
            # a chair belongs first.
            return [facing] + sides + [(cx, cy)]
        if self.arrangement is Arrangement.FLOATING:
            # Two flanking, then across - a floated sofa has space on the far
            # side that a wall-anchored one does not.
            return sides + [facing, (cx, cy)]
        return sides + [facing, (cx, cy)]

    def _lamp_anchors(
        self,
        w: float,
        d: float,
        m: float,
        sofa: _Placed | None,
        placed: list[_Placed],
    ) -> list[tuple[float, float]]:
        """Beside the seating, then the far corners - and never two in one spot.

        A lamp belongs at the end of a sofa or over a reading chair, which is a
        small number of real positions. The previous version offered the same
        list to every lamp, so a second one took the position next to the first
        and the pair rendered as a single confused object beside the sofa.

        Later lamps therefore start further down the list: the corner opposite
        the seating, which is the only other place a floor lamp reads as
        deliberate. Anything beyond that is decoration the room did not ask for.
        """
        n = sum(1 for p in placed if p.item.role is Role.FLOOR_LAMP)

        # Corners ordered away from the sofa, so a second lamp lights the part
        # of the room the first one does not.
        corners = [
            (m, m),
            (self.W - m - w, m),
            (m, self.D - m - d),
            (self.W - m - w, self.D - m - d),
        ]
        if sofa is not None:
            corners.sort(
                key=lambda c: -((c[0] - sofa.rect.cx) ** 2 + (c[1] - sofa.rect.cy) ** 2)
            )
            if n == 0:
                # The first lamp stands at the sofa's end, clear of its arm
                # rather than jammed against it.
                gap = config.CLEARANCE_LADDER_CM[0]
                beside = [
                    (sofa.rect.x2 + gap, sofa.rect.y),
                    (sofa.rect.x - w - gap, sofa.rect.y),
                ]
                return beside + corners
        return corners

    def _focal_wall(self) -> Wall:
        """The wall the seating backs onto, corrected for what the camera sees.

        The room's compass is defined by the camera: south is the wall behind
        it, north the wall it faces. So the analysis can name a wall that is
        legal geometry but a bad picture, and two answers have to be corrected
        rather than trusted:

        SOUTH is the camera's own wall. A sofa placed there sits between the
        lens and the room, so it either fills the frame or is not in it.

        EAST and WEST are seen edge-on. A sofa against a side wall runs away
        from the viewer and off the edge of the picture - which is exactly what
        it looked like - so they are only worth keeping when the room is deep
        enough that a side wall is the longer one, and the sofa genuinely has
        more room along it.

        Everything else passes through untouched. This corrects the framing,
        not the taste.
        """
        focal = self.room.focal_wall
        if focal is Wall.SOUTH:
            return Wall.NORTH
        if focal in (Wall.EAST, Wall.WEST) and self.D <= self.W:
            # A wide room seen from the south: the side walls are the short
            # ones and face away from the camera. The far wall is better.
            return Wall.NORTH
        return focal

    def _against(self, wall: Wall, w: float, d: float) -> tuple[float, float]:
        """Top-left corner for a piece flush against the given wall, centred."""
        m = self.margin
        if wall is Wall.NORTH:
            return ((self.W - w) / 2, m)
        if wall is Wall.SOUTH:
            return ((self.W - w) / 2, self.D - m - d)
        if wall is Wall.WEST:
            return (m, (self.D - d) / 2)
        return (self.W - m - w, (self.D - d) / 2)

    @staticmethod
    def _find(placed: list[_Placed], role: Role) -> _Placed | None:
        return next((p for p in placed if p.item.role is role), None)

    def _ring(self, ix: float, iy: float, w: float, d: float):
        """Yield the ideal anchor, then positions spiralling outward from it.

        Bounded by SEARCH_SPAN_CM so a piece nudges a little to fit but never
        teleports across the room: a coffee table 100cm from its sofa is a
        worse outcome than reporting it as unplaceable and letting the caller
        fall back deliberately.
        """
        seen: set[tuple[int, int]] = set()
        step = config.SEARCH_STEP_CM
        rings = int(config.SEARCH_SPAN_CM / step)

        for ring in range(rings + 1):
            offsets = (
                [(0.0, 0.0)]
                if ring == 0
                else [
                    (dx * step, dy * step)
                    for dx in range(-ring, ring + 1)
                    for dy in range(-ring, ring + 1)
                    # Only the perimeter of each ring is new.
                    if max(abs(dx), abs(dy)) == ring
                ]
            )
            # Prefer small total displacement within a ring.
            offsets.sort(key=lambda o: abs(o[0]) + abs(o[1]))
            for dx, dy in offsets:
                # Clamping keeps the candidate in bounds, but a clamped point
                # far from the anchor is not a good placement - only accept it
                # if it stays within the search span of where we wanted it.
                x, y = self._clamp(ix + dx, iy + dy, w, d)
                if (
                    abs(x - ix) > config.SEARCH_SPAN_CM
                    or abs(y - iy) > config.SEARCH_SPAN_CM
                ):
                    continue
                key = (round(x), round(y))
                if key not in seen:
                    seen.add(key)
                    yield x, y

    def _scan(self, w: float, d: float):
        """Coarse full-room sweep, used only when every anchor has failed.

        This is the difference between "we could not put the lamp where it
        belongs" and "the lamp does not fit in the room" - only the latter
        should ever produce a skip.
        """
        m = self.margin
        step = config.SEARCH_STEP_CM * 2
        y = m
        while y + d <= self.D - m + EPS:
            x = m
            while x + w <= self.W - m + EPS:
                yield x, y
                x += step
            y += step

    # --- main loop --------------------------------------------------------

    def _order_index(self, role: Role) -> float:
        """Placement priority for a role under the active arrangement.

        PLACEMENT_ORDER puts the rug first, which suits a solver that centres
        it in the room: claim the floor, then lay everything over it. But a rug
        belongs under the SEATING, not under the room's midpoint, and it cannot
        be positioned from a sofa that has not been placed yet.

        So the rug moves half a step behind the sofa - late enough to anchor on
        it, early enough to stay ahead of every piece that wants to sit on it.
        Nothing downstream is disturbed by the move because a rug does not
        collide (NON_COLLIDING_ROLES): it never takes space another piece
        wanted, whenever it is placed.
        """
        base = (
            PLACEMENT_ORDER.index(role)
            if role in PLACEMENT_ORDER
            else len(PLACEMENT_ORDER)
        )
        if role is Role.RUG:
            return PLACEMENT_ORDER.index(Role.SOFA) + 0.5
        return float(base)


    def solve(self, items: list[CatalogItem]) -> LayoutResult:
        ordered = sorted(
            items,
            key=lambda i: self._order_index(i.role),
        )

        placed: list[_Placed] = []
        skipped: list[SkippedItem] = []
        withheld: list[WithheldItem] = []

        exact_ok = self.room.supports_exact_placement

        for item in ordered:
            # Precision gate. A wall-hugging piece positioned from a photo
            # estimate is a guess dressed up as a plan: if the room is 40cm
            # narrower than we think, the sofa does not fit and the console
            # fouls the door. Withhold rather than guess, and say what would
            # unlock it.
            if ROLE_PRECISION.get(item.role) is Precision.EXACT and not exact_ok:
                withheld.append(
                    WithheldItem(
                        item_id=item.id,
                        name=item.title,
                        role=item.role,
                        reason=(
                            "This piece has to sit against a wall and clear of "
                            "doors, which needs measurements more exact than a "
                            "photo estimate."
                        ),
                        needs=_measurement_requests(self.room, item.role),
                    )
                )
                continue

            # A coffee table exists to serve seating. Without a sofa it strands
            # mid-floor and blocks the remaining pieces, so it follows the
            # seating it belongs to - withheld if the sofa is merely awaiting
            # measurements, skipped if the sofa genuinely does not fit.
            if item.role is Role.COFFEE_TABLE and self._find(placed, Role.SOFA) is None:
                sofa_withheld = any(w.role is Role.SOFA for w in withheld)
                if sofa_withheld:
                    withheld.append(
                        WithheldItem(
                            item_id=item.id,
                            name=item.title,
                            role=item.role,
                            reason=(
                                "This table is positioned relative to the sofa, "
                                "which is waiting on measurements."
                            ),
                            needs=_measurement_requests(self.room, item.role),
                        )
                    )
                else:
                    skipped.append(
                        SkippedItem(
                            item_id=item.id,
                            name=item.title,
                            role=item.role,
                            reason="no_fit",
                            detail="no seating was placed for this table to serve",
                        )
                    )
                continue

            result = self._place_one(item, placed)
            if result is None:
                w, d = item.dimensions.width_cm, item.dimensions.depth_cm
                usable_w = self.W - 2 * self.margin
                usable_d = self.D - 2 * self.margin
                too_big = min(w, d) > max(usable_w, usable_d) or (
                    min(w, d) > min(usable_w, usable_d)
                    and max(w, d) > max(usable_w, usable_d)
                )
                skipped.append(
                    SkippedItem(
                        item_id=item.id,
                        name=item.title,
                        role=item.role,
                        reason="too_large" if too_big else "no_fit",
                        detail=(
                            f"needs {w:.0f}x{d:.0f}cm; "
                            f"no free position in {self.W:.0f}x{self.D:.0f}cm room"
                        ),
                    )
                )
            else:
                self._score(result, placed)
                # A position cannot be more certain than the room it sits in.
                # Furniture placed inside a ±60cm estimate honestly has at
                # least ±60cm of slack, however tidy the geometry looks.
                result.tolerance_cm = max(
                    result.tolerance_cm, self.room.dimension_tolerance_cm
                )
                if (
                    self.room.dimension_source is DimensionSource.CONFIRMED
                    and result.confidence is Confidence.HIGH
                ):
                    # Confirmed-by-eye is good enough to place against a wall,
                    # but not good enough to call the position exact.
                    result.confidence = Confidence.MEDIUM
                placed.append(result)

        return LayoutResult(
            room_width_cm=self.W,
            room_depth_cm=self.D,
            placements=[
                Placement(
                    item_id=p.item.id,
                    name=p.item.title,
                    role=p.item.role,
                    x_cm=round(p.rect.x, 1),
                    y_cm=round(p.rect.y, 1),
                    w_cm=round(p.rect.w, 1),
                    d_cm=round(p.rect.d, 1),
                    rotation=p.rotation,
                    z=0 if p.item.role in NON_COLLIDING_ROLES else 1,
                    swatch=p.item.swatch,
                    price_cents=p.item.price_cents,
                    merchant=p.item.merchant,
                    confidence=p.confidence,
                    tolerance_cm=round(p.tolerance_cm, 1),
                    rationale=p.rationale,
                )
                for p in placed
            ],
            skipped=skipped,
            withheld=withheld,
        )

    def _score(self, p: _Placed, placed: list[_Placed]) -> None:
        """Attach confidence and a positional tolerance to a placement.

        Tolerance is the honest part: how far this piece could move before the
        layout reads as wrong. A rug centred in open floor can shift 40cm and
        nobody notices; a chair wedged between a sofa and a wall cannot.
        """
        role = p.item.role
        free = self._free_margin(p.rect, placed)

        if role is Role.RUG:
            p.confidence = Confidence.HIGH
            p.tolerance_cm = 40.0
            p.rationale = "centred in the room; everything else sits on top"
            return

        if role is Role.COFFEE_TABLE:
            sofa = self._find(placed, Role.SOFA)
            if sofa:
                p.confidence = Confidence.HIGH
                p.tolerance_cm = 20.0
                p.rationale = "centred on the sofa at walking clearance"
            else:
                p.confidence = Confidence.MEDIUM
                p.tolerance_cm = 30.0
                p.rationale = "centred in the room; no seating to anchor to"
            return

        if role in (Role.ACCENT_CHAIR, Role.FLOOR_LAMP):
            # These float, so they are forgiving - unless the room has closed
            # in around them, in which case the position is load-bearing.
            if free >= 30.0:
                p.confidence = Confidence.HIGH
                p.tolerance_cm = min(free, 35.0)
                p.rationale = "free-standing with room to move"
            else:
                p.confidence = Confidence.MEDIUM
                p.tolerance_cm = max(free, 5.0)
                p.rationale = f"only {free:.0f}cm of slack around it"
            return

        # EXACT-tier roles only reach here when the room was measured, so the
        # position is trustworthy; the remaining question is how tight it is.
        p.confidence = Confidence.HIGH if free >= 15.0 else Confidence.MEDIUM
        p.tolerance_cm = min(free, 15.0)
        p.rationale = "positioned against the wall from measured dimensions"

    def _free_margin(self, r: Rect, placed: list[_Placed]) -> float:
        """Slack around this rect: how far it could move before it reads wrong.

        Contact with a wall is excluded. A sofa flush against the wall it was
        anchored to has a zero gap there by design, and counting that as
        tightness would mark every correctly-placed piece as low confidence.
        """
        wall_gaps = [
            r.x - self.margin,
            r.y - self.margin,
            (self.W - self.margin) - r.x2,
            (self.D - self.margin) - r.y2,
        ]
        gaps = [g for g in wall_gaps if g > EPS] or [max(wall_gaps)]
        for other in placed:
            if not other.collides or other.rect is r:
                continue
            o = other.rect
            # Only count separation along the axis on which they actually face
            # each other; two rects side by side are not constrained vertically.
            if r.y < o.y2 and o.y < r.y2:
                gaps.append(o.x - r.x2 if o.x >= r.x2 else r.x - o.x2)
            if r.x < o.x2 and o.x < r.x2:
                gaps.append(o.y - r.y2 if o.y >= r.y2 else r.y - o.y2)
        return max(0.0, min(g for g in gaps))

    def _place_one(self, item: CatalogItem, placed: list[_Placed]) -> _Placed | None:
        collides = item.role not in NON_COLLIDING_ROLES
        # Every role may turn, rugs included: a 220x155 rug laid the other way
        # round is the same rug, and refusing the rotation drops it from a room
        # it plainly fits. Rotation only swaps extents, so the rug's own anchor
        # (centred) is unaffected.
        rotations: list[Rotation] = [0, 90]
        base_w = item.dimensions.width_cm
        base_d = item.dimensions.depth_cm

        # Clearance degrades before an item is abandoned. A hard 40cm rule is
        # unsatisfiable in small rooms (90cm sofa + 40 + 60cm table = 190cm of a
        # 200cm depth), so a tight-but-usable layout beats dropping the piece.
        ladder = (
            config.CLEARANCE_LADDER_CM
            if item.role in (Role.COFFEE_TABLE, Role.ACCENT_CHAIR)
            else [0.0]
        )

        def accepts(cand: Rect, clearance: float) -> bool:
            # A non-colliding item (a rug) is free to sit under everything
            # already placed, but must still stay in bounds and clear of door
            # swings - a rug under a door still stops it opening.
            if not collides:
                return self._inside(cand) and not any(
                    overlaps(cand, z) for z in self.blocked
                )
            return self._feasible(cand, placed, clearance)

        # Pass 1: near the ideal anchor, best clearance first. This is where a
        # good layout comes from - sofa on the focal wall, table in front of it.
        for clearance in ladder:
            for rotation in rotations:
                w, d = effective_extents(base_w, base_d, rotation)
                if w > self.W - 2 * self.margin or d > self.D - 2 * self.margin:
                    continue  # cannot fit at any position in this orientation
                for ix, iy in self._anchors(item.role, w, d, placed):
                    for x, y in self._ring(ix, iy, w, d):
                        if accepts(Rect(x, y, w, d), clearance):
                            return _Placed(Rect(x, y, w, d), rotation, item, collides)

        # Pass 2: anywhere it legally fits. The result is less considered, but
        # a piece placed awkwardly still beats one dropped from the design.
        for rotation in rotations:
            w, d = effective_extents(base_w, base_d, rotation)
            if w > self.W - 2 * self.margin or d > self.D - 2 * self.margin:
                continue
            for x, y in self._scan(w, d):
                if accepts(Rect(x, y, w, d), 0.0):
                    return _Placed(Rect(x, y, w, d), rotation, item, collides)

        return None


def _opposite(wall: Wall) -> Wall:
    return {
        Wall.NORTH: Wall.SOUTH,
        Wall.SOUTH: Wall.NORTH,
        Wall.EAST: Wall.WEST,
        Wall.WEST: Wall.EAST,
    }[wall]


# --- selftest --------------------------------------------------------------


def selftest() -> None:
    """Assert the solver's invariants across roomy, tight and cramped rooms."""
    from .seed_data import SEED_ITEMS

    def pick(role: Role) -> CatalogItem:
        return next(i for i in SEED_ITEMS if i.role is role and i.in_stock)

    wishlist = [pick(r) for r in PLACEMENT_ORDER]

    cases = [
        ("roomy   400x320", 400.0, 320.0),
        ("tight   250x200", 250.0, 200.0),
        ("cramped 160x140", 160.0, 140.0),
    ]

    for label, w, d in cases:
        # Measured room with known openings: the full set is placeable.
        room = RoomAnalysis(
            width_cm=w,
            depth_cm=d,
            dimension_source=DimensionSource.MEASURED,
            openings=[Opening(kind="door", wall=Wall.NORTH, offset_cm=10, width_cm=80, swing_cm=80)],
        )
        result = LayoutSolver(room).solve(wishlist)

        # Invariant 1: everything inside the room.
        for p in result.placements:
            assert p.x_cm >= 0 and p.y_cm >= 0, f"{label}: {p.name} outside (neg)"
            assert p.x_cm + p.w_cm <= w + EPS, f"{label}: {p.name} exceeds width"
            assert p.y_cm + p.d_cm <= d + EPS, f"{label}: {p.name} exceeds depth"

        # Invariant 2: no overlap among colliding pieces.
        solid = [p for p in result.placements if p.role not in NON_COLLIDING_ROLES]
        for i, a in enumerate(solid):
            for b in solid[i + 1 :]:
                ra = Rect(a.x_cm, a.y_cm, a.w_cm, a.d_cm)
                rb = Rect(b.x_cm, b.y_cm, b.w_cm, b.d_cm)
                assert not overlaps(ra, rb), f"{label}: {a.name} overlaps {b.name}"

        # Invariant 3: nothing silently vanishes.
        assert (
            len(result.placements) + len(result.skipped) + len(result.withheld)
            == len(wishlist)
        ), f"{label}: item accounting mismatch"

        # Invariant 5: nothing blocks a door swing, rugs included.
        for zone in blocked_zones(room):
            for p in result.placements:
                assert not overlaps(
                    Rect(p.x_cm, p.y_cm, p.w_cm, p.d_cm), zone
                ), f"{label}: {p.name} blocks the door"

        # Invariant 4: design intent, not just legality. A coffee table must
        # end up near the seating it serves - a legal but stranded table on the
        # far side of the room passes the overlap test and is still wrong.
        by_role = {p.role: p for p in result.placements}
        table, sofa = by_role.get(Role.COFFEE_TABLE), by_role.get(Role.SOFA)
        assert not (table and not sofa), f"{label}: table placed without seating"
        if table and sofa:
            gap = max(
                0.0,
                max(sofa.y_cm - (table.y_cm + table.d_cm), table.y_cm - (sofa.y_cm + sofa.d_cm)),
                max(sofa.x_cm - (table.x_cm + table.w_cm), table.x_cm - (sofa.x_cm + sofa.w_cm)),
            )
            assert gap <= 120.0, f"{label}: table stranded {gap:.0f}cm from sofa"

        print(f"{label}: {len(result.placements)} placed, {len(result.skipped)} skipped")
        for p in sorted(result.placements, key=lambda p: p.z):
            print(
                f"    {p.role.value:<13} {p.name:<26} "
                f"({p.x_cm:>6.1f},{p.y_cm:>6.1f}) {p.w_cm:>5.0f}x{p.d_cm:<5.0f} rot{p.rotation}"
            )
        for s in result.skipped:
            print(f"    SKIP {s.role.value:<13} {s.name:<26} {s.reason}: {s.detail}")
        print()

    # --- precision gate ---------------------------------------------------
    # Same room, same wishlist, differing only in how well it is measured.
    estimated = RoomAnalysis(width_cm=400, depth_cm=320)  # ESTIMATED by default
    est = LayoutSolver(estimated).solve(wishlist)
    exact_roles = {r for r, p in ROLE_PRECISION.items() if p is Precision.EXACT}

    withheld_roles = {w.role for w in est.withheld}
    placed_roles = {p.role for p in est.placements}
    # Every EXACT-tier role must be withheld. The coffee table joins them by
    # cascade - it is positioned relative to the sofa, so it waits on the same
    # measurements rather than being stranded mid-floor.
    assert exact_roles <= withheld_roles, (
        f"unmeasured room must withhold {exact_roles}, withheld {withheld_roles}"
    )
    assert withheld_roles <= exact_roles | {Role.COFFEE_TABLE}, (
        f"unexpected role withheld: {withheld_roles - exact_roles}"
    )
    assert not (placed_roles & exact_roles), "placed a wall-hugging piece unmeasured"
    assert all(w.needs for w in est.withheld), "withheld item asks for nothing"
    # Nothing withheld may appear in the layout.
    assert not (withheld_roles & placed_roles), "item both withheld and placed"

    measured = RoomAnalysis(width_cm=400, depth_cm=320,
                            dimension_source=DimensionSource.MEASURED)
    meas = LayoutSolver(measured).solve(wishlist)
    assert not meas.withheld, "measured room should withhold nothing"
    assert exact_roles <= {p.role for p in meas.placements}, (
        "measured room should place wall-hugging pieces"
    )

    # An irregular room cannot support wall anchors even when measured.
    lshaped = RoomAnalysis(width_cm=400, depth_cm=320, irregular=True,
                           dimension_source=DimensionSource.MEASURED)
    assert LayoutSolver(lshaped).solve(wishlist).withheld, (
        "irregular room should withhold wall-hugging pieces"
    )

    # CONFIRMED is the middle tier: the user glanced at the estimate and said
    # it looked right. That is enough to place against a wall, but the layout
    # must not claim the precision of a real measurement.
    confirmed = RoomAnalysis(
        width_cm=400, depth_cm=320, dimension_source=DimensionSource.CONFIRMED
    )
    conf = LayoutSolver(confirmed).solve(wishlist)
    assert not conf.withheld, "confirmed dimensions should release the gate"
    assert exact_roles <= {p.role for p in conf.placements}, (
        "confirmed room should place wall-hugging pieces"
    )
    # Every placement inherits at least the room's own uncertainty.
    floor = confirmed.dimension_tolerance_cm
    assert all(p.tolerance_cm >= floor for p in conf.placements), (
        f"confirmed placements must carry at least ±{floor:.0f}cm"
    )
    assert all(p.confidence is not Confidence.HIGH for p in conf.placements), (
        "confirmed-by-eye must not report HIGH placement confidence"
    )

    print("precision gate:")
    print(f"    estimated room -> {len(est.placements)} placed, "
          f"{len(est.withheld)} withheld pending measurements")
    for wi in est.withheld:
        print(f"      {wi.role.value:<9} needs: {', '.join(n.field for n in wi.needs)}")
    print(f"    measured room  -> {len(meas.placements)} placed, 0 withheld")
    print()

    # Confidence must be attached and must vary by how constrained a piece is.
    confidences = {p.role.value: (p.confidence.value, p.tolerance_cm)
                   for p in meas.placements}
    assert all(c for c, _ in confidences.values()), "placement missing confidence"
    print("confidence / tolerance:")
    for role, (conf, tol) in confidences.items():
        print(f"    {role:<14} {conf:<7} ±{tol:.0f}cm")
    print()

    print("all invariants hold")


if __name__ == "__main__":
    selftest()
