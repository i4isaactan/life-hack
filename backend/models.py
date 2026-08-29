"""Pydantic schemas shared by the real and offline provider paths.

Both providers return these exact types. That is what makes the "runs without an
API key" guarantee enforceable rather than aspirational: a mock that drifted from
the real shape would fail validation here rather than at the client.

Units: all spatial values are centimetres, all money is integer cents.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

# --- Vocabulary ------------------------------------------------------------


class Role(str, Enum):
    """Functional slot a piece fills. Drives retrieval and placement order."""

    RUG = "rug"
    SOFA = "sofa"
    COFFEE_TABLE = "coffee_table"
    ACCENT_CHAIR = "accent_chair"
    FLOOR_LAMP = "floor_lamp"


class Wall(str, Enum):
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"


# Placement priority: pieces that define the room claim space before decor, so
# the most constrained items are never boxed out by a lamp.
PLACEMENT_ORDER: list[Role] = [
    Role.RUG,
    Role.SOFA,
    Role.COFFEE_TABLE,
    Role.ACCENT_CHAIR,
    Role.FLOOR_LAMP,
]

# Rugs are deliberately non-colliding: the plan is 2D but the room is 3D, and
# sofas and tables physically rest on top of a rug. Treating one as solid would
# push every other piece off it, which is the opposite of correct layout.
NON_COLLIDING_ROLES: frozenset[Role] = frozenset({Role.RUG})


class Precision(str, Enum):
    """How much positional accuracy a role actually demands.

    A sofa floating in a large room tolerates being 20cm off; nobody notices.
    A console that must sit flush against a wall, clear of a door swing, does
    not - being 20cm off is the difference between fitting and not. Roles in
    the EXACT tier are only placed when the room data supports that precision.
    """

    # Forgiving: centred or free-floating, wide tolerance, no hard adjacency.
    APPROXIMATE = "approximate"
    # Unforgiving: must hug a wall, clear an opening, or abut another piece.
    EXACT = "exact"


ROLE_PRECISION: dict[Role, Precision] = {
    Role.RUG: Precision.APPROXIMATE,
    Role.COFFEE_TABLE: Precision.APPROXIMATE,
    Role.ACCENT_CHAIR: Precision.APPROXIMATE,
    Role.FLOOR_LAMP: Precision.APPROXIMATE,
    # Wall-hugging: a gap behind a sofa reads as a mistake, and it is the piece
    # most likely to foul a door swing or a window.
    Role.SOFA: Precision.EXACT,
}


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Opening(BaseModel):
    """A door or window along one wall.

    `offset_cm` is measured from the wall's left end looking into the room.
    Doors carry a swing clearance that furniture must stay out of.
    """

    kind: Literal["door", "window"]
    wall: Wall
    offset_cm: float = Field(ge=0)
    width_cm: float = Field(gt=0)
    swing_cm: float = 0.0


class MeasurementRequest(BaseModel):
    """What the user could supply to unlock a withheld piece."""

    field: str
    question: str
    affects: list[str] = Field(default_factory=list)


class DimensionSource(str, Enum):
    """Where the room's dimensions came from, and how far to trust them.

    The middle state is the point. Asking a user to measure their room before
    anything happens is friction most will meet by inventing a number, which is
    worse than an estimate because it is trusted just as absolutely. Showing
    them the AI's estimate and letting them accept it costs one tap, and people
    are far better at spotting a wrong number than producing a right one.
    """

    # Read from a photo, unseen by anyone. Cannot support wall-hugging pieces.
    ESTIMATED = "estimated"
    # An estimate the user looked at and accepted. Good enough to build on.
    CONFIRMED = "confirmed"
    # Numbers the user supplied. Tightest tolerances.
    MEASURED = "measured"


class DimensionProposal(BaseModel):
    """An estimate offered back to the user for confirmation.

    Sent as its own event so a client can render an editable prefilled field
    rather than a blank one - "is this about right?" instead of "measure your
    room". Accepting promotes the room to CONFIRMED and releases the pieces
    the solver is holding back.
    """

    width_cm: float = Field(gt=0)
    depth_cm: float = Field(gt=0)
    source: DimensionSource = DimensionSource.ESTIMATED
    # Roles currently withheld that accepting these numbers would release.
    unlocks: list[str] = Field(default_factory=list)
    question: str = ""


# --- Camera ----------------------------------------------------------------


class FloorQuad(BaseModel):
    """Four image-space points marking a rectangle on the floor, clockwise
    from the near-left corner, in normalized [0,1] image coordinates.

    This is the bridge between the solver's centimetres and the photo's
    pixels. Four corresponding points are exactly what a homography needs,
    and a floor rectangle is the one thing a vision model can pick out of a
    room photo with any reliability.
    """

    near_left: tuple[float, float]
    near_right: tuple[float, float]
    far_right: tuple[float, float]
    far_left: tuple[float, float]

    def as_list(self) -> list[tuple[float, float]]:
        return [self.near_left, self.near_right, self.far_right, self.far_left]


class CameraCalibration(BaseModel):
    """How the room's floor plane maps into the photo.

    Without this a render cannot know where in the image a piece at
    (128cm, 237cm) belongs, and the whole point of Tier 2 is that the solver -
    not the image model - decides position.

    Crucially the quad bounds the floor *the camera can see*, which is almost
    never the whole room: a photographer standing in the doorway cannot see the
    floor they are standing on. `near_depth_cm` and `far_depth_cm` say which
    slice of the room the quad actually spans. Assuming it spans all of it
    projects every piece too far forward, sliding near-wall furniture out of
    frame entirely.
    """

    quad: FloorQuad
    horizon_y: float = Field(ge=0, le=1)
    source: Literal["openai", "mock"] = "mock"

    # Room-depth coordinates of the quad's two edges, in solver centimetres.
    # near_depth_cm is the edge closest to the camera (largest y), far_depth_cm
    # the one at the back wall. Defaults describe a camera that sees the back
    # of the room but stands ~80cm inside it, which is typical of a phone photo
    # taken from a doorway.
    near_depth_cm: float | None = Field(default=None, gt=0)
    far_depth_cm: float = Field(default=0.0, ge=0)

    # A calibration derived from a photo estimate is itself an estimate. This
    # caps how much a render may claim, independent of the layout's own
    # confidence.
    confidence: Confidence = Confidence.MEDIUM

    def depth_span(self, room_depth_cm: float) -> tuple[float, float]:
        """The (near, far) room depths this quad spans, clamped and ordered.

        `near_depth_cm` unset means the quad reaches the near wall, which is
        the right default only when the photo genuinely shows the whole floor.
        """
        near = room_depth_cm if self.near_depth_cm is None else self.near_depth_cm
        near = min(max(near, 0.0), room_depth_cm)
        far = min(max(self.far_depth_cm, 0.0), room_depth_cm)
        return (near, far)


# --- Detection -------------------------------------------------------------


class Detection(BaseModel):
    """An existing piece of furniture found in the user's photo.

    Boxes are normalized [0,1] image coordinates, x1/y1 top-left. Detections
    drive three things: what to erase before compositing, a prior on what the
    user already owns, and the crop that reverse-search matches against the
    catalog.
    """

    # None for furniture we recognise but do not sell - a bookshelf, a bed.
    # Worth keeping: it is still a real object in the room, it still tells us
    # about the user's taste, and dropping it would make the detection list
    # look wrong to anyone comparing it against their own photo. Only a role
    # that is not None can be erased or replaced.
    role: Role | None = None
    label: str
    score: float = Field(ge=0, le=1)
    x1: float
    y1: float
    x2: float
    y2: float
    # Populated once segmentation runs; a base64 PNG of the binary mask.
    mask_b64: str | None = None

    # A visual description of THIS instance - "low-slung two-seat sofa in
    # oatmeal boucle, tapered oak legs" - written to be embedded against
    # CatalogItem.embed_text. Distinct from `label`, which is just the class
    # name the detector emitted and carries no appearance at all.
    caption: str = ""

    @property
    def area(self) -> float:
        """Fraction of the photo this box covers.

        Used to rank detections: the largest piece is the one a user means by
        "the sofa", and a tiny box is usually a reflection or a cushion.
        """
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def crop_box(self, width: int, height: int, pad: float = 0.04) -> tuple[int, int, int, int]:
        """Pixel crop box for this detection, padded and clamped to the image.

        A little padding matters for matching: a box cut exactly to the object
        loses the legs and the silhouette edge, which are most of what
        distinguishes one armchair from another.
        """
        px, py = pad * (self.x2 - self.x1), pad * (self.y2 - self.y1)
        left = int(max(0.0, self.x1 - px) * width)
        top = int(max(0.0, self.y1 - py) * height)
        right = int(min(1.0, self.x2 + px) * width)
        bottom = int(min(1.0, self.y2 + py) * height)
        # A degenerate box would make PIL raise; give it at least one pixel.
        return (left, top, max(left + 1, right), max(top + 1, bottom))


class DetectedMatch(BaseModel):
    """One catalog item proposed as the identity of a detected object."""

    item_id: str
    title: str
    merchant: str
    price_cents: int
    currency: str = "SGD"
    image_url: str
    # Where to actually buy it. Reverse search's whole point is answering
    # "where do I get that?", so a match with no way to act on it is half an
    # answer.
    checkout_url: str = ""
    # Fused rank score, 0-1. This is a ranking signal, not a probability that
    # the item IS the object.
    score: float = Field(ge=0, le=1)
    # The component scores behind `score`, so a client (and a developer
    # debugging a bad match) can see which signal actually found this. None
    # means that signal did not rank the item at all - image_score is None on
    # every item when CLIP is not installed.
    image_score: float | None = None
    text_score: float | None = None
    # Which signals contributed: "image", "text", or "both". Fusion makes the
    # combined number hard to interpret alone; this says where it came from.
    matched_by: Literal["image", "text", "both"] = "text"


class ReverseSearchResult(BaseModel):
    """What one detected object matched to in the catalog.

    `confident` is the honest flag a client should gate its wording on: a
    match list is always returned when the role has any stock at all, because
    the nearest neighbour of anything is still something. Only `confident`
    distinguishes "this looks like the LANDSKRONA" from "we found nothing
    like it and these are just the closest sofas we sell".
    """

    detection: Detection
    matches: list[DetectedMatch] = Field(default_factory=list)
    confident: bool = False


# --- Render ----------------------------------------------------------------


class RenderMethod(str, Enum):
    """How a render image was produced. Surfaced so a client can label it."""

    # SDXL inpaint conditioned on solver depth plus the product image, one call
    # per item, masked to that item's projected footprint.
    GENERATIVE = "generative"
    # One Gemini call composing the whole room from the photo plus the catalog
    # product shots. Renders every piece at once, so lighting is coherent and
    # each piece is derived from its actual product photograph.
    COMPOSED = "composed"
    # Pillow schematic: solver rectangles projected onto the photo. Offline.
    SCHEMATIC = "schematic"


class RenderResult(BaseModel):
    """One visualization of a catalog item placed in the user's room.

    NOT a photograph. A generative render approximates the product; it is
    conditioned on the real product image but reconstructs the pixels, so it
    can differ in detail from what actually ships. The disclaimer travels with
    the payload for the same reason CheckoutResult carries one.
    """

    item_id: str
    name: str
    role: Role
    method: RenderMethod
    # Data URI (offline schematic) or provider URL (generative).
    image_url: str
    # Inherited from the placement and capped by the calibration: a render of
    # a MEDIUM-confidence position cannot itself be HIGH.
    confidence: Confidence = Confidence.MEDIUM
    replaced: str | None = None  # label of the detected piece this stands in for
    elapsed_ms: int = 0
    simulated: Literal[True] = True
    disclaimer: str = (
        "AI visualization - an approximation of this product in your room, "
        "not a photograph of it."
    )


class RenderFailure(BaseModel):
    """A render that could not be produced. Never a silent gap."""

    item_id: str
    name: str
    role: Role
    reason: Literal[
        "no_photo",
        "no_calibration",
        # The placement is real but sits outside the floor the photo captured -
        # a distinct answer from "we could not calibrate", and one the user can
        # act on by stepping back and retaking the shot.
        "out_of_frame",
        "not_placed",
        "provider_error",
        "no_product_image",
    ]
    detail: str = ""


class RoomRender(BaseModel):
    """One image of the whole design composed into the user's room.

    Distinct from RenderResult, which is one item at a time. A composing model
    edits the entire photo in a single pass, so the unit of output is the room:
    there is no per-item image to hand back, and pretending otherwise would
    imply a precision the call does not have.

    NOT a photograph. The pieces are conditioned on real product shots but the
    model reconstructs every pixel, so details can differ from what ships.
    """

    image_url: str
    method: RenderMethod = RenderMethod.COMPOSED
    # The items the model was actually shown a reference for, in the order they
    # appeared in the prompt.
    item_ids: list[str] = Field(default_factory=list)
    # Placed items left out because the reference budget ran out, or because no
    # product image could be fetched. Named so the client can say what is
    # missing rather than leaving the user to spot it.
    omitted: list[str] = Field(default_factory=list)
    # Existing furniture the model was told to remove.
    replaced: list[str] = Field(default_factory=list)
    confidence: Confidence = Confidence.MEDIUM
    elapsed_ms: int = 0
    simulated: Literal[True] = True
    disclaimer: str = (
        "AI visualization - an approximation of these products in your room, "
        "not a photograph of them."
    )


class RenderRequest(BaseModel):
    session_id: str
    # Restrict to specific items; empty means every placed item.
    item_ids: list[str] = Field(default_factory=list)
    # Force the per-item path even when a composing backend is configured.
    # Useful for comparing the two, and for re-rendering a single swap.
    per_item: bool = False


# --- Catalog ---------------------------------------------------------------


class Dimensions(BaseModel):
    width_cm: float = Field(gt=0)
    depth_cm: float = Field(gt=0)
    height_cm: float = Field(gt=0)


class CatalogItem(BaseModel):
    """One purchasable item. See seed_data.py for where the catalog comes from."""

    id: str
    merchant: str
    title: str
    role: Role
    price_cents: int = Field(ge=0)
    currency: str = "SGD"
    dimensions: Dimensions
    materials: list[str] = Field(default_factory=list)
    primary_color: str
    swatch: str  # hex, for rendering the piece on a floor plan
    style_tags: list[str] = Field(default_factory=list)
    image_url: str
    checkout_url: str
    in_stock: bool = True

    # The vendor's own prose. This is the strongest semantic signal available -
    # "light, airy design with high legs and slim lines" is what someone asking
    # for an airy room is actually searching for, and no amount of structured
    # tagging reproduces it. Empty when the source has none.
    description: str = ""
    # The visible upholstery or finish, e.g. "Gunnared light green". Distinct
    # from `materials`, which is a full bill of materials listing the foam and
    # fibreboard inside the piece as well as the fabric on it.
    finish: str = ""
    seating_capacity: int | None = None
    # IKEA's product line, e.g. "LANDSKRONA" or "IKEA PS 2026". Two pieces
    # sharing one are the same designed range - matching frame, fabric and
    # proportions - which is the strongest "these go together" signal the
    # catalog actually contains.
    series: str = ""

    def embed_text(self) -> str:
        """The text embedded into the vector index for this item.

        Ordered most- to least-discriminating. Embeddings weight the whole
        string, so filler dilutes the signal: a construction bill of materials
        ("polypropylene, polyurethane, fibreboard") describes the inside of a
        sofa and matches nothing anyone would ask for, which is why `materials`
        is capped and comes last.
        """
        d = self.dimensions
        parts = [
            self.title,
            self.role.value.replace("_", " "),
        ]
        if self.description:
            parts.append(self.description)
        if self.style_tags:
            parts.append(f"Style: {', '.join(self.style_tags)}")
        parts.append(f"Color: {self.primary_color}")
        if self.finish:
            parts.append(f"Upholstery: {self.finish}")
        if self.seating_capacity:
            parts.append(f"Seats {self.seating_capacity}")
        if self.materials:
            parts.append(f"Materials: {', '.join(self.materials[:4])}")
        parts.append(
            f"Dimensions {d.width_cm:.0f}x{d.depth_cm:.0f}x{d.height_cm:.0f} cm"
        )
        parts.append(f"Sold by {self.merchant}")
        # Parts are joined with ". ", so a part that already ends in a full
        # stop (vendor prose usually does) would produce "..".
        return ". ".join(x.rstrip(" .") for x in parts if x.strip()) + "."


# --- Alternatives ----------------------------------------------------------


class Alternative(BaseModel):
    """A catalog item the user could swap in for the one that was chosen.

    Retrieval already returns several candidates per role and selection keeps
    one; the rest are just as valid, and discarding them silently presents a
    single pick as if it were the only option. Everything here is what a client
    needs to render a clickable card without a second request.
    """

    item_id: str
    name: str
    merchant: str
    role: Role
    price_cents: int
    # Difference against the currently selected item, negative when cheaper.
    price_delta_cents: int
    swatch: str
    image_url: str
    materials: list[str] = Field(default_factory=list)
    primary_color: str = ""
    style_tags: list[str] = Field(default_factory=list)
    width_cm: float
    depth_cm: float
    height_cm: float

    # Whether swapping this in keeps the design inside budget. False items are
    # still offered - the user may want to spend more - but a client should
    # mark them rather than let the total silently go over.
    affordable: bool = True


class RoleOptions(BaseModel):
    """The chosen item for one role, plus everything else that fit."""

    role: Role
    selected_id: str
    alternatives: list[Alternative] = Field(default_factory=list)


# --- Conversational intent -------------------------------------------------


class IntentKind(str, Enum):
    """What the user is actually asking for this turn.

    Without this the message text only ever reached the embedding query, where
    "make it cheaper" is just three tokens with no more force than "oak" - the
    design came back at the same budget. Parsing intent is what turns a search
    box into a conversation.
    """

    # Design or redesign the room from scratch.
    DESIGN = "design"
    # Adjust the existing design: cheaper, warmer, bigger sofa.
    REFINE = "refine"
    # Replace one role's item, optionally with a stated preference.
    REPLACE = "replace"
    # Explain a choice already made. Answered from state, no re-solve.
    EXPLAIN = "explain"
    # Supply room facts (dimensions, doors) without changing the brief.
    MEASURE = "measure"
    # Anything else - greetings, off-topic. Answered without touching state.
    CHITCHAT = "chitchat"


class Intent(BaseModel):
    """Structured reading of one user message, in the graph's own vocabulary.

    Every field is optional and only set when the message actually says so, so
    an unmentioned constraint carries forward from the session rather than
    being reset to a default on each turn.
    """

    kind: IntentKind = IntentKind.DESIGN

    # --- constraint updates ---
    budget_cents: int | None = None
    aesthetic: str | None = None
    # Free-text flavour that should steer retrieval ("warmer", "less busy").
    style_note: str | None = None

    # --- targeted changes ---
    # Roles the user wants re-picked, e.g. "show me a different rug".
    reroll_roles: list[Role] = Field(default_factory=list)
    # Items explicitly rejected; never re-offered this session.
    reject_item_ids: list[str] = Field(default_factory=list)
    # Roles to drop from the design entirely ("I don't need a rug").
    remove_roles: list[Role] = Field(default_factory=list)
    # Per-role size ceiling in cm, from "the sofa is too big".
    max_width_cm: dict[str, float] = Field(default_factory=dict)

    # --- explanation ---
    explain_role: Role | None = None

    # What to say back when no re-solve is needed (explain/chitchat).
    reply: str = ""
    # Why the parser read it this way, for debugging and for the client to show.
    reasoning: str = ""


# --- Bundles ---------------------------------------------------------------


class BundleBasis(str, Enum):
    """Why two pieces are suggested together.

    NOTE ON WHAT IS ABSENT. There is no "customers also bought" here, because
    the catalog carries no purchase, basket or view history - only product
    attributes. Inventing co-purchase counts would present fabricated social
    proof as real shopper behaviour, so every basis below is a property of the
    products themselves, checkable against the catalog.
    """

    # Same IKEA product line: matching frame, fabric and proportions. The
    # strongest signal the data actually contains.
    SAME_SERIES = "same_series"
    # Different lines that share style tags and sit in the same colour family.
    STYLE_MATCH = "style_match"
    # A role the design is missing, filled with something that fits the rest.
    COMPLETES_ROOM = "completes_room"


_BASIS_LABEL: dict[str, str] = {
    "same_series": "Matching set",
    "style_match": "Completes the look",
    "completes_room": "Finishes the room",
}


class BundleItem(BaseModel):
    """One piece inside a suggested bundle."""

    item_id: str
    name: str
    role: Role
    price_cents: int
    image_url: str
    swatch: str
    series: str = ""
    # False when this piece is already in the cart: a bundle shown against a
    # design usually extends it rather than replacing it wholesale, and the
    # client needs to know which rows are new spend.
    is_new: bool = True


class Bundle(BaseModel):
    """A set of pieces suggested together, with the reason stated.

    `reason` is user-facing and must describe the actual basis. A bundle whose
    reason cannot be justified from the catalog should not be built.
    """

    id: str
    basis: BundleBasis
    label: str
    reason: str
    items: list[BundleItem]
    # Total for the whole bundle, and for only the pieces not already owned.
    total_cents: int
    added_cents: int
    currency: str = "SGD"
    # Whether adding the new pieces keeps the design within budget.
    affordable: bool = True
    # Every piece physically fits the room alongside what is already placed.
    # False bundles are still shown, flagged, because a user may be planning a
    # different room - but they must never be presented as drop-in additions.
    fits_room: bool = True


class SwapRequest(BaseModel):
    """Replace one selected item with an alternative and re-solve.

    A swap is never just a re-render: a different sofa has different
    dimensions, so the layout has to be re-solved and the cart re-billed. The
    item may not even fit, which is a legitimate outcome the client must be
    able to show.
    """

    session_id: str
    role: Role
    item_id: str
    # Skip the image render and return only the new layout and cart. Useful for
    # previewing the cost of a swap before paying for a render.
    layout_only: bool = False


# --- Room analysis ---------------------------------------------------------


class RoomAnalysis(BaseModel):
    """Output of the vision node. Identical shape from gpt-4o and the mock."""

    width_cm: float = Field(gt=0)
    depth_cm: float = Field(gt=0)
    focal_wall: Wall = Wall.SOUTH
    wall_color: str = "warm white"
    flooring: str = "light oak"
    lighting: str = "natural, north-facing"
    notes: str = ""
    # Which path produced this analysis, surfaced so a client can show it
    # honestly. "default" is the one worth distinguishing: it means no photo
    # was supplied, so there was nothing to analyse - NOT that the vision
    # provider is unavailable, which is what "mock" means. Conflating the two
    # makes an ordinary no-photo turn look like a misconfiguration.
    source: Literal["openai", "mock", "default"] = "default"

    # How the dimensions were obtained. This gates the EXACT precision tier:
    # a photo estimate nobody has looked at can be metres off, and metre-scale
    # error is the difference between a sofa fitting and not.
    dimension_source: DimensionSource = DimensionSource.ESTIMATED
    # Known doors and windows. Empty means unknown, not "none" — an unknown
    # door is why a wall-hugging piece stays withheld.
    openings: list[Opening] = Field(default_factory=list)
    # Set when the room is not a plain rectangle (L-shaped, alcoves), which
    # invalidates the wall-anchor maths.
    irregular: bool = False

    # How the floor plane maps into the photo. None when there was no photo,
    # or when the vision model could not identify a floor quad - in which case
    # rendering is impossible and says so rather than guessing a projection.
    camera: CameraCalibration | None = None

    # Furniture already in the photo. Empty means either no photo, or a photo
    # nothing was found in - `source` is what distinguishes those. Populated at
    # analysis time rather than at render time because what the user already
    # owns is a retrieval signal, not just a list of pixels to erase.
    detections: list[Detection] = Field(default_factory=list)

    @property
    def replaceable(self) -> list[Detection]:
        """Detections that map onto a role we actually sell.

        The rest stay in `detections` for context and for the client to show,
        but nothing downstream can erase or replace them.
        """
        return [d for d in self.detections if d.role is not None]

    @property
    def existing_style(self) -> str:
        """A style prior drawn from what is already in the room.

        What someone already owns is a stronger signal for what they want than
        any adjective they type, so this is folded into the retrieval query.
        Captions are ordered biggest-object-first, because a sofa says more
        about a room than a lamp does.
        """
        ranked = sorted(self.detections, key=lambda d: d.area, reverse=True)
        return ". ".join(d.caption for d in ranked if d.caption)[:400]

    @property
    def measured(self) -> bool:
        """Whether a human has vouched for these dimensions.

        Kept as a property so existing callers read the same flag they always
        did. The distinction that matters is not "typed by hand" versus
        "estimated" - it is whether anyone has *looked*. An estimate the user
        glanced at and accepted is trustworthy; an estimate nobody has seen is
        the one that silently produces a wrong layout.
        """
        return self.dimension_source is not DimensionSource.ESTIMATED

    @property
    def dimension_tolerance_cm(self) -> float:
        """How far these dimensions might be out.

        Feeds placement tolerance: furniture positioned inside a room whose own
        size is ±40cm cannot honestly claim ±5cm.
        """
        return {
            DimensionSource.ESTIMATED: 60.0,
            DimensionSource.CONFIRMED: 25.0,
            DimensionSource.MEASURED: 5.0,
        }[self.dimension_source]

    @property
    def supports_exact_placement(self) -> bool:
        """True when the data is good enough to position against walls."""
        return self.measured and not self.irregular


# --- Layout ----------------------------------------------------------------


class Placement(BaseModel):
    """A resolved position. x/y is the top-left corner, origin top-left of room."""

    item_id: str
    name: str
    role: Role
    x_cm: float
    y_cm: float
    # Effective extents after rotation is applied, so a renderer can draw the
    # rect directly without repeating the rotation math.
    w_cm: float
    d_cm: float
    rotation: Literal[0, 90, 180, 270] = 0
    z: int = 1  # 0 renders underneath (rugs); 1 is normal furniture
    swatch: str = "#8B7355"
    price_cents: int = 0
    merchant: str = ""

    # How much to trust this position. A renderer should show anything below
    # HIGH as provisional rather than as a firm plan.
    confidence: Confidence = Confidence.HIGH
    # Rough positional tolerance. "roughly here, give or take 30cm."
    tolerance_cm: float = 0.0
    # Why this position was chosen, in one phrase, for the client to display.
    rationale: str = ""


class SkippedItem(BaseModel):
    """An item that could not be placed. Never overlapped to force a fit."""

    item_id: str
    name: str
    role: Role
    reason: Literal["no_fit", "too_large", "over_budget"]
    detail: str = ""


class WithheldItem(BaseModel):
    """A piece deliberately not placed pending better measurements.

    Distinct from SkippedItem: nothing is wrong with the item or the room, we
    simply will not guess a position that has to be right. Supplying `needs`
    unlocks it.
    """

    item_id: str
    name: str
    role: Role
    reason: str
    needs: list[MeasurementRequest] = Field(default_factory=list)


class LayoutResult(BaseModel):
    room_width_cm: float
    room_depth_cm: float
    placements: list[Placement] = Field(default_factory=list)
    skipped: list[SkippedItem] = Field(default_factory=list)
    withheld: list[WithheldItem] = Field(default_factory=list)

    @property
    def needs_measurements(self) -> bool:
        return bool(self.withheld)


# --- Cart ------------------------------------------------------------------


class CartLine(BaseModel):
    item_id: str
    name: str
    merchant: str
    role: Role
    price_cents: int
    qty: int = 1
    checkout_url: str = ""
    image_url: str = ""

    @property
    def line_total_cents(self) -> int:
        return self.price_cents * self.qty


class Cart(BaseModel):
    lines: list[CartLine] = Field(default_factory=list)
    subtotal_cents: int = 0
    budget_cents: int = 0
    currency: str = "SGD"

    @property
    def over_budget(self) -> bool:
        return self.budget_cents > 0 and self.subtotal_cents > self.budget_cents


# --- Checkout (simulated) --------------------------------------------------


class CheckoutRequest(BaseModel):
    item_ids: list[str]
    session_id: str | None = None


class MerchantGroup(BaseModel):
    merchant: str
    lines: list[CartLine]
    subtotal_cents: int


class CheckoutResult(BaseModel):
    """No payment is processed. Every field here is fabricated for the demo."""

    simulated: Literal[True] = True
    disclaimer: str = (
        "SIMULATED DEMO — no payment was processed and no order was placed."
    )
    order_id: str
    payment_token: str
    groups: list[MerchantGroup]
    total_cents: int
    currency: str = "SGD"


# --- Chat ------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


# --- Payment (simulated Visa rail) -----------------------------------------
#
# Nothing below touches a real payment network. The shapes deliberately mirror
# how a card transaction actually moves - intent, step-up challenge, per-
# merchant authorization - because the point of this flow is to show a user
# how an agent asks permission to spend their money, and a flow that skips the
# steps a real one has would teach the wrong habit.
#
# The invariant the whole module is built around: the agent may ASSEMBLE a
# purchase, but only a human may AUTHORIZE one. Every type here exists to keep
# those two acts separate and legible.


class CardNetwork(str, Enum):
    VISA = "visa"
    MASTERCARD = "mastercard"
    AMEX = "amex"


class PaymentMethod(BaseModel):
    """A stored card, as the user would recognise it.

    Only ever the last four digits. A demo that carried a full PAN around -
    even a fake one - would be modelling the one thing real systems are built
    to never do, and someone would eventually paste a real card into it.
    """

    id: str
    network: CardNetwork = CardNetwork.VISA
    last4: str = Field(min_length=4, max_length=4)
    exp_month: int = Field(ge=1, le=12)
    exp_year: int = Field(ge=2024)
    holder: str
    label: str = ""  # "Personal", "Household" - what the user calls it
    is_default: bool = False
    # Ceiling this card will authorize in one transaction without a step-up
    # challenge. Below it the user still confirms; above it they must also
    # prove they are present.
    step_up_threshold_cents: int = 50_000

    # Whether this card has been enrolled into the (simulated) Visa Token
    # Service. An un-enrolled card can still be used by a human at the
    # confirmation screen; only an enrolled card can back an agent token.
    tokenized: bool = False

    @property
    def display(self) -> str:
        return f"{self.network.value.title()} ···· {self.last4}"


class RiskSignal(BaseModel):
    """One reason this transaction is or is not routine.

    Surfaced to the user rather than kept internal: "why am I being asked to
    verify?" deserves an answer, and an agent that spends money without
    explaining its own risk assessment is asking for blind trust.
    """

    code: Literal[
        "amount_over_threshold",
        "multi_merchant",
        "new_merchant",
        "over_budget",
        "agent_initiated",
        "routine",
        # Visa Agentic Stack signals. Kept in the same list as the others
        # because to the user they are the same kind of thing: a reason this
        # purchase is or is not routine.
        "mandate_scoped",
        "mandate_violation",
        "token_presented",
        "velocity",
    ]
    detail: str
    # True when this signal is why a step-up challenge is required.
    triggers_step_up: bool = False


class MerchantCharge(BaseModel):
    """What one merchant will charge, on which card.

    Items in a single design routinely come from three different shops, so a
    single "total" hides the fact that the user is about to authorize three
    separate charges that will appear as three separate lines on a statement.
    This type is what makes that visible before the fact rather than after.
    """

    merchant: str
    lines: list[CartLine]
    subtotal_cents: int
    shipping_cents: int = 0
    tax_cents: int = 0
    total_cents: int
    # Which stored card pays this merchant. Per-merchant so a user can put the
    # sofa on one card and the rug on another.
    payment_method_id: str
    # Set once authorized.
    auth_code: str | None = None
    status: Literal["pending", "approved", "declined"] = "pending"
    decline_reason: str = ""
    # Fictional, but the user is entitled to know when the thing arrives
    # before they agree to pay for it.
    eta_days: int = 7

    # --- Visa Agentic Stack ------------------------------------------------
    # Merchant category code this charge transacts under. Shown to the user
    # because it is what the mandate's category lock is actually checked
    # against - naming the merchant alone would not explain a refusal.
    mcc: str = ""
    # The network token presented for this leg, never the funding PAN. Its
    # last4 differs from the card's on purpose: that is how a user matches a
    # statement line to the agent that created it.
    token_last4: str = ""
    # Single-use cryptogram id for this leg of the split settlement.
    cryptogram_id: str = ""


class PaymentIntentStatus(str, Enum):
    """Where a transaction sits between "proposed" and "paid".

    REQUIRES_CONFIRMATION is the state that matters: the agent has done all it
    is permitted to do and is now waiting on a human. An intent cannot leave
    that state except by a user action.
    """

    REQUIRES_CONFIRMATION = "requires_confirmation"
    REQUIRES_VERIFICATION = "requires_verification"
    AUTHORIZING = "authorizing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class PaymentIntent(BaseModel):
    """A fully-priced, not-yet-authorized purchase awaiting human approval.

    This is the transaction preview. It is created by the agent, it is
    complete enough that nothing about the charge can change after the user
    reads it, and it is inert: holding one confers no ability to move money.
    """

    simulated: Literal[True] = True
    disclaimer: str = (
        "SIMULATED DEMO - no payment is processed and no order is placed. "
        "Card details shown are fictional test values."
    )

    id: str
    session_id: str | None = None
    status: PaymentIntentStatus = PaymentIntentStatus.REQUIRES_CONFIRMATION
    charges: list[MerchantCharge] = Field(default_factory=list)

    subtotal_cents: int = 0
    shipping_cents: int = 0
    tax_cents: int = 0
    total_cents: int = 0
    currency: str = "SGD"

    # The budget the design was solved against, so the preview can say whether
    # this purchase honours the constraint the user set at the start.
    budget_cents: int = 0

    risk: list[RiskSignal] = Field(default_factory=list)
    requires_step_up: bool = False

    # Who proposed this. "agent" is the honest answer for anything Room Hack
    # assembled on the user's behalf, and it is itself a risk signal.
    initiated_by: Literal["agent", "user"] = "agent"

    # Unix seconds. A preview the user leaves open for an hour should not still
    # be chargeable: prices, stock and intent all go stale.
    created_at: float = 0.0
    expires_at: float = 0.0

    # Set after authorization.
    order_id: str | None = None

    # --- Visa Agentic Stack ------------------------------------------------
    # Which agent token was presented to price this, if any. Absent means the
    # user is checking out by hand, which is always permitted - the mandate
    # constrains the AGENT, not the person.
    agent_token_id: str = ""
    # How this purchase measured against the mandate. Present on every priced
    # intent so the preview can show headroom, not just refusals.
    mandate: "MandateEvaluation | None" = None
    # A mandate violation is fatal to an agent-initiated purchase: it cannot be
    # cleared by verifying identity, because the user already said no in
    # advance. Separate from requires_step_up for exactly that reason.
    mandate_blocked: bool = False
    # Whether the step-up must be a passkey rather than an OTP.
    step_up_method: Literal["passkey", "sms_otp"] = "passkey"

    @property
    def over_budget(self) -> bool:
        return self.budget_cents > 0 and self.total_cents > self.budget_cents

    @property
    def merchant_count(self) -> int:
        return len(self.charges)


class PaymentIntentRequest(BaseModel):
    """Ask the agent to price a purchase. Creates a preview, charges nothing."""

    item_ids: list[str]
    session_id: str | None = None
    # Per-merchant card assignment; merchants absent fall back to the default
    # card. Lets the user split a multi-shop order across cards.
    payment_method_ids: dict[str, str] = Field(default_factory=dict)
    # The signed mandate the agent holds. It is a bearer credential the agent
    # cannot forge or widen: every claim in it is verified server-side, so
    # presenting a tampered one fails rather than raising the agent's limits.
    mandate_credential: str = ""


class VerificationChallenge(BaseModel):
    """A step-up identity check, in the shape of a 3-D Secure prompt.

    The code is delivered out of band in a real system. Here it is returned in
    the response and labelled as such, because a demo that hid it would be
    unusable and pretending otherwise would be the dishonest option.
    """

    simulated: Literal[True] = True
    intent_id: str
    challenge_id: str
    # "passkey" is the default in the Visa Payment Passkey flow; the OTP
    # methods remain only as a fallback for a device with no authenticator.
    method: Literal["passkey", "sms_otp", "app_push"] = "passkey"
    # Masked destination, as a real challenge would show it.
    sent_to: str = "··· ··· ·· 4417"
    # DEMO ONLY. A real challenge never returns its own answer.
    demo_code: str = ""
    expires_at: float = 0.0
    attempts_remaining: int = 3


class VerifyRequest(BaseModel):
    intent_id: str
    challenge_id: str
    code: str


class AuthorizeRequest(BaseModel):
    """The user's explicit instruction to charge. The only way money moves.

    `idempotency_key` is the client's guarantee that a double-click, a retry
    or a flaky connection cannot authorize the same purchase twice - the one
    failure mode of an agent-driven checkout that costs the user real money.
    """

    intent_id: str
    idempotency_key: str
    # Echoed back from the preview the user actually read. If the intent has
    # changed since it was displayed, authorization is refused rather than
    # charging a total the user never saw.
    confirmed_total_cents: int
    # The passkey assertion that proved the cardholder was present, bound to
    # this intent and this amount. Required whenever the mandate demands user
    # presence, which every real mandate does.
    assertion_id: str = ""


class AuthorizationReceipt(BaseModel):
    """Outcome of a simulated authorization, per merchant."""

    simulated: Literal[True] = True
    disclaimer: str = (
        "SIMULATED DEMO - no payment was processed and no order was placed."
    )
    intent_id: str
    order_id: str
    status: PaymentIntentStatus
    charges: list[MerchantCharge] = Field(default_factory=list)
    total_cents: int = 0
    approved_cents: int = 0
    declined_cents: int = 0
    currency: str = "SGD"
    authorized_at: float = 0.0
    # A plain-language record of what the user agreed to and when. An audit
    # trail is what makes an agent's spending reviewable after the fact.
    audit: list[str] = Field(default_factory=list)


# --- Visa Agentic Payments Stack (simulated) --------------------------------
#
# Three layers sit on top of the rail above, and each answers a question the
# rail alone cannot:
#
#   Visa Payment Passkey (FIDO2)  "is the cardholder really here?"
#   Visa Token Service (AI_AGENT) "what is the agent allowed to spend?"
#   Agent mandate                 "and can the user take that back?"
#
# Nothing here contacts Visa. The shapes mirror the real APIs because the
# point is to model how authority is delegated and withdrawn, and a model that
# skipped the parts a real one has would teach the wrong habit.


class PasskeyCredentialSummary(BaseModel):
    """A registered passkey, as the user would recognise it in a settings list.

    Public metadata only. The private key never leaves the device's secure
    enclave, and no biometric data exists on the server at any point - the
    device verifies the face or fingerprint locally and releases a signature.
    """

    credential_id: str
    label: str = "This device"
    created_at: float = 0.0
    sign_count: int = 0
    # Synced to an iCloud/Google keychain, so it survives losing the device.
    backed_up: bool = False
    transports: list[str] = Field(default_factory=list)


class PasskeyRegistrationRequest(BaseModel):
    """The browser's response to navigator.credentials.create()."""

    credential_id: str
    client_data_json: str      # base64url
    attestation_object: str    # base64url
    transports: list[str] = Field(default_factory=list)
    label: str = ""


class PasskeyAssertionRequest(BaseModel):
    """The browser's response to navigator.credentials.get().

    `intent_id` and the amount are echoed so the server can confirm the
    signature it is about to accept was issued for this exact payment. A
    signature for one purchase must never authorize another.
    """

    credential_id: str
    client_data_json: str      # base64url
    authenticator_data: str    # base64url
    signature: str             # base64url
    intent_id: str | None = None
    purpose: Literal["payment", "provisioning"] = "payment"


class MandateScope(BaseModel):
    """The guardrails on an agent token, in the user's terms.

    Every field is a limit the user chose. Together they answer "what is the
    worst this agent can do with my money" with a number instead of a hope.
    """

    per_transaction_cap_cents: int
    cumulative_cap_cents: int
    spent_cents: int = 0
    remaining_cents: int = 0
    allowed_mccs: list[str] = Field(default_factory=list)
    category_label: str = "Furniture & Home Decor"
    allowed_merchants: list[str] = Field(default_factory=list)
    max_merchants_per_transaction: int = 5
    require_user_presence: bool = True
    expires_at: float = 0.0


class AgentTokenSummary(BaseModel):
    """A provisioned AI_AGENT network token and the mandate scoping it."""

    simulated: Literal[True] = True

    token_id: str
    funding_method_id: str
    funding_display: str = ""
    # Deliberately not the card's last4. A network token is a different number
    # for the same funding account, which is what makes it revocable alone.
    token_last4: str
    presentation_type: Literal["AI_AGENT", "ECOMMERCE"] = "AI_AGENT"
    status: Literal["active", "suspended", "revoked", "expired"] = "active"
    mandate_id: str
    scope: MandateScope
    created_at: float = 0.0
    revoked_at: float | None = None
    revocation_reason: str = ""
    # Token Assurance Level. A device biometric bound to a hardware key is the
    # highest band; a knowledge factor like an OTP is materially lower.
    assurance_level: int = 0
    assurance_method: str = "none"
    # Spend history against this mandate, for after-the-fact review.
    uses: list[dict] = Field(default_factory=list)


class ProvisionTokenRequest(BaseModel):
    """Enroll a card and mint a scoped agent token.

    Requires a fresh passkey assertion: granting an agent standing permission
    to spend is exactly as sensitive as the spending it later permits, so it
    demands the same proof of presence.
    """

    funding_method_id: str
    per_transaction_cap_cents: int = Field(gt=0)
    cumulative_cap_cents: int = Field(gt=0)
    allowed_merchants: list[str] = Field(default_factory=list)
    max_merchants_per_transaction: int = Field(default=5, ge=1, le=20)
    ttl_hours: int = Field(default=24, ge=1, le=720)
    # The assertion id returned by a just-completed passkey verification.
    assertion_id: str = ""


class MandateEvaluation(BaseModel):
    """How a proposed purchase measures against the mandate.

    Returned on every preview, passing or failing. A guardrail display that
    only appears when something is wrong teaches users to fear it rather than
    to read it; showing remaining headroom on a routine purchase is what makes
    the limit legible.
    """

    ok: bool = True
    token_id: str = ""
    amount_cents: int = 0
    per_transaction_cap_cents: int = 0
    cumulative_cap_cents: int = 0
    spent_cents: int = 0
    remaining_cents: int = 0
    allowed_mccs: list[str] = Field(default_factory=list)
    merchant_mccs: dict[str, str] = Field(default_factory=dict)
    expires_at: float = 0.0
    violations: list[dict[str, str]] = Field(default_factory=list)


class RevokeMandateRequest(BaseModel):
    token_id: str
    reason: str = "revoked by user"
