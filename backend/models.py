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
    TV_UNIT = "tv_unit"
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
    Role.TV_UNIT,
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
    # Wall-hugging pieces: a gap behind them reads as a mistake, and they are
    # the pieces most likely to foul a door swing or a window.
    Role.SOFA: Precision.EXACT,
    Role.TV_UNIT: Precision.EXACT,
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
    not the image model - decides position. `quad` is assumed to bound the
    full usable floor: its near edge is y=depth_cm, its far edge y=0.
    """

    quad: FloorQuad
    horizon_y: float = Field(ge=0, le=1)
    source: Literal["openai", "mock"] = "mock"

    # A calibration derived from a photo estimate is itself an estimate. This
    # caps how much a render may claim, independent of the layout's own
    # confidence.
    confidence: Confidence = Confidence.MEDIUM


# --- Detection -------------------------------------------------------------


class Detection(BaseModel):
    """An existing piece of furniture found in the user's photo.

    Boxes are normalized [0,1] image coordinates, x1/y1 top-left. Detections
    drive two things: what to erase before compositing, and a prior on what
    the user already owns.
    """

    role: Role
    label: str
    score: float = Field(ge=0, le=1)
    x1: float
    y1: float
    x2: float
    y2: float
    # Populated once segmentation runs; a base64 PNG of the binary mask.
    mask_b64: str | None = None


# --- Render ----------------------------------------------------------------


class RenderMethod(str, Enum):
    """How a render image was produced. Surfaced so a client can label it."""

    # SDXL inpaint conditioned on solver depth plus the product image.
    GENERATIVE = "generative"
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
        "no_photo", "no_calibration", "not_placed", "provider_error", "no_product_image"
    ]
    detail: str = ""


class RenderRequest(BaseModel):
    session_id: str
    # Restrict to specific items; empty means every placed item.
    item_ids: list[str] = Field(default_factory=list)


# --- Catalog ---------------------------------------------------------------


class Dimensions(BaseModel):
    width_cm: float = Field(gt=0)
    depth_cm: float = Field(gt=0)
    height_cm: float = Field(gt=0)


class CatalogItem(BaseModel):
    """One purchasable item. Merchants are fictional; see seed_data.py."""

    id: str
    merchant: str
    title: str
    role: Role
    price_cents: int = Field(ge=0)
    currency: str = "USD"
    dimensions: Dimensions
    materials: list[str] = Field(default_factory=list)
    primary_color: str
    swatch: str  # hex, for rendering the piece on a floor plan
    style_tags: list[str] = Field(default_factory=list)
    image_url: str
    checkout_url: str
    in_stock: bool = True

    def embed_text(self) -> str:
        """The text embedded into the vector index for this item."""
        d = self.dimensions
        return (
            f"{self.title}. {self.role.value.replace('_', ' ')}. "
            f"Style: {', '.join(self.style_tags)}. "
            f"Color: {self.primary_color}. "
            f"Materials: {', '.join(self.materials)}. "
            f"Dimensions {d.width_cm:.0f}x{d.depth_cm:.0f}x{d.height_cm:.0f} cm. "
            f"Sold by {self.merchant}."
        )


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
    # "openai" or "mock" — surfaced to the client so a demo can honestly show
    # which path produced the analysis.
    source: Literal["openai", "mock"] = "mock"

    # How the dimensions were obtained. A photo estimate can be metres off, so
    # it cannot support wall-hugging placement; a user-supplied or floor-plan
    # measurement can. This is the gate on the EXACT precision tier.
    measured: bool = False
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
    currency: str = "USD"

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
    currency: str = "USD"


# --- Chat ------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
