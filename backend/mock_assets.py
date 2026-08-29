"""Generator for the synthetic room plates used as test fixtures.

WHAT THESE ARE. Four empty rooms drawn with Pillow primitives - walls, a
perspective floor, skirting, doors and windows at their real offsets. Nothing
is downloaded and no image model is called, so they regenerate identically on
any machine with Pillow.

WHY THEY ARE GENERATED RATHER THAN PHOTOGRAPHED. The floor quad is *known*
rather than estimated: the same projection that drew the walls produces the
calibration, so the render path can be tested against a camera that is exactly
right. A photo's quad is always a guess. This is also what lets these fixtures
honestly set `measured=True` - the dimensions are not read off the image, they
are the numbers the image was drawn from - which makes them the only fixtures
that exercise the EXACT precision tier.

WHAT THEY ARE NOT. Photorealistic, or a substitute for a real room photo in
anything user-facing. They are diagrams with perspective.

Product imagery is NOT generated here: the catalog is real IKEA listings with
real product photos (see seed_data.py). This module used to draw schematic
product plates as well, which became dead weight the moment real photography
was available.

Run `python -m backend.mock_assets` to (re)generate them into
`backend/assets/rooms/`.
"""

from __future__ import annotations

import colorsys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from .models import CameraCalibration, Confidence, FloorQuad, Opening, Wall

# Rendered at 2x then downsampled, which is the cheapest way to get clean edges
# out of Pillow's un-antialiased polygon fills.
SUPERSAMPLE = 2

ASSET_DIR = Path(__file__).resolve().parent / "assets"
ROOM_DIR = ASSET_DIR / "rooms"

Color = tuple[int, int, int]


# --- colour helpers --------------------------------------------------------


def _shade(rgb: Color, factor: float) -> Color:
    """Lighten (factor > 1) or darken (factor < 1) in HLS.

    Done in HLS rather than by scaling RGB so a saturated colour keeps its hue
    instead of washing toward grey as it lightens.
    """
    r, g, b = (c / 255 for c in rgb)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0.0, min(1.0, l * factor))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (round(r * 255), round(g * 255), round(b * 255))


# --- room plates -----------------------------------------------------------


class MockRoom:
    """An empty-room photo paired with the ground truth used to draw it.

    The point of generating rooms rather than shooting them is that the floor
    quad is *known* rather than estimated: the same projection that drew the
    walls produces the quad, so the render path can be tested against a
    calibration that is exactly right. A photo's quad is always a guess.
    """

    def __init__(self, slug: str, width_cm: float, depth_cm: float,
                 wall: Color, floor: Color, openings: list[Opening],
                 flooring: str, wall_color: str, notes: str = "") -> None:
        self.slug = slug
        self.width_cm = width_cm
        self.depth_cm = depth_cm
        self.wall = wall
        self.floor = floor
        self.openings = openings
        self.flooring = flooring
        self.wall_color = wall_color
        self.notes = notes
        self.near_depth_cm = depth_cm - self.CAMERA_STANDOFF_CM

    # How much of the floor the camera actually sees. These are room-depth
    # COORDINATES, not distances from the camera: the visible floor runs from
    # the back wall (far_depth_cm = 0) forward to near_depth_cm. A photographer
    # standing 80cm inside the near wall cannot see the floor at their feet,
    # so near_depth_cm is the room's depth minus that 80cm - computed per room
    # in __init__ rather than fixed, since it depends on the room's size.
    CAMERA_STANDOFF_CM = 80.0
    far_depth_cm = 0.0

    # Normalized image-space floor quad. Fixed across rooms because all mock
    # rooms use the same virtual camera; only the room's metric size changes.
    QUAD = FloorQuad(
        near_left=(0.02, 0.98),
        near_right=(0.98, 0.98),
        far_right=(0.78, 0.56),
        far_left=(0.22, 0.56),
    )

    def calibration(self) -> CameraCalibration:
        return CameraCalibration(
            quad=self.QUAD,
            horizon_y=0.44,
            source="mock",
            near_depth_cm=self.near_depth_cm,
            far_depth_cm=self.far_depth_cm,
            # Synthetic, so the quad is exact rather than inferred. HIGH is
            # honest here in a way it would never be for a real photo.
            confidence=Confidence.HIGH,
        )


def render_room(room: MockRoom, size: tuple[int, int] = (1280, 960)) -> Image.Image:
    """Draw an empty room whose floor matches `room.QUAD` exactly."""
    S = SUPERSAMPLE
    W, H = size[0] * S, size[1] * S
    img = Image.new("RGB", (W, H), room.wall)
    d = ImageDraw.Draw(img)

    q = room.QUAD
    px = lambda p: (p[0] * W, p[1] * H)
    nl, nr, fr, fl = px(q.near_left), px(q.near_right), px(q.far_right), px(q.far_left)

    # Back wall sits above the quad's far edge.
    back_top = H * 0.10
    d.polygon([fl, fr, (fr[0], back_top), (fl[0], back_top)],
              fill=_shade(room.wall, 1.04))
    # Side walls, darker, so the corners read.
    d.polygon([nl, fl, (fl[0], back_top), (0, 0)], fill=_shade(room.wall, 0.90))
    d.polygon([nr, fr, (fr[0], back_top), (W, 0)], fill=_shade(room.wall, 0.94))
    # Ceiling.
    d.polygon([(0, 0), (W, 0), (fr[0], back_top), (fl[0], back_top)],
              fill=_shade(room.wall, 1.08))

    # Floor.
    d.polygon([nl, nr, fr, fl], fill=room.floor)

    # Floorboards running away from the camera, converging on the vanishing
    # point, which is what sells a floor as a floor.
    plank = _shade(room.floor, 0.93)
    for i in range(1, 14):
        f = i / 14
        d.line([
            (nl[0] + (nr[0] - nl[0]) * f, nl[1]),
            (fl[0] + (fr[0] - fl[0]) * f, fl[1]),
        ], fill=plank, width=max(1, S))
    # Cross-boards, spaced by perspective so they crowd toward the back wall.
    for i in range(1, 9):
        f = (i / 9) ** 1.7
        y = nl[1] + (fl[1] - nl[1]) * f
        lx = nl[0] + (fl[0] - nl[0]) * f
        rx = nr[0] + (fr[0] - nr[0]) * f
        d.line([(lx, y), (rx, y)], fill=plank, width=max(1, S))

    # Skirting along the back wall.
    d.rectangle([fl[0], fl[1] - H * 0.018, fr[0], fl[1]], fill=_shade(room.wall, 0.82))

    _draw_openings(d, room, (fl, fr), back_top, (W, H))

    # Soft vignette; a flat render reads as a diagram, and the render engine's
    # downstream steps expect photo-like tonal falloff.
    img = _vignette(img)
    return img.resize(size, Image.LANCZOS)


def _draw_openings(d, room: MockRoom, far_edge, back_top: float, dims) -> None:
    """Doors and windows on the back wall, positioned from their real offsets."""
    fl, fr = far_edge
    W, H = dims
    wall_span = fr[0] - fl[0]
    floor_y, ceil_y = fl[1], back_top

    for op in room.openings:
        # Only the focal (back) wall is drawn face-on; side-wall openings would
        # need real perspective and are left out of the plate rather than
        # faked at the wrong angle.
        if op.wall is not Wall.NORTH:
            continue
        x0 = fl[0] + wall_span * (op.offset_cm / room.width_cm)
        x1 = fl[0] + wall_span * ((op.offset_cm + op.width_cm) / room.width_cm)
        if op.kind == "window":
            top = ceil_y + (floor_y - ceil_y) * 0.14
            bot = ceil_y + (floor_y - ceil_y) * 0.66
            d.rectangle([x0, top, x1, bot], fill=(226, 232, 236),
                        outline=(255, 255, 255), width=max(2, int(W * 0.004)))
            d.line([((x0 + x1) / 2, top), ((x0 + x1) / 2, bot)],
                   fill=(255, 255, 255), width=max(2, int(W * 0.003)))
            # Light spill on the floor below the window.
            d.polygon([
                (x0, floor_y), (x1, floor_y),
                (x1 + (x1 - x0) * 0.30, floor_y + H * 0.10),
                (x0 - (x1 - x0) * 0.30, floor_y + H * 0.10),
            ], fill=_shade(room.floor, 1.07))
        else:
            top = ceil_y + (floor_y - ceil_y) * 0.18
            d.rectangle([x0, top, x1, floor_y], fill=(64, 62, 60))
            d.rectangle([x0, top, x1, floor_y], outline=(255, 255, 255),
                        width=max(2, int(W * 0.004)))


def _vignette(img: Image.Image) -> Image.Image:
    W, H = img.size
    mask = Image.new("L", (W, H), 0)
    md = ImageDraw.Draw(mask)
    md.ellipse([-W * 0.20, -H * 0.20, W * 1.20, H * 1.20], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(min(W, H) * 0.10))
    dark = Image.new("RGB", (W, H), (0, 0, 0))
    return Image.composite(img, Image.blend(img, dark, 0.35), mask)


# The rooms themselves. Sizes and openings are plausible for their type and,
# unlike a photo estimate, are exact by construction - these ARE the numbers
# the image was drawn from, which is what makes them usable as `measured=True`.
MOCK_ROOMS: list[MockRoom] = [
    MockRoom(
        "loft-open", 520, 420,
        wall=(238, 235, 229), floor=(196, 166, 124),
        openings=[
            Opening(kind="window", wall=Wall.NORTH, offset_cm=90, width_cm=180),
            Opening(kind="door", wall=Wall.SOUTH, offset_cm=40, width_cm=90, swing_cm=90),
        ],
        flooring="light oak", wall_color="warm white",
        notes="Open-plan loft corner, large north window, generous circulation.",
    ),
    MockRoom(
        "apartment-narrow", 340, 460,
        wall=(232, 230, 228), floor=(168, 140, 106),
        openings=[
            Opening(kind="window", wall=Wall.NORTH, offset_cm=120, width_cm=110),
            Opening(kind="door", wall=Wall.WEST, offset_cm=30, width_cm=80, swing_cm=80),
        ],
        flooring="mid oak", wall_color="soft grey",
        notes="Narrow city apartment; depth well exceeds width, so a long wall "
              "is the only viable sofa run.",
    ),
    MockRoom(
        "studio-square", 400, 400,
        wall=(240, 238, 233), floor=(210, 202, 190),
        openings=[
            Opening(kind="door", wall=Wall.NORTH, offset_cm=150, width_cm=90, swing_cm=90),
        ],
        flooring="pale concrete", wall_color="chalk white",
        notes="Square studio with a door on the focal wall - the swing "
              "clearance rules out centring a console there.",
    ),
    MockRoom(
        "cottage-small", 300, 330,
        wall=(236, 228, 214), floor=(150, 116, 82),
        openings=[
            Opening(kind="window", wall=Wall.NORTH, offset_cm=60, width_cm=90),
            Opening(kind="window", wall=Wall.NORTH, offset_cm=190, width_cm=90),
            Opening(kind="door", wall=Wall.EAST, offset_cm=20, width_cm=76, swing_cm=76),
        ],
        flooring="dark walnut", wall_color="cream",
        notes="Tight cottage room; twin windows leave little uninterrupted wall.",
    ),
]


def room_analysis(room: MockRoom):
    """The RoomAnalysis a perfect vision model would return for this plate.

    `measured=True` is the unusual claim here, and it is defensible only
    because these rooms are synthetic: the dimensions are not estimated from
    the image, they are the numbers the image was *drawn from*. That makes
    these fixtures the only ones in the project that legitimately exercise the
    EXACT precision tier - wall-hugging sofas and consoles - which a photo
    estimate can never unlock.
    """
    from .models import DimensionSource, RoomAnalysis

    return RoomAnalysis(
        width_cm=room.width_cm,
        depth_cm=room.depth_cm,
        focal_wall=Wall.NORTH,
        wall_color=room.wall_color,
        flooring=room.flooring,
        lighting="even, diffuse",
        notes=room.notes,
        source="mock",
        # MEASURED, not ESTIMATED: these dimensions are not read off the image,
        # they are the numbers the image was drawn from. `measured` derives
        # from this, and it is the gate on exact-tier placement.
        dimension_source=DimensionSource.MEASURED,
        openings=list(room.openings),
        irregular=False,
        camera=room.calibration(),
    )


def load_room_fixture(slug: str) -> tuple[str, object]:
    """A mock room as (base64 PNG, RoomAnalysis), ready to feed the pipeline.

    Raises if the plate has not been generated yet rather than silently
    returning an analysis with no matching image - a mismatch between the two
    is exactly the bug this fixture set exists to rule out.
    """
    import base64

    room = next((r for r in MOCK_ROOMS if r.slug == slug), None)
    if room is None:
        raise KeyError(f"unknown mock room {slug!r}; have "
                       f"{[r.slug for r in MOCK_ROOMS]}")
    path = ROOM_DIR / f"{slug}.png"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing - run `python -m backend.mock_assets` first"
        )
    return base64.b64encode(path.read_bytes()).decode(), room_analysis(room)


# --- generation ------------------------------------------------------------


def generate_all(verbose: bool = True) -> dict[str, int]:
    """Write every room plate. Safe to re-run; output is deterministic."""
    ROOM_DIR.mkdir(parents=True, exist_ok=True)

    for room in MOCK_ROOMS:
        render_room(room).save(ROOM_DIR / f"{room.slug}.png")
        if verbose:
            print(f"  room  {room.slug:<18} {room.width_cm:.0f}x{room.depth_cm:.0f}cm")

    return {"rooms": len(MOCK_ROOMS)}


if __name__ == "__main__":
    counts = generate_all()
    print(f"\nwrote {counts['rooms']} room plates -> {ROOM_DIR}")
