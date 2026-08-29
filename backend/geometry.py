"""Floor-plane projection: solver centimetres to photo pixels.

The solver decides where furniture goes; this module decides where that is in
the user's photo. A room floor is planar, so the map from floor coordinates to
image coordinates is a homography - eight unknowns, recoverable from the four
corner correspondences a CameraCalibration provides.

Pure Python and dependency-free: an 8x8 solve is trivial and pulling numpy in
for it would be the largest dependency in the project.

Floor coordinates match the solver exactly: origin top-left, x -> right across
the room's width, y -> away from the camera across its depth. The quad's near
edge (closest to the camera, lowest in frame) is the larger y.

A photo almost never shows the whole floor - you cannot photograph the floor
you are standing on - so the quad spans only part of the room's depth, and the
calibration says which part. Fitting as though it covered everything stretches
the homography over floor that was never in frame and slides near-wall
furniture out of the bottom of the picture.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import CameraCalibration

# Below this the 8x8 system is effectively singular - a collapsed or
# self-intersecting quad. Better to refuse than to emit garbage coordinates.
_SINGULAR_EPS = 1e-9

# How far past the photographed floor a piece's BACK edge may sit and still be
# drawn. Small: past this the piece really is behind the camera, not merely
# cropped by the bottom of the frame.
_BACK_EDGE_TOLERANCE_CM = 30.0


class ProjectionError(ValueError):
    """The calibration does not describe a usable floor plane."""


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float] | None:
    """Gaussian elimination with partial pivoting. None if singular."""
    n = len(matrix)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < _SINGULAR_EPS:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]

        inv = 1.0 / aug[col][col]
        for row in range(col + 1, n):
            factor = aug[row][col] * inv
            if factor == 0.0:
                continue
            for k in range(col, n + 1):
                aug[row][k] -= factor * aug[col][k]

    out = [0.0] * n
    for row in range(n - 1, -1, -1):
        total = aug[row][n] - sum(aug[row][k] * out[k] for k in range(row + 1, n))
        out[row] = total / aug[row][row]
    return out


@dataclass(frozen=True)
class FloorProjection:
    """Maps room floor centimetres onto normalized image coordinates.

    `h` is the row-major homography with h33 fixed at 1.
    """

    h: tuple[float, ...]
    room_width_cm: float
    room_depth_cm: float
    # The depth range the source photo actually shows. Anything nearer than
    # near_depth_cm was never in frame, so a placement there cannot be drawn.
    near_depth_cm: float = 0.0
    far_depth_cm: float = 0.0

    @classmethod
    def from_calibration(
        cls, camera: CameraCalibration, width_cm: float, depth_cm: float
    ) -> FloorProjection:
        """Fit the homography from the four floor-corner correspondences.

        The quad spans only the floor the camera can see, so its near edge maps
        to `near_depth_cm` rather than to the room's near wall. Fitting against
        the full depth instead is what pushes furniture out of the bottom of
        the frame - the homography is then stretched over floor that was never
        photographed.
        """
        if width_cm <= 0 or depth_cm <= 0:
            raise ProjectionError("room dimensions must be positive")

        near_y, far_y = camera.depth_span(depth_cm)
        if near_y - far_y < 1.0:
            raise ProjectionError(
                f"floor quad spans only {near_y - far_y:.0f}cm of room depth; "
                "too little to fit a projection"
            )

        # Source: floor cm. Destination: normalized image. The quad is listed
        # near-left, near-right, far-right, far-left, and "near" is the camera
        # side, which is the larger y in solver coordinates.
        src = [
            (0.0, near_y),
            (width_cm, near_y),
            (width_cm, far_y),
            (0.0, far_y),
        ]
        dst = camera.quad.as_list()

        rows: list[list[float]] = []
        rhs: list[float] = []
        for (x, y), (u, v) in zip(src, dst):
            rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
            rhs.append(u)
            rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
            rhs.append(v)

        solved = _solve(rows, rhs)
        if solved is None:
            raise ProjectionError("floor quad is degenerate; cannot fit a projection")

        return cls(
            h=tuple(solved) + (1.0,),
            room_width_cm=width_cm,
            room_depth_cm=depth_cm,
            near_depth_cm=near_y,
            far_depth_cm=far_y,
        )

    def project(self, x_cm: float, y_cm: float) -> tuple[float, float]:
        """One floor point to normalized image coordinates.

        Points at or behind the horizon have a non-positive homogeneous w. That
        is geometrically meaningless rather than merely out of frame, so it
        raises instead of returning a plausible-looking wrong answer.
        """
        a, b, c, d, e, f, g, i, _ = self.h
        w = g * x_cm + i * y_cm + 1.0
        if abs(w) < _SINGULAR_EPS or w < 0:
            raise ProjectionError(
                f"floor point ({x_cm:.0f},{y_cm:.0f})cm projects behind the camera"
            )
        return ((a * x_cm + b * y_cm + c) / w, (d * x_cm + e * y_cm + f) / w)

    def visible_fraction(self, y_cm: float, d_cm: float) -> float:
        """How much of a footprint's depth falls inside the photographed floor.

        Reported for honesty - it drives confidence, not the decision to draw.
        """
        if d_cm <= 0:
            return 1.0 if y_cm <= self.near_depth_cm else 0.0
        visible = min(y_cm + d_cm, self.near_depth_cm) - y_cm
        return max(0.0, min(visible / d_cm, 1.0))

    def is_visible(self, y_cm: float, d_cm: float = 0.0) -> bool:
        """Whether a piece at this footprint appears in the photo at all.

        The test is the BACK edge, not the fraction of floor in frame. A sofa
        against the near wall typically has only a sliver of its footprint
        inside the photographed floor - the camera cannot see the carpet under
        the front of the seat - yet the sofa itself is plainly in the picture,
        occupying most of the foreground. Judging it by footprint coverage
        drops the single piece the user most wants to see.

        So a piece renders when its back edge was photographed. Its front may
        extend past the bottom of the frame, which is exactly what furniture
        does in a real room photo.
        """
        return y_cm <= self.near_depth_cm + _BACK_EDGE_TOLERANCE_CM

    def project_footprint(
        self, x_cm: float, y_cm: float, w_cm: float, d_cm: float
    ) -> list[tuple[float, float]]:
        """A placement's floor rectangle as four projected image points.

        Corners run near-left, near-right, far-right, far-left so a renderer
        can fill the polygon directly. Perspective is preserved: the near edge
        comes out wider than the far one.
        """
        near_y, far_y = y_cm + d_cm, y_cm
        if not self.is_visible(y_cm, d_cm):
            raise ProjectionError(
                f"a piece starting at {y_cm:.0f}cm sits in front of the "
                f"photographed floor (visible to {self.near_depth_cm:.0f}cm of "
                f"{self.room_depth_cm:.0f}cm)"
            )
        return [
            self.project(x_cm, near_y),
            self.project(x_cm + w_cm, near_y),
            self.project(x_cm + w_cm, far_y),
            self.project(x_cm, far_y),
        ]

    def scale_at(self, x_cm: float, y_cm: float) -> float:
        """Normalized image units per centimetre near a floor point.

        Perspective makes this depth-dependent: the same sofa covers far more
        pixels at the front of the room than the back. Measured over a 10cm
        step so it stays a local estimate rather than a room-wide average.
        """
        step = 10.0
        x0, _ = self.project(x_cm, y_cm)
        x1, _ = self.project(min(x_cm + step, self.room_width_cm), y_cm)
        return abs(x1 - x0) / step

    def height_offset(self, x_cm: float, y_cm: float, height_cm: float) -> float:
        """How far up the image a point `height_cm` above the floor sits.

        A floor plane homography only maps the floor, so vertical extent has to
        be approximated. Using the local scale is not exact - true vertical
        foreshortening depends on focal length - but it is stable, monotonic in
        depth, and good enough to size a bounding box for an inpaint mask.
        """
        return self.scale_at(x_cm, y_cm) * height_cm

    def item_box(
        self,
        x_cm: float,
        y_cm: float,
        w_cm: float,
        d_cm: float,
        height_cm: float,
    ) -> tuple[float, float, float, float]:
        """Image-space bounding box (x1, y1, x2, y2) enclosing a placed item.

        The footprint gives the floor extent; the height lifts the top edge.
        This is what an inpaint mask is cut from.
        """
        footprint = self.project_footprint(x_cm, y_cm, w_cm, d_cm)
        xs = [p[0] for p in footprint]
        ys = [p[1] for p in footprint]

        # Lift from the far edge: it is the highest point of the floor rect, so
        # the box covers the whole piece rather than clipping its back.
        lift = self.height_offset(x_cm + w_cm / 2, y_cm, height_cm)
        return (min(xs), min(ys) - lift, max(xs), max(ys))


def selftest() -> None:
    """Check the projection against a known camera."""
    from .models import Confidence, FloorQuad

    camera = CameraCalibration(
        quad=FloorQuad(
            near_left=(0.02, 0.98),
            near_right=(0.98, 0.98),
            far_right=(0.72, 0.44),
            far_left=(0.28, 0.44),
        ),
        horizon_y=0.38,
        source="mock",
        confidence=Confidence.MEDIUM,
    )
    W, D = 420.0, 330.0
    # This camera sees from the back wall to 80cm short of the near wall, which
    # is what a phone photo from a doorway actually captures.
    camera = camera.model_copy(update={"near_depth_cm": D - 80.0})
    proj = FloorProjection.from_calibration(camera, W, D)
    NEAR = D - 80.0

    # 1. Corners round-trip to the quad they were fitted from - the near edge
    # to the visible depth, not to the room's near wall.
    for (x, y), expect in zip(
        [(0, NEAR), (W, NEAR), (W, 0), (0, 0)], camera.quad.as_list()
    ):
        got = proj.project(x, y)
        assert abs(got[0] - expect[0]) < 1e-6 and abs(got[1] - expect[1]) < 1e-6, (
            f"corner ({x},{y}) -> {got}, expected {expect}"
        )

    # 2. Perspective is the right way round: the far edge is narrower and
    # higher in frame than the near edge.
    near = proj.project(W / 2, NEAR)
    far = proj.project(W / 2, 0)
    assert far[1] < near[1], "far edge should sit higher in the image"
    assert proj.scale_at(W / 2, 0) < proj.scale_at(W / 2, NEAR), (
        "objects at the back should render smaller"
    )

    # 3. Floor nearer than the camera saw is refused, not projected off-frame.
    # This is the bug that slid furniture out of the bottom of the picture.
    assert proj.is_visible(NEAR) and not proj.is_visible(D + 100), (
        "visibility should end past the photographed near edge"
    )
    # A sofa against the near wall is normally cropped, not absent: most of it
    # is in frame, so it must still render.
    # Its back edge is in frame even though most of its footprint is not, and
    # the fraction is reported honestly for confidence scoring.
    assert proj.is_visible(D - 88, 88), "cropped sofa should still render"
    assert proj.visible_fraction(D - 88, 88) < 0.25, "most of its floor is cropped"
    proj.project_footprint(100, D - 88, 200, 88)

    # A piece whose back edge is past the photographed floor cannot be drawn.
    assert not proj.is_visible(D + 50, 80)
    try:
        proj.project_footprint(100, D + 50, 200, 80)
    except ProjectionError:
        pass
    else:
        raise AssertionError("a placement outside the photo should have raised")

    # A calibration that omits the span still covers the whole room, which is
    # correct when the photo genuinely shows all of the floor.
    full = FloorProjection.from_calibration(
        camera.model_copy(update={"near_depth_cm": None}), W, D
    )
    assert full.is_visible(D), "an unspanned quad should reach the near wall"
    assert full.visible_fraction(0, D) == 1.0, "full-floor quad should see all depth"

    # 4. A footprint stays ordered and convex-ish: near edge wider than far.
    fp = proj.project_footprint(100, 100, 200, 90)
    near_w = abs(fp[1][0] - fp[0][0])
    far_w = abs(fp[2][0] - fp[3][0])
    assert near_w > far_w, "footprint lost its perspective"

    # 5. A bounding box lifts above the floor and contains the footprint.
    x1, y1, x2, y2 = proj.item_box(100, 100, 200, 90, 78)
    assert y1 < min(p[1] for p in fp), "box top should clear the floor rect"
    assert x1 <= min(p[0] for p in fp) and x2 >= max(p[0] for p in fp)

    # 6. A degenerate quad is refused rather than silently fitted.
    bad = camera.model_copy(
        update={
            "quad": FloorQuad(
                near_left=(0.5, 0.5),
                near_right=(0.5, 0.5),
                far_right=(0.5, 0.5),
                far_left=(0.5, 0.5),
            )
        }
    )
    try:
        FloorProjection.from_calibration(bad, W, D)
    except ProjectionError:
        pass
    else:
        raise AssertionError("degenerate quad should have raised")

    print(f"near-edge scale: {proj.scale_at(W/2, NEAR):.5f} img-units/cm")
    print(f"far-edge  scale: {proj.scale_at(W/2, 0):.5f} img-units/cm")
    print("all projection invariants hold")


if __name__ == "__main__":
    selftest()
