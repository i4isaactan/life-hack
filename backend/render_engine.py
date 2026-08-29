"""Tier 2 rendering: replace the furniture in a room photo with catalog items.

The pipeline, per room:

    detect -> segment -> erase          (once, cached as an empty-room plate)
    depth-condition -> inpaint          (per item, against that plate)

The important structural claim is that the *solver* decides position, not the
image model. A placement already carries exact floor coordinates; geometry.py
projects them into the photo; the renderer hands the model a depth map and a
mask cut from that projection. The model's job is appearance - materials,
lighting, contact shadow - never composition.

Two providers, the same discipline as rag_engine:

    RenderProvider   Replicate: Grounding DINO, SAM 2, LaMa, SDXL inpaint
                     offline:   Pillow schematic over the projected footprint

Both return RenderResult. The offline path exercises every coordinate
transform the real one does, so a projection bug surfaces without a key.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
from dataclasses import dataclass, field

from . import config
from .geometry import FloorProjection, ProjectionError
from .models import (
    CatalogItem,
    Confidence,
    Detection,
    LayoutResult,
    Placement,
    RenderFailure,
    RenderMethod,
    RenderResult,
    RoomAnalysis,
    Role,
)

log = logging.getLogger(__name__)

# What to hand Grounding DINO for each role we might need to erase. The model
# takes free text, and the plain English word outperforms our enum's snake_case.
_DETECTION_PROMPTS: dict[Role, str] = {
    Role.SOFA: "sofa . couch . loveseat",
    Role.COFFEE_TABLE: "coffee table",
    Role.RUG: "rug . carpet",
    Role.TV_UNIT: "tv stand . media console . sideboard",
    Role.ACCENT_CHAIR: "armchair . accent chair",
    Role.FLOOR_LAMP: "floor lamp",
}

# Held out of the erase pass. A rug lies flat under everything, so removing it
# tears up the floor the other pieces stand on - and a replacement rug is
# composited over the plate anyway.
_ERASE_EXCLUDED: frozenset[Role] = frozenset({Role.RUG})


def _data_uri(img, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"


def _decode(image_b64: str):
    from PIL import Image

    return Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")


@dataclass
class RoomPlate:
    """A room photo with its original furniture erased, plus what was found.

    Cached per session: erasing is the expensive irreversible step, and every
    candidate replacement renders against the same clean plate.
    """

    plate_b64: str
    detections: list[Detection] = field(default_factory=list)
    # False when erasing was skipped or failed; the plate is then the original
    # photo, and renders will show old furniture behind the new piece.
    erased: bool = False

    def find(self, role: Role) -> Detection | None:
        """The strongest detection for a role, if any."""
        matches = [d for d in self.detections if d.role is role]
        return max(matches, key=lambda d: d.score) if matches else None


def build_prompt(item: CatalogItem, room: RoomAnalysis) -> str:
    """Text conditioning for one item, from catalog and room facts.

    Deliberately close to CatalogItem.embed_text: the fields that describe an
    item well for retrieval describe it well for a diffusion model too.
    """
    parts = [
        item.title,
        item.role.value.replace("_", " "),
        f"{item.primary_color} {', '.join(item.materials)}" if item.materials else item.primary_color,
    ]
    if item.style_tags:
        parts.append(f"{', '.join(item.style_tags)} style")
    parts.append(f"in a room with {room.wall_color} walls and {room.flooring} floor")
    parts.append(room.lighting)
    parts.append("photorealistic interior photograph, natural lighting, sharp focus")
    return ", ".join(p for p in parts if p)


NEGATIVE_PROMPT = (
    "cartoon, illustration, render, cgi, distorted proportions, floating "
    "furniture, duplicate furniture, watermark, text, blurry"
)


def _render_confidence(placement: Placement, room: RoomAnalysis) -> Confidence:
    """A render is never more trustworthy than what it was derived from.

    Three things cap it: the placement's own confidence, whether the room was
    measured, and the calibration's. Taking the weakest is the honest answer -
    a perfectly-placed sofa projected through a guessed camera is still a guess.
    """
    ladder = [Confidence.LOW, Confidence.MEDIUM, Confidence.HIGH]
    caps = [placement.confidence]
    caps.append(Confidence.HIGH if room.measured else Confidence.MEDIUM)
    if room.camera is not None:
        caps.append(room.camera.confidence)
    return min(caps, key=ladder.index)


class RenderProvider:
    """Produces room visualizations, preferring Replicate, falling back offline."""

    def __init__(self) -> None:
        self._client = None
        self.source = "mock"
        if config.HAS_REPLICATE:
            try:
                import replicate

                self._client = replicate.Client(api_token=config.REPLICATE_API_TOKEN)
                self.source = "replicate"
            except ImportError:
                log.warning(
                    "REPLICATE_API_TOKEN is set but the 'replicate' package is "
                    "not installed; falling back to schematic renders"
                )
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("Replicate unavailable, using schematic renders: %s", exc)

    @property
    def method(self) -> RenderMethod:
        return (
            RenderMethod.GENERATIVE
            if self._client is not None
            else RenderMethod.SCHEMATIC
        )

    # --- plate ------------------------------------------------------------

    async def prepare_plate(self, image_b64: str, roles: list[Role]) -> RoomPlate:
        """Detect existing furniture and erase it, yielding a clean plate.

        Only roles we intend to replace are erased. Erasing a chair the user is
        keeping would delete furniture they never asked us to touch.
        """
        if self._client is None:
            # Nothing to erase offline: the schematic draws over the photo.
            return RoomPlate(plate_b64=image_b64, detections=[], erased=False)

        try:
            detections = await self._detect(image_b64, roles)
        except Exception as exc:
            log.warning("detection failed, using the photo unerased: %s", exc)
            return RoomPlate(plate_b64=image_b64, detections=[], erased=False)

        targets = [d for d in detections if d.role not in _ERASE_EXCLUDED]
        if not targets:
            return RoomPlate(plate_b64=image_b64, detections=detections, erased=False)

        try:
            masks = await asyncio.gather(
                *(self._segment(image_b64, d) for d in targets)
            )
            for det, mask in zip(targets, masks):
                det.mask_b64 = mask
            combined = self._merge_masks(
                image_b64, [m for m in masks if m], config.MASK_DILATE_PX
            )
            plate = await self._erase(image_b64, combined)
            return RoomPlate(plate_b64=plate, detections=detections, erased=True)
        except Exception as exc:
            log.warning("erase failed, using the photo unerased: %s", exc)
            return RoomPlate(plate_b64=image_b64, detections=detections, erased=False)

    # --- per-item render --------------------------------------------------

    async def render_item(
        self,
        plate: RoomPlate,
        placement: Placement,
        item: CatalogItem,
        room: RoomAnalysis,
        projection: FloorProjection,
    ) -> RenderResult | RenderFailure:
        """Render one catalog item into the plate at its solved position."""
        started = time.perf_counter()
        base = _decode(plate.plate_b64)

        try:
            box = projection.item_box(
                placement.x_cm,
                placement.y_cm,
                placement.w_cm,
                placement.d_cm,
                item.dimensions.height_cm,
            )
        except ProjectionError as exc:
            return RenderFailure(
                item_id=item.id,
                name=item.title,
                role=item.role,
                reason="no_calibration",
                detail=str(exc),
            )

        replaced = plate.find(item.role)
        confidence = _render_confidence(placement, room)

        if self._client is None:
            image = self._schematic(base, placement, item, projection)
            method = RenderMethod.SCHEMATIC
        else:
            try:
                image_url = await asyncio.wait_for(
                    self._inpaint(base, box, placement, item, room, projection),
                    timeout=config.RENDER_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                return RenderFailure(
                    item_id=item.id,
                    name=item.title,
                    role=item.role,
                    reason="provider_error",
                    detail=f"render exceeded {config.RENDER_TIMEOUT_SECONDS:.0f}s",
                )
            except Exception as exc:
                log.warning("render failed for %s: %s", item.id, exc)
                return RenderFailure(
                    item_id=item.id,
                    name=item.title,
                    role=item.role,
                    reason="provider_error",
                    detail=str(exc),
                )
            return RenderResult(
                item_id=item.id,
                name=item.title,
                role=item.role,
                method=RenderMethod.GENERATIVE,
                image_url=image_url,
                confidence=confidence,
                replaced=replaced.label if replaced else None,
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )

        return RenderResult(
            item_id=item.id,
            name=item.title,
            role=item.role,
            method=method,
            image_url=_data_uri(image),
            confidence=confidence,
            replaced=replaced.label if replaced else None,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    # --- offline renderer -------------------------------------------------

    def _schematic(
        self,
        base,
        placement: Placement,
        item: CatalogItem,
        projection: FloorProjection,
    ):
        """Draw the item's projected footprint and volume onto the photo.

        Not photorealistic and not trying to be - it is legibly a diagram. But
        it runs the same projection the generative path does, so if furniture
        lands on a wall here it would land on a wall there too.
        """
        from PIL import Image, ImageDraw

        img = base.copy().convert("RGBA")
        W, H = img.size
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        footprint = projection.project_footprint(
            placement.x_cm, placement.y_cm, placement.w_cm, placement.d_cm
        )
        floor = [(x * W, y * H) for x, y in footprint]

        rgb = _hex_rgb(item.swatch)
        # The floor rect reads as the piece's true footprint, so it stays
        # translucent; the volume above it is what the eye reads as furniture.
        draw.polygon(floor, fill=rgb + (110,), outline=rgb + (220,))

        lift = (
            projection.height_offset(
                placement.x_cm + placement.w_cm / 2,
                placement.y_cm,
                item.dimensions.height_cm,
            )
            * H
        )
        if placement.z > 0 and lift > 1:
            top = [(x, y - lift) for x, y in floor]
            # Only the two faces a viewer could actually see from the front.
            draw.polygon([floor[0], floor[1], top[1], top[0]], fill=rgb + (165,))
            draw.polygon(top, fill=rgb + (200,), outline=rgb + (240,))
            # A tall narrow piece drawn as a solid block reads as a column, not
            # a lamp. Marking the stem makes the silhouette legible at a glance.
            if item.dimensions.height_cm > 120 and max(placement.w_cm, placement.d_cm) < 60:
                cx = sum(x for x, _ in floor) / 4
                base_y = sum(y for _, y in floor) / 4
                draw.line([(cx, base_y), (cx, base_y - lift)], fill=rgb + (255,), width=3)

        img.alpha_composite(overlay)
        return img.convert("RGB")

    # --- Replicate stages -------------------------------------------------

    async def _run(self, model: str, payload: dict):
        """One Replicate prediction, off the event loop."""
        if self._client is None:  # pragma: no cover - guarded by callers
            raise RuntimeError("no Replicate client")
        return await asyncio.to_thread(self._client.run, model, input=payload)

    async def _detect(self, image_b64: str, roles: list[Role]) -> list[Detection]:
        """Grounding DINO over the roles we might replace."""
        query = " . ".join(_DETECTION_PROMPTS[r] for r in roles if r in _DETECTION_PROMPTS)
        if not query:
            return []

        img = _decode(image_b64)
        W, H = img.size
        out = await self._run(
            config.GROUNDING_DINO_MODEL,
            {
                "image": _data_uri(img, "JPEG"),
                "query": query,
                "box_threshold": config.DETECTION_THRESHOLD,
                "text_threshold": 0.25,
            },
        )

        detections: list[Detection] = []
        for raw in (out or {}).get("detections", []):
            label = str(raw.get("label", "")).lower()
            role = _role_for_label(label)
            if role is None or role not in roles:
                continue
            box = raw.get("bbox") or raw.get("box")
            if not box or len(box) != 4:
                continue
            # Grounding DINO returns pixels; everything downstream is normalized.
            x1, y1, x2, y2 = (float(v) for v in box)
            detections.append(
                Detection(
                    role=role,
                    label=label,
                    score=float(raw.get("confidence", raw.get("score", 0.0))),
                    x1=max(0.0, min(x1 / W, 1.0)),
                    y1=max(0.0, min(y1 / H, 1.0)),
                    x2=max(0.0, min(x2 / W, 1.0)),
                    y2=max(0.0, min(y2 / H, 1.0)),
                )
            )
        return detections

    async def _segment(self, image_b64: str, det: Detection) -> str | None:
        """SAM 2, prompted with a detection box. Returns a base64 PNG mask."""
        img = _decode(image_b64)
        W, H = img.size
        try:
            out = await self._run(
                config.SAM2_MODEL,
                {
                    "image": _data_uri(img, "JPEG"),
                    "box": [det.x1 * W, det.y1 * H, det.x2 * W, det.y2 * H],
                },
            )
        except Exception as exc:
            log.warning("segmentation failed for %s: %s", det.label, exc)
            return None
        return await _fetch_b64(out)

    def _merge_masks(self, image_b64: str, masks: list[str], dilate_px: int) -> str:
        """Union the per-item masks into one, dilated.

        Dilation matters more than it looks: a segmenter cuts tightly to the
        object, leaving the contact shadow and a rim of colour-bled pixels
        behind. Inpainting without it leaves a furniture-shaped ghost.
        """
        from PIL import Image, ImageFilter

        base = _decode(image_b64)
        combined = Image.new("L", base.size, 0)
        for mask_b64 in masks:
            try:
                m = Image.open(io.BytesIO(base64.b64decode(mask_b64))).convert("L")
            except Exception:
                continue
            if m.size != base.size:
                m = m.resize(base.size, Image.NEAREST)
            combined = Image.composite(
                Image.new("L", base.size, 255), combined, m.point(lambda v: 255 if v > 127 else 0)
            )
        if dilate_px > 0:
            combined = combined.filter(ImageFilter.MaxFilter(_odd(dilate_px)))
        return base64.b64encode(_png_bytes(combined)).decode()

    async def _erase(self, image_b64: str, mask_b64: str) -> str:
        """LaMa over the merged mask, producing the empty-room plate."""
        out = await self._run(
            config.LAMA_MODEL,
            {
                "image": _data_uri(_decode(image_b64), "JPEG"),
                "mask": f"data:image/png;base64,{mask_b64}",
            },
        )
        result = await _fetch_b64(out)
        if result is None:
            raise RuntimeError("inpainting model returned no image")
        return result

    async def _inpaint(
        self,
        base,
        box: tuple[float, float, float, float],
        placement: Placement,
        item: CatalogItem,
        room: RoomAnalysis,
        projection: FloorProjection,
    ) -> str:
        """SDXL inpaint into the item's projected box.

        The mask comes from the solver's coordinates, not from the model's
        judgement, which is the whole point: composition is decided by geometry
        that has already been checked for collisions and bounds.
        """
        mask = self._placement_mask(base.size, box)
        payload = {
            "image": _data_uri(base, "JPEG"),
            "mask": f"data:image/png;base64,{mask}",
            "prompt": build_prompt(item, room),
            "negative_prompt": NEGATIVE_PROMPT,
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "strength": 0.95,
        }

        # Appearance conditioning on the real product shot. The seed catalog
        # points at a placeholder host, so this is normally absent and the
        # render falls back to text conditioning - a like-styled piece rather
        # than this exact SKU. Real imagery is what closes that gap.
        product = await _fetch_product_image(item.image_url)
        if product is not None:
            payload["ip_adapter_image"] = product
            payload["ip_adapter_scale"] = 0.7

        out = await self._run(config.INPAINT_MODEL, payload)
        url = _first_url(out)
        if url is None:
            raise RuntimeError("inpainting model returned no image")
        return url

    def _placement_mask(
        self, size: tuple[int, int], box: tuple[float, float, float, float]
    ) -> str:
        """White where the item goes, black elsewhere, feathered at the edge.

        Feathering avoids a hard rectangular seam where the generated pixels
        meet the photograph.
        """
        from PIL import Image, ImageDraw, ImageFilter

        W, H = size
        x1, y1, x2, y2 = box
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle(
            [
                max(0, int(x1 * W)),
                max(0, int(y1 * H)),
                min(W, int(x2 * W)),
                min(H, int(y2 * H)),
            ],
            fill=255,
        )
        mask = mask.filter(ImageFilter.GaussianBlur(6))
        return base64.b64encode(_png_bytes(mask)).decode()


# --- helpers ---------------------------------------------------------------


def _odd(n: int) -> int:
    """MaxFilter requires an odd kernel size."""
    n = max(3, int(n))
    return n if n % 2 else n + 1


def _png_bytes(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _hex_rgb(swatch: str) -> tuple[int, int, int]:
    s = swatch.lstrip("#")
    if len(s) != 6:
        return (139, 115, 85)
    try:
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    except ValueError:
        return (139, 115, 85)


def _role_for_label(label: str) -> Role | None:
    """Map a detector's free-text label back onto our role vocabulary."""
    text = label.lower()
    # Order matters: "coffee table" must beat a bare "table", and "tv stand"
    # must not be caught by a generic match first.
    for needles, role in [
        (("coffee table",), Role.COFFEE_TABLE),
        (("tv stand", "media console", "sideboard", "tv unit"), Role.TV_UNIT),
        (("armchair", "accent chair"), Role.ACCENT_CHAIR),
        (("floor lamp",), Role.FLOOR_LAMP),
        (("sofa", "couch", "loveseat"), Role.SOFA),
        (("rug", "carpet"), Role.RUG),
    ]:
        if any(n in text for n in needles):
            return role
    return None


def _first_url(out) -> str | None:
    """Replicate returns a URL, a list of them, or a file-like object."""
    if out is None:
        return None
    if isinstance(out, str):
        return out
    if isinstance(out, (list, tuple)):
        return _first_url(out[0]) if out else None
    url = getattr(out, "url", None)
    return url if isinstance(url, str) else None


async def _fetch_b64(out) -> str | None:
    """Download a Replicate output image and return it base64-encoded."""
    url = _first_url(out)
    if url is None:
        return None
    if url.startswith("data:"):
        return url.split(",", 1)[-1]
    try:
        import urllib.request

        def _get() -> bytes:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return resp.read()

        return base64.b64encode(await asyncio.to_thread(_get)).decode()
    except Exception as exc:
        log.warning("could not fetch %s: %s", url, exc)
        return None


async def _fetch_product_image(url: str) -> str | None:
    """Fetch a catalog product shot for appearance conditioning.

    Returns None for anything unfetchable, which the seed catalog's .invalid
    host always is. That is the intended behaviour for demo data, not an error
    worth surfacing to the user.
    """
    if not url or ".invalid" in url:
        return None
    try:
        import urllib.request

        def _get() -> bytes:
            with urllib.request.urlopen(url, timeout=15) as resp:
                return resp.read()

        raw = await asyncio.to_thread(_get)
        return f"data:image/jpeg;base64,{base64.b64encode(raw).decode()}"
    except Exception as exc:
        log.info("no product image for conditioning (%s): %s", url, exc)
        return None


# --- orchestration ---------------------------------------------------------


async def render_layout(
    provider: RenderProvider,
    image_b64: str | None,
    room: RoomAnalysis,
    layout: LayoutResult,
    items: dict[str, CatalogItem],
    only: list[str] | None = None,
):
    """Render every placed item, yielding results as they finish.

    An async generator so the caller can stream each render the moment it is
    ready. At 15-40s an item, waiting for the whole set before showing anything
    would make the feature feel broken.
    """
    placements = [p for p in layout.placements if not only or p.item_id in only]

    if not image_b64:
        for p in placements:
            yield RenderFailure(
                item_id=p.item_id,
                name=p.name,
                role=p.role,
                reason="no_photo",
                detail="a room photo is required to visualize replacements",
            )
        return

    if room.camera is None:
        for p in placements:
            yield RenderFailure(
                item_id=p.item_id,
                name=p.name,
                role=p.role,
                reason="no_calibration",
                detail="could not locate the floor plane in the photo",
            )
        return

    try:
        projection = FloorProjection.from_calibration(
            room.camera, room.width_cm, room.depth_cm
        )
    except ProjectionError as exc:
        for p in placements:
            yield RenderFailure(
                item_id=p.item_id,
                name=p.name,
                role=p.role,
                reason="no_calibration",
                detail=str(exc),
            )
        return

    roles = list({p.role for p in placements})
    plate = await provider.prepare_plate(image_b64, roles)

    # Bounded concurrency: renders overlap to hide latency, but a wide fan-out
    # of GPU calls is the quickest way to a rate limit.
    semaphore = asyncio.Semaphore(max(1, config.RENDER_CONCURRENCY))

    async def one(p: Placement):
        item = items.get(p.item_id)
        if item is None:
            return RenderFailure(
                item_id=p.item_id,
                name=p.name,
                role=p.role,
                reason="not_placed",
                detail="item is no longer in the catalog",
            )
        async with semaphore:
            return await provider.render_item(plate, p, item, room, projection)

    # Painter's order: rugs first, then back of the room to front, so a piece
    # nearer the camera occludes one behind it. Renders are dispatched
    # concurrently but yielded in this order, which costs nothing in wall-clock
    # (they overlap regardless) and lets a client composite by arrival.
    ordered = sorted(placements, key=lambda p: (p.z, -(p.y_cm + p.d_cm)))
    tasks = [asyncio.create_task(one(p)) for p in ordered]
    try:
        for task in tasks:
            yield await task
    finally:
        for task in tasks:
            task.cancel()


# --- selftest --------------------------------------------------------------


def selftest() -> None:
    """Assert the render pipeline's invariants offline."""
    from .models import CameraCalibration, Confidence, FloorQuad
    from .rag_engine import _MOCK_ROOM
    from .seed_data import SEED_ITEMS
    from .solver import LayoutSolver

    room = _MOCK_ROOM.model_copy(update={"measured": True})
    items = {i.id: i for i in SEED_ITEMS}
    wishlist = [
        next(i for i in SEED_ITEMS if i.role is r)
        for r in (Role.RUG, Role.SOFA, Role.COFFEE_TABLE, Role.TV_UNIT)
    ]
    layout = LayoutSolver(room).solve(wishlist)
    assert layout.placements, "solver placed nothing; cannot exercise renderer"

    photo = _blank_photo()
    provider = RenderProvider()

    async def _run(**kwargs):
        return [
            r
            async for r in render_layout(provider, photo, room, layout, items, **kwargs)
        ]

    # 1. Every placement produces exactly one outcome - nothing vanishes.
    results = asyncio.run(_run())
    assert len(results) == len(layout.placements), "render count != placement count"
    ok = [r for r in results if isinstance(r, RenderResult)]
    assert len(ok) == len(results), (
        f"offline renders should never fail: {[r.reason for r in results if isinstance(r, RenderFailure)]}"
    )

    # 2. Painter's order: rugs first, then back of the room forward. A client
    # compositing in arrival order must get correct occlusion.
    order = {r.item_id: n for n, r in enumerate(ok)}
    by_id = {p.item_id: p for p in layout.placements}
    for a in ok:
        for b in ok:
            pa, pb = by_id[a.item_id], by_id[b.item_id]
            if pa.z < pb.z:
                assert order[a.item_id] < order[b.item_id], "z order violated"
            elif pa.z == pb.z and (pa.y_cm + pa.d_cm) > (pb.y_cm + pb.d_cm):
                assert order[a.item_id] < order[b.item_id], (
                    f"{a.name} is nearer than {b.name} but renders first"
                )

    # 3. A render never claims more confidence than its inputs support. An
    # unmeasured room caps everything at MEDIUM however sure the solver was.
    estimated = room.model_copy(update={"measured": False})
    est_layout = LayoutSolver(estimated).solve(wishlist)
    if est_layout.placements:
        est = asyncio.run(
            _collect(render_layout(provider, photo, estimated, est_layout, items))
        )
        for r in est:
            if isinstance(r, RenderResult):
                assert r.confidence is not Confidence.HIGH, (
                    f"{r.name}: HIGH confidence from an unmeasured room"
                )

    # 4. Missing prerequisites fail explicitly rather than silently rendering
    # nothing. No photo and no calibration are different, reportable states.
    no_photo = asyncio.run(_collect(render_layout(provider, None, room, layout, items)))
    assert no_photo and all(
        isinstance(r, RenderFailure) and r.reason == "no_photo" for r in no_photo
    ), "a missing photo must be reported per item"

    blind = room.model_copy(update={"camera": None})
    no_cal = asyncio.run(_collect(render_layout(provider, photo, blind, layout, items)))
    assert no_cal and all(
        isinstance(r, RenderFailure) and r.reason == "no_calibration" for r in no_cal
    ), "a missing calibration must be reported per item"

    # 5. A degenerate quad is refused, not fitted into nonsense coordinates.
    flat = room.model_copy(
        update={
            "camera": CameraCalibration(
                quad=FloorQuad(
                    near_left=(0.5, 0.5),
                    near_right=(0.5, 0.5),
                    far_right=(0.5, 0.5),
                    far_left=(0.5, 0.5),
                ),
                horizon_y=0.4,
                source="mock",
                confidence=Confidence.LOW,
            )
        }
    )
    bad = asyncio.run(_collect(render_layout(provider, photo, flat, layout, items)))
    assert all(isinstance(r, RenderFailure) for r in bad), (
        "a degenerate floor quad must fail rather than render"
    )

    # 6. Filtering renders only what was asked for.
    target = layout.placements[0].item_id
    subset = asyncio.run(_run(only=[target]))
    assert [r.item_id for r in subset] == [target], "only= filter did not apply"

    for r in ok:
        print(
            f"    {r.role.value:<13} {r.name:<26} {r.method.value:<10} "
            f"{r.confidence.value:<6} {r.elapsed_ms:>4}ms"
        )
    print("all render invariants hold")


async def _collect(gen) -> list:
    return [r async for r in gen]


def _blank_photo() -> str:
    """A plain plate standing in for a room photo."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (1024, 768), (232, 226, 216)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


if __name__ == "__main__":
    import asyncio as _asyncio

    from .rag_engine import _MOCK_ROOM
    from .seed_data import SEED_ITEMS
    from .solver import LayoutSolver

    logging.basicConfig(level=logging.INFO)

    async def _main() -> None:
        from PIL import Image

        # Measured: an unmeasured room withholds every wall-hugging piece, so
        # the demo would render a rug and a lamp and nothing worth looking at.
        room = _MOCK_ROOM.model_copy(update={"measured": True})
        wishlist = [
            next(i for i in SEED_ITEMS if i.role is r)
            for r in (Role.RUG, Role.SOFA, Role.COFFEE_TABLE, Role.FLOOR_LAMP)
        ]
        layout = LayoutSolver(room).solve(wishlist)

        # Stand-in for a room photo: a plain plate, so the schematic is legible.
        buf = io.BytesIO()
        Image.new("RGB", (1024, 768), (232, 226, 216)).save(buf, format="JPEG")
        photo = base64.b64encode(buf.getvalue()).decode()

        provider = RenderProvider()
        print(f"renderer: {provider.source} ({provider.method.value})\n")

        results = []
        async for res in render_layout(
            provider, photo, room, layout, {i.id: i for i in SEED_ITEMS}
        ):
            results.append(res)
            if isinstance(res, RenderResult):
                print(
                    f"  {res.role.value:<13} {res.name:<26} "
                    f"{res.method.value:<10} {res.confidence.value:<6} "
                    f"{res.elapsed_ms:>4}ms"
                )
            else:
                print(f"  FAIL {res.role.value:<13} {res.name:<26} {res.reason}")

        ok = [r for r in results if isinstance(r, RenderResult)]
        if ok:
            out = "/tmp/roomcrafter_render.png"
            payload = ok[0].image_url.split(",", 1)[-1]
            with open(out, "wb") as fh:
                fh.write(base64.b64decode(payload))
            print(f"\nwrote {out}")

    _asyncio.run(_main())
