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
import urllib.parse
import base64
import io
import logging
import time
from dataclasses import dataclass, field

from . import config
from .geometry import FloorProjection, ProjectionError
from .rag_engine import _clamp01, role_for_label
from .models import (
    DimensionSource,
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
    RoomRender,
)

log = logging.getLogger(__name__)

# What to hand Grounding DINO for each role we might need to erase. The model
# takes free text, and the plain English word outperforms our enum's snake_case.
_DETECTION_PROMPTS: dict[Role, str] = {
    Role.SOFA: "sofa . couch . loveseat",
    Role.COFFEE_TABLE: "coffee table",
    Role.RUG: "rug . carpet",
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


def _render_confidence(
    placement: Placement,
    room: RoomAnalysis,
    projection: FloorProjection | None = None,
) -> Confidence:
    """A render is never more trustworthy than what it was derived from.

    Four things cap it: the placement's own confidence, whether the room was
    measured, the calibration's, and how much of the piece's floor the camera
    actually saw. Taking the weakest is the honest answer - a perfectly-placed
    sofa projected through a guessed camera is still a guess.
    """
    ladder = [Confidence.LOW, Confidence.MEDIUM, Confidence.HIGH]
    caps = [placement.confidence]
    caps.append(Confidence.HIGH if room.measured else Confidence.MEDIUM)
    if room.camera is not None:
        caps.append(room.camera.confidence)
    if projection is not None:
        # Past the photographed floor the homography is extrapolating, and
        # extrapolation degrades with distance. A piece sitting mostly outside
        # the captured floor is positioned by arithmetic, not by evidence.
        seen = projection.visible_fraction(placement.y_cm, placement.d_cm)
        caps.append(
            Confidence.HIGH
            if seen >= 0.75
            else Confidence.MEDIUM
            if seen >= 0.25
            else Confidence.LOW
        )
    return min(caps, key=ladder.index)


class RenderProvider:
    """Produces room visualizations, preferring Replicate, falling back offline."""

    def __init__(self) -> None:
        self._client = None
        self.source = "mock"
        # Composing backend, used for whole-room renders. Independent of the
        # per-item one: a deployment may have either, both, or neither.
        self.composer = GeminiComposer()
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
        if self.composer.available and self._client is None:
            self.source = self.composer.source

    @property
    def can_compose(self) -> bool:
        """Whether a whole-room render is available."""
        return self.composer.available

    @property
    def method(self) -> RenderMethod:
        """The method a per-item render would use."""
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
            kept = [m for m in masks if m]
            # Every segmentation failed, so there is nothing to erase. Calling
            # LaMa with an all-black mask spends a prediction to get the photo
            # back unchanged, and returning erased=True would tell the client
            # the room was cleared while every render still shows the old
            # furniture behind the new piece.
            if not kept:
                log.warning("no masks produced, using the photo unerased")
                return RoomPlate(
                    plate_b64=image_b64, detections=detections, erased=False
                )
            combined = self._merge_masks(image_b64, kept, config.MASK_DILATE_PX)
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
        # Base64 + JPEG decode of a full-resolution photo is tens of
        # milliseconds of pure CPU. On the event loop it blocks every other
        # render and stalls the SSE heartbeats that keep the connection open.
        base = await asyncio.to_thread(_decode, plate.plate_b64)

        try:
            box = projection.item_box(
                placement.x_cm,
                placement.y_cm,
                placement.w_cm,
                placement.d_cm,
                item.dimensions.height_cm,
            )
        except ProjectionError as exc:
            # The calibration is fine; this piece simply sits on floor the
            # camera never saw. Saying so lets the client suggest a wider shot
            # instead of implying the photo could not be read at all.
            return RenderFailure(
                item_id=item.id,
                name=item.title,
                role=item.role,
                reason="out_of_frame",
                detail=str(exc),
            )

        replaced = plate.find(item.role)
        confidence = _render_confidence(placement, room, projection)

        if self._client is None:
            try:
                # The whole offline workload: RGBA copy, overlay allocation,
                # polygon fills and an alpha composite over a 1024px image.
                # Threaded so RENDER_CONCURRENCY actually buys concurrency
                # instead of serializing on the loop.
                image = await asyncio.to_thread(
                    self._schematic, base, placement, item, projection
                )
            except ProjectionError as exc:
                return RenderFailure(
                    item_id=item.id,
                    name=item.title,
                    role=item.role,
                    reason="out_of_frame",
                    detail=str(exc),
                )
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
            role = role_for_label(label)
            if role is None or role not in roles:
                continue
            box = raw.get("bbox") or raw.get("box")
            if not box or len(box) != 4:
                continue
            # Grounding DINO returns pixels; everything downstream is
            # normalized. Parsed defensively for the same reasons as
            # rag_engine._parse_detections: a null or out-of-range confidence
            # would fail Detection's [0,1] validation and abort the whole
            # loop, and prepare_plate swallows that - so one bad box would
            # silently skip erasing for the entire photo.
            try:
                x1, y1, x2, y2 = (float(v) for v in box)
            except (TypeError, ValueError):
                continue
            x1, x2 = sorted((_clamp01(x1 / W), _clamp01(x2 / W)))
            y1, y2 = sorted((_clamp01(y1 / H), _clamp01(y2 / H)))
            # A zero-area box segments to nothing and would waste a SAM 2 call.
            if x2 - x1 <= 0.0 or y2 - y1 <= 0.0:
                continue
            try:
                score = _clamp01(raw.get("confidence", raw.get("score", 0.0)) or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            detections.append(
                Detection(
                    role=role,
                    label=label,
                    score=score,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
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


def _fetchable(url: str) -> bool:
    """Whether this URL may be fetched for appearance conditioning.

    Catalog image_urls are vendor CDN links, so fetching one is an outbound
    request the server makes on behalf of a user request. `image_url` is data,
    not code - it arrives from a scrape and could arrive from a user-supplied
    catalog later - so the host is checked against an allowlist rather than
    trusted. Without that, anything that can write a catalog entry can make
    this server fetch an arbitrary URL, including one on a private network.
    """
    if not url:
        return False
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").lower()
    return any(
        host == allowed or host.endswith("." + allowed)
        for allowed in config.IMAGE_FETCH_ALLOWED_HOSTS
    )


async def _fetch_product_image(url: str) -> str | None:
    """Fetch a catalog product shot for appearance conditioning.

    Returns None for anything unfetchable rather than raising: a missing
    reference degrades a render, it does not break one, and the caller already
    reports the piece as omitted.
    """
    if not _fetchable(url):
        log.info("product image host not allowed, skipping: %s", url[:100])
        return None
    try:
        import urllib.request

        def _get() -> bytes:
            # An explicit UA matters: a default "Python-urllib/3.x" is the
            # first thing a CDN blocks, and a silent block would degrade every
            # render to a text-conditioned guess with no obvious cause.
            req = urllib.request.Request(
                url, headers={"User-Agent": config.IMAGE_FETCH_USER_AGENT}
            )
            with urllib.request.urlopen(
                req, timeout=config.IMAGE_FETCH_TIMEOUT_SECONDS
            ) as resp:
                # Read one byte past the cap so an oversized body is detected
                # rather than silently truncated into a corrupt image.
                data = resp.read(config.IMAGE_FETCH_MAX_BYTES + 1)
                if len(data) > config.IMAGE_FETCH_MAX_BYTES:
                    raise ValueError(
                        f"product image exceeds {config.IMAGE_FETCH_MAX_BYTES} bytes"
                    )
                return data

        raw = await asyncio.to_thread(_get)
        # The declared type is cosmetic - every consumer decodes the base64 and
        # lets PIL sniff the real format - but it should not claim JPEG for a
        # PNG. Sniff the magic bytes instead of trusting the URL's extension.
        mime = "image/png" if raw[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
        return f"data:{mime};base64,{base64.b64encode(raw).decode()}"
    except Exception as exc:
        log.info("no product image for conditioning (%s): %s", url[:100], exc)
        return None


# --- composing renderer (Gemini) -------------------------------------------

# Roles the model is shown a reference for first when the budget is tight. The
# pieces that define a room read as wrong if they are missing; a lamp does not.
_REFERENCE_PRIORITY: list[Role] = [
    Role.SOFA,
    Role.RUG,
    Role.COFFEE_TABLE,
    Role.ACCENT_CHAIR,
    Role.FLOOR_LAMP,
]


def describe_position(placement: Placement, room: RoomAnalysis) -> str:
    """The solver's coordinates as spatial language a model can act on.

    A composing model edits the whole image and takes no mask, so the solver's
    geometry has to travel as description. Fractions of the room read better
    than centimetres: the model has no metre stick, but it can see that
    something belongs two-thirds of the way back and slightly left of centre.
    """
    cx = (placement.x_cm + placement.w_cm / 2) / max(room.width_cm, 1)
    cy = (placement.y_cm + placement.d_cm / 2) / max(room.depth_cm, 1)

    if cx < 0.33:
        across = "on the left side"
    elif cx > 0.67:
        across = "on the right side"
    else:
        across = "centred left-to-right"

    if cy < 0.3:
        along = "against the far wall"
    elif cy > 0.7:
        along = "in the foreground, nearest the camera"
    else:
        along = "in the middle of the room"

    width_share = placement.w_cm / max(room.width_cm, 1)
    return (
        f"{along}, {across}, spanning roughly {width_share * 100:.0f}% of the "
        f"room's width ({placement.w_cm:.0f}cm wide)"
    )


def build_composition_prompt(
    room: RoomAnalysis,
    placements: list[Placement],
    items: dict[str, CatalogItem],
    replaced: list[str],
) -> str:
    """The instruction accompanying the room photo and the product references.

    Numbered to match the order the reference images are appended, because the
    model has no other way to tell which photo is the sofa.
    """
    lines = [
        "You are compositing a photorealistic interior visualization.",
        "",
        "The FIRST image is a photograph of a real room. Each image after it is "
        "a product photograph of a piece of furniture.",
        "",
        "Produce a single photorealistic image of that same room, from the same "
        "camera position and with the same walls, windows, flooring and "
        "daylight, but furnished with the products shown.",
        "",
    ]

    if replaced:
        lines += [
            "First remove the existing furniture: "
            + ", ".join(sorted(set(replaced)))
            + ". Reconstruct the floor and walls behind it convincingly.",
            "",
        ]

    lines.append("Then place each product:")
    for n, p in enumerate(placements, start=2):
        item = items[p.item_id]
        relation = _relate(p, placements, items)
        lines.append(
            f"  {n}. {item.title} ({item.role.value.replace('_', ' ')}) - "
            f"{describe_position(p, room)}"
            + (f"; {relation}" if relation else "")
            + "."
        )

    lines += [
        "",
        "Requirements:",
        "- Match each piece to its reference photograph: same shape, same "
        "materials, same colour, same proportions. These are real products a "
        "customer will buy, so silhouette and finish must be recognisable.",
        "- Respect the stated positions and relative sizes. Furniture rests on "
        "the floor with contact shadows; nothing floats or intersects.",
        f"- Keep the room's own character: {room.wall_color} walls, "
        f"{room.flooring} flooring, {room.lighting}.",
        "- Relight each product to match the room's existing light direction "
        "and warmth rather than keeping its studio lighting.",
        "- Photographic realism. No text, labels, watermarks or illustration.",
    ]
    return "\n".join(lines)


def _relate(
    placement: Placement,
    placements: list[Placement],
    items: dict[str, CatalogItem],
) -> str:
    """How this piece sits relative to the others, in one phrase.

    Absolute positions alone lose the relationships that make a layout read as
    designed rather than scattered: a coffee table belongs in front of its
    sofa, and everything belongs on top of the rug. The solver already
    guarantees these, so stating them costs nothing and is what stops the model
    arranging the same pieces into an unrelated room.
    """
    by_role = {p.role: p for p in placements}
    sofa = by_role.get(Role.SOFA)
    rug = by_role.get(Role.RUG)

    parts: list[str] = []
    if placement.role is Role.COFFEE_TABLE and sofa:
        parts.append("directly in front of the sofa, within easy reach of it")
    elif placement.role is Role.ACCENT_CHAIR and sofa:
        parts.append("angled towards the sofa, beside it")
    elif placement.role is Role.FLOOR_LAMP and sofa:
        parts.append("standing beside the sofa")

    # The rug is the one piece everything else rests on, which a flat list of
    # positions would otherwise leave the model to guess at.
    if rug is not None and placement.role not in (Role.RUG, Role.FLOOR_LAMP):
        if _overlaps_rug(placement, rug):
            parts.append("with its front legs resting on the rug")
    if placement.role is Role.RUG:
        parts.append("lying flat on the floor beneath the other furniture")

    return ", ".join(parts)


def _overlaps_rug(placement: Placement, rug: Placement) -> bool:
    """Whether a piece's footprint sits over the rug's."""
    return (
        placement.x_cm < rug.x_cm + rug.w_cm
        and rug.x_cm < placement.x_cm + placement.w_cm
        and placement.y_cm < rug.y_cm + rug.d_cm
        and rug.y_cm < placement.y_cm + placement.d_cm
    )


class GeminiComposer:
    """Renders a whole design in one multi-image call.

    Nano Banana composes up to 14 object references, which is what makes this
    worth doing: the recommended furniture is rendered from its actual catalog
    photograph rather than from a text description of it. That is the product
    fidelity the per-item inpaint path could only approximate.
    """

    def __init__(self) -> None:
        self._client = None
        self.source = "mock"
        if config.HAS_GEMINI:
            try:
                from google import genai

                self._client = genai.Client(api_key=config.GEMINI_API_KEY)
                self.source = "gemini"
            except ImportError:
                log.warning(
                    "GEMINI_API_KEY is set but the 'google-genai' package is "
                    "not installed; falling back to schematic renders"
                )
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("Gemini unavailable, using schematic renders: %s", exc)

    @property
    def available(self) -> bool:
        return self._client is not None

    async def compose(
        self,
        image_b64: str,
        room: RoomAnalysis,
        placements: list[Placement],
        items: dict[str, CatalogItem],
        replaced: list[str],
    ) -> RoomRender:
        """One call: room photo + product photos + the solver's layout."""
        if self._client is None:  # pragma: no cover - guarded by callers
            raise RuntimeError("no Gemini client")

        started = time.perf_counter()

        # Priority order, then the reference cap. Fewer, well-chosen references
        # compose better than the maximum the model will accept.
        ordered = sorted(
            placements,
            key=lambda p: _REFERENCE_PRIORITY.index(p.role)
            if p.role in _REFERENCE_PRIORITY
            else len(_REFERENCE_PRIORITY),
        )
        budget = max(1, config.GEMINI_MAX_REFERENCES)
        omitted = [p.item_id for p in ordered[budget:]]
        ordered = ordered[:budget]

        room_img = _decode(image_b64)
        # References are vendor CDN fetches, so they go out concurrently rather
        # than one after another. Order is preserved by gathering in `ordered`
        # order, which matters: the prompt refers to the references by
        # position, so shuffling them would mislabel every piece.
        wanted = [(p, items.get(p.item_id)) for p in ordered]
        fetched = await asyncio.gather(
            *[
                _fetch_product_pil(item.image_url) if item is not None else _none()
                for _p, item in wanted
            ]
        )

        parts: list = [room_img]
        used: list[Placement] = []
        for (p, item), product in zip(wanted, fetched):
            if item is None or product is None:
                # No reference means the model would invent this piece from its
                # name alone, which is exactly the fidelity gap this path
                # exists to close. Leave it out and say so.
                omitted.append(p.item_id)
                continue
            parts.append(product)
            used.append(p)

        if not used:
            raise RuntimeError(
                "no product images could be fetched; nothing to compose from"
            )

        prompt = build_composition_prompt(room, used, items, replaced)
        image = await asyncio.wait_for(
            asyncio.to_thread(self._generate, [prompt] + parts),
            timeout=config.RENDER_TIMEOUT_SECONDS,
        )

        return RoomRender(
            image_url=_data_uri(image),
            method=RenderMethod.COMPOSED,
            item_ids=[p.item_id for p in used],
            omitted=omitted,
            replaced=replaced,
            confidence=_compose_confidence(used, room),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    def _generate(self, contents: list):
        """Blocking SDK call, run off the event loop."""
        from google.genai import types

        resp = self._client.models.generate_content(
            model=config.GEMINI_IMAGE_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                # Composition should follow the references, not improvise on
                # them. Low temperature keeps the products recognisable.
                temperature=0.4,
            ),
        )
        return _extract_image(resp)


def _extract_image(resp):
    """Pull the generated image out of a GenerateContentResponse.

    The SDK returns candidates whose parts may mix text and inline image data,
    so this walks for the first image part rather than assuming position.
    """
    from PIL import Image

    for candidate in getattr(resp, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            inline = getattr(part, "inline_data", None)
            data = getattr(inline, "data", None) if inline else None
            if data:
                raw = base64.b64decode(data) if isinstance(data, str) else data
                return Image.open(io.BytesIO(raw)).convert("RGB")

    # A refusal or a safety block comes back as text, not as an error, so
    # surface whatever the model said instead of a bare "no image".
    text = (getattr(resp, "text", None) or "").strip()
    raise RuntimeError(f"model returned no image{': ' + text[:200] if text else ''}")


def _compose_confidence(placements: list[Placement], room: RoomAnalysis) -> Confidence:
    """How much to trust a composed render.

    Capped one notch below the per-item path by construction: a composing model
    takes no mask, so the solver's geometry reaches it as description rather
    than as a constraint. Positions are a strong suggestion, not a guarantee.
    """
    ladder = [Confidence.LOW, Confidence.MEDIUM, Confidence.HIGH]
    caps = [Confidence.MEDIUM]  # description-driven placement never rates HIGH
    caps += [p.confidence for p in placements]
    if not room.measured:
        caps.append(Confidence.LOW)
    return min(caps, key=ladder.index)


async def _none():
    """Awaitable None, so a missing item can sit in an asyncio.gather list."""
    return None


async def _fetch_product_pil(url: str):
    """A catalog product shot as a PIL image, or None if unfetchable."""
    data_uri = await _fetch_product_image(url)
    if data_uri is None:
        return None
    try:
        return _decode(data_uri.split(",", 1)[-1])
    except Exception as exc:
        log.info("could not decode product image %s: %s", url, exc)
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

    async def _render_one(p: Placement):
        item = items.get(p.item_id)
        if item is None:
            return RenderFailure(
                item_id=p.item_id,
                name=p.name,
                role=p.role,
                reason="not_placed",
                detail="item is no longer in the catalog",
            )
        return await provider.render_item(plate, p, item, room, projection)

    async def one(p: Placement):
        # The semaphore wraps the whole call, including render_item's own
        # decode. Acquiring it further in would let every task run its setup
        # immediately, which is most of the work on the offline path.
        async with semaphore:
            return await _render_one(p)

    # Painter's order: rugs first, then back of the room to front, so a piece
    # nearer the camera occludes one behind it. Renders are dispatched
    # concurrently but yielded in this order, which costs nothing in wall-clock
    # (they overlap regardless) and lets a client composite by arrival.
    #
    # Ascending on the back edge (y_cm + d_cm): the piece deepest in the room
    # is emitted first and a nearer one paints over it. Negating this reverses
    # the occlusion and draws the far sofa on top of the near coffee table.
    ordered = sorted(placements, key=lambda p: (p.z, p.y_cm + p.d_cm))
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

    room = _MOCK_ROOM.model_copy(update={"dimension_source": DimensionSource.MEASURED})
    items = {i.id: i for i in SEED_ITEMS}
    wishlist = [
        next(i for i in SEED_ITEMS if i.role is r)
        for r in (Role.RUG, Role.SOFA, Role.COFFEE_TABLE, Role.ACCENT_CHAIR)
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
            elif pa.z == pb.z and (pa.y_cm + pa.d_cm) < (pb.y_cm + pb.d_cm):
                # a sits further back, so it must render first and be painted
                # over by b rather than the other way round.
                assert order[a.item_id] < order[b.item_id], (
                    f"{a.name} is further back than {b.name} but renders second"
                )

    # 3. A render never claims more confidence than its inputs support. An
    # unmeasured room caps everything at MEDIUM however sure the solver was.
    estimated = room.model_copy(update={"dimension_source": DimensionSource.ESTIMATED})
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

    # 7. The composition prompt is the whole geometry channel for the composing
    # backend - it takes no mask - so it must actually carry the layout.
    prompt = build_composition_prompt(
        room, layout.placements, items, ["old green sofa"]
    )
    assert "old green sofa" in prompt, "removal instruction lost"
    for n, p in enumerate(layout.placements, start=2):
        assert f"  {n}. {items[p.item_id].title}" in prompt, (
            f"{p.name} missing or misnumbered in the prompt"
        )
    # Numbering must match reference order: the model has no other way to tell
    # which product photo is the sofa.
    positions = [prompt.index(f"  {n}. ") for n in range(2, len(layout.placements) + 2)]
    assert positions == sorted(positions), "prompt numbering is out of order"
    table = next((p for p in layout.placements if p.role is Role.COFFEE_TABLE), None)
    if table and any(p.role is Role.SOFA for p in layout.placements):
        assert "in front of the sofa" in prompt, "table lost its relation to seating"

    # 8. Composing is refused cleanly when unavailable, rather than half-running.
    if not provider.can_compose:
        failure = asyncio.run(render_room(provider, photo, room, layout, items))
        assert isinstance(failure, RenderFailure), "compose should fail with no backend"
        assert failure.reason == "provider_error", failure.reason
        no_photo_room = asyncio.run(render_room(provider, None, room, layout, items))
        assert (
            isinstance(no_photo_room, RenderFailure)
            and no_photo_room.reason == "no_photo"
        ), "a missing photo must be reported before the backend check"

    for r in ok:
        print(
            f"    {r.role.value:<13} {r.name:<26} {r.method.value:<10} "
            f"{r.confidence.value:<6} {r.elapsed_ms:>4}ms"
        )
    print(f"    composition prompt: {len(prompt)} chars, "
          f"{len(layout.placements)} products referenced")
    print("all render invariants hold")


async def _collect(gen) -> list:
    return [r async for r in gen]


def _blank_photo() -> str:
    """A plain plate standing in for a room photo."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (1024, 768), (232, 226, 216)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


async def render_room(
    provider: RenderProvider,
    image_b64: str | None,
    room: RoomAnalysis,
    layout: LayoutResult,
    items: dict[str, CatalogItem],
    only: list[str] | None = None,
) -> RoomRender | RenderFailure:
    """Compose the whole design into the room photo in one call.

    The counterpart to render_layout: that yields one image per item, this
    returns one image of the finished room. Preferred when a composing backend
    is configured, because every piece is rendered from its real product
    photograph and the lighting is solved once for the whole scene.
    """
    placements = [p for p in layout.placements if not only or p.item_id in only]

    if not placements:
        return RenderFailure(
            item_id="",
            name="room",
            role=Role.SOFA,
            reason="not_placed",
            detail="no items were placed to render",
        )

    if not image_b64:
        return RenderFailure(
            item_id="",
            name="room",
            role=placements[0].role,
            reason="no_photo",
            detail="a room photo is required to visualize replacements",
        )

    if not provider.can_compose:
        return RenderFailure(
            item_id="",
            name="room",
            role=placements[0].role,
            reason="provider_error",
            detail="no composing renderer is configured",
        )

    # Name the existing furniture so the prompt can ask for its removal.
    # These come from the room analysis, which already looked at this photo -
    # re-detecting here would repeat that work, and on a deployment with no
    # Replicate token it would find nothing at all and silently drop the
    # removal hints. Best-effort: with no detections we simply do not mention
    # it, and the model is told to furnish the room rather than clear it.
    #
    # Only `replaceable` - the detections mapping onto a role we sell. The rest
    # are the user's bookshelf, bed or TV unit: nothing here can replace them,
    # so asking the model to erase them would delete possessions and put
    # nothing back. This matches the per-item path, which filters the same set
    # through _ERASE_EXCLUDED.
    replaced = [d.label for d in room.replaceable]

    try:
        return await provider.composer.compose(
            image_b64, room, placements, items, replaced
        )
    except asyncio.TimeoutError:
        return RenderFailure(
            item_id="",
            name="room",
            role=placements[0].role,
            reason="provider_error",
            detail=f"render exceeded {config.RENDER_TIMEOUT_SECONDS:.0f}s",
        )
    except Exception as exc:
        log.warning("room composition failed: %s", exc)
        return RenderFailure(
            item_id="",
            name="room",
            role=placements[0].role,
            reason="provider_error",
            detail=str(exc),
        )


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
        room = _MOCK_ROOM.model_copy(update={"dimension_source": DimensionSource.MEASURED})
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
            out = "/tmp/roomhack_render.png"
            payload = ok[0].image_url.split(",", 1)[-1]
            with open(out, "wb") as fh:
                fh.write(base64.b64decode(payload))
            print(f"\nwrote {out}")

    _asyncio.run(_main())
