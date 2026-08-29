"""Vector catalog and the provider abstraction behind the offline guarantee.

Two provider pairs implement the same interfaces:

  EmbeddingProvider  OpenAI text-embedding-3-small  |  deterministic hash embedder
  VisionProvider     gpt-4o multimodal              |  canned room analysis

Both embedders emit EMBEDDING_DIM values, so the Qdrant collection schema does
not depend on which one is active. Both vision providers return a RoomAnalysis.
The app therefore runs end to end with no API key.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import math
import re

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

from . import config
from .models import (
    CameraCalibration,
    CatalogItem,
    Confidence,
    FloorQuad,
    RoomAnalysis,
    Role,
    Wall,
)
from .seed_data import SEED_ITEMS, validate_seed

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


# --- Embeddings ------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def hash_embed(text: str) -> list[float]:
    """Deterministic offline embedding: hashed bag of words, L2-normalized.

    Not semantically comparable to a real model — "sofa" and "couch" land in
    unrelated buckets — but it is stable, dependency-free, and good enough for
    lexical matching over a 38-item catalog. Retrieval also applies payload
    filters (role, price, dimensions), which carry most of the selection work,
    so the vector only has to break ties sensibly.
    """
    vec = [0.0] * config.EMBEDDING_DIM
    tokens = _tokenize(text)
    for token in tokens:
        # Two independent buckets per token reduces collision damage.
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        h = int.from_bytes(digest, "big")
        primary = h % config.EMBEDDING_DIM
        secondary = (h >> 20) % config.EMBEDDING_DIM
        vec[primary] += 1.0
        vec[secondary] += 0.5

    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        # An empty or punctuation-only query still needs a valid unit vector.
        vec[0] = 1.0
        return vec
    return [v / norm for v in vec]


class EmbeddingProvider:
    """Embeds text, preferring OpenAI and falling back to the hash embedder."""

    def __init__(self) -> None:
        self._client = None
        self.source = "mock"
        if config.HAS_OPENAI:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=config.OPENAI_API_KEY)
                self.source = "openai"
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("OpenAI embeddings unavailable, using offline: %s", exc)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._client is not None:
            try:
                resp = self._client.embeddings.create(
                    model=config.EMBEDDING_MODEL, input=texts
                )
                return [d.embedding for d in resp.data]
            except Exception as exc:
                # A mid-demo API failure degrades to offline rather than 500ing.
                log.warning("embedding call failed, falling back offline: %s", exc)
        return [hash_embed(t) for t in texts]


# --- Vision ----------------------------------------------------------------

_VISION_PROMPT = """You are an interior architect analysing a room photo.
Estimate the room's floor dimensions and describe its finishes.

Respond with ONLY a JSON object, no prose or code fences:
{
  "width_cm": <number, 200-800>,
  "depth_cm": <number, 200-800>,
  "focal_wall": "north"|"south"|"east"|"west",
  "wall_color": "<short description>",
  "flooring": "<short description>",
  "lighting": "<short description>",
  "notes": "<one sentence on architectural anchors: windows, doors, focal point>",
  "floor_quad": {
    "near_left":  [<x>, <y>],
    "near_right": [<x>, <y>],
    "far_right":  [<x>, <y>],
    "far_left":   [<x>, <y>]
  },
  "horizon_y": <number 0-1>,
  "floor_near_depth_cm": <number>,
  "floor_far_depth_cm": <number>
}

focal_wall is the wall a sofa should face or sit against — typically the one
with the fireplace, television, or main window.

floor_quad traces the visible floor as a quadrilateral, in NORMALIZED image
coordinates where [0,0] is the top-left of the photo and [1,1] the
bottom-right. "near" is the edge closest to the camera (lower in frame), "far"
is the edge at the back wall. Follow the floor's actual perspective: in a
typical photo the near edge is wider than the far edge. If the floor is mostly
hidden or you cannot trace it, omit floor_quad entirely rather than guessing.

horizon_y is the eye-level horizon as a fraction of image height.

floor_near_depth_cm and floor_far_depth_cm say WHICH PART of the room's depth
the quad covers, measured from the back wall (0) toward the camera. A photo
almost never shows the whole floor - the photographer is standing on the part
that is missing - so these are usually not 0 and depth_cm.

Example: in a 400cm-deep room shot from the doorway, the visible floor might
start at the back wall and stop 90cm short of the camera. Then
floor_far_depth_cm is 0 and floor_near_depth_cm is 310.

Getting this wrong pushes furniture out of the bottom of the picture, so
estimate it rather than defaulting to the full depth."""

# Used when there is no key, no image, or a malformed model response. A
# mid-size rectangular living room: large enough to place a full set.
_MOCK_ROOM = RoomAnalysis(
    width_cm=420,
    depth_cm=330,
    focal_wall=Wall.SOUTH,
    wall_color="warm white",
    flooring="light oak",
    lighting="bright, large west-facing window",
    notes="Rectangular living room with a window on the west wall and the "
    "main focal point on the south wall.",
    source="mock",
    # A plausible eye-level camera looking down the room: the near floor edge
    # spans most of the frame, the far edge narrows with perspective. Lets the
    # schematic renderer exercise the real projection maths offline.
    camera=CameraCalibration(
        quad=FloorQuad(
            near_left=(0.02, 0.98),
            near_right=(0.98, 0.98),
            far_right=(0.72, 0.44),
            far_left=(0.28, 0.44),
        ),
        horizon_y=0.38,
        # The mock room is 330cm deep and this camera stands ~80cm inside it,
        # so the near 80cm of floor is not in frame. Matching a real photo here
        # means the offline schematic exercises the same clipping the
        # generative path will.
        near_depth_cm=250.0,
        far_depth_cm=0.0,
        source="mock",
        confidence=Confidence.MEDIUM,
    ),
)


class VisionProvider:
    """Turns an optional room photo into a RoomAnalysis."""

    def __init__(self) -> None:
        self._client = None
        self.source = "mock"
        if config.HAS_OPENAI:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=config.OPENAI_API_KEY)
                self.source = "openai"
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("OpenAI vision unavailable, using offline: %s", exc)

    def analyze(self, image_b64: str | None) -> RoomAnalysis:
        if self._client is None or not image_b64:
            return _MOCK_ROOM.model_copy()
        try:
            resp = self._client.chat.completions.create(
                model=config.VISION_MODEL,
                max_tokens=400,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _VISION_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}",
                                    "detail": "low",
                                },
                            },
                        ],
                    }
                ],
            )
            raw = (resp.choices[0].message.content or "").strip()
            return self._parse(raw)
        except Exception as exc:
            log.warning("vision call failed, using offline analysis: %s", exc)
            return _MOCK_ROOM.model_copy()

    def _parse(self, raw: str) -> RoomAnalysis:
        """Parse the model's JSON, tolerating code fences and stray prose."""
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
        # Models occasionally wrap JSON in a sentence; take the outermost object.
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            log.warning("vision returned no JSON object; using offline analysis")
            return _MOCK_ROOM.model_copy()
        try:
            data = json.loads(text[start : end + 1])
            room = RoomAnalysis(**{**data, "source": "openai"})
        except Exception as exc:
            log.warning("vision JSON invalid (%s); using offline analysis", exc)
            return _MOCK_ROOM.model_copy()

        # Clamp implausible estimates rather than trusting them into the solver,
        # where a 30cm or 5000cm room would produce nonsense.
        room.width_cm = min(max(room.width_cm, 150.0), 1200.0)
        room.depth_cm = min(max(room.depth_cm, 150.0), 1200.0)

        # floor_quad and horizon_y are not RoomAnalysis fields, so pydantic
        # drops them; lift them into a calibration by hand. A bad quad is worse
        # than none - it would project furniture onto a wall - so anything that
        # fails validation leaves camera as None and rendering declines.
        room.camera = _parse_camera(data)
        return room


def _positive(value) -> float | None:
    """A positive float from model output, or None for anything else."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if num > 0 else None


def _parse_camera(data: dict) -> CameraCalibration | None:
    """Build a CameraCalibration from the vision payload, or None."""
    quad_raw = data.get("floor_quad")
    if not isinstance(quad_raw, dict):
        return None
    try:
        corners = {
            key: (float(quad_raw[key][0]), float(quad_raw[key][1]))
            for key in ("near_left", "near_right", "far_right", "far_left")
        }
    except (KeyError, TypeError, ValueError, IndexError):
        log.warning("floor_quad malformed; rendering will be unavailable")
        return None

    # Reject anything outside the frame or degenerate. A quad with a collapsed
    # edge yields a singular homography, which would raise deep inside the
    # renderer instead of here where the cause is obvious.
    if any(not (-0.2 <= v <= 1.2) for pt in corners.values() for v in pt):
        log.warning("floor_quad outside image bounds; ignoring")
        return None
    near_w = abs(corners["near_right"][0] - corners["near_left"][0])
    far_w = abs(corners["far_right"][0] - corners["far_left"][0])
    if near_w < 0.05 or far_w < 0.02:
        log.warning("floor_quad edges degenerate; ignoring")
        return None

    horizon = data.get("horizon_y", 0.35)
    try:
        horizon = min(max(float(horizon), 0.0), 1.0)
    except (TypeError, ValueError):
        horizon = 0.35

    # An unreported near depth stays None, which means "the quad reaches the
    # near wall". That is the optimistic reading, so it is only correct when
    # the model actually declined to answer - which the prompt discourages.
    near_depth = _positive(data.get("floor_near_depth_cm"))
    far_depth = _positive(data.get("floor_far_depth_cm")) or 0.0
    if near_depth is not None and near_depth <= far_depth:
        log.warning("floor depth span inverted (%s <= %s); ignoring", near_depth, far_depth)
        near_depth, far_depth = None, 0.0

    return CameraCalibration(
        quad=FloorQuad(**corners),
        horizon_y=horizon,
        near_depth_cm=near_depth,
        far_depth_cm=far_depth,
        source="openai",
        # Derived from an estimate, so it is an estimate. Only a measured room
        # promotes a render past MEDIUM.
        confidence=Confidence.MEDIUM,
    )


# --- Image preprocessing ---------------------------------------------------


def prepare_image(raw: bytes) -> str | None:
    """Downscale and re-encode an upload to base64 JPEG.

    Room analysis does not need full camera resolution, and base64 inflates
    payloads ~33%, so shrinking here cuts both latency and token cost.
    Returns None if the bytes are not a decodable image.
    """
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(raw))
        img = img.convert("RGB")
        longest = max(img.size)
        if longest > config.MAX_IMAGE_EDGE_PX:
            scale = config.MAX_IMAGE_EDGE_PX / longest
            new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
            img = img.resize(new_size, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=82)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as exc:
        log.warning("could not decode uploaded image: %s", exc)
        return None


# --- Catalog index ---------------------------------------------------------


class CatalogIndex:
    """Qdrant-backed catalog. Owns seeding and filtered vector search."""

    def __init__(self, embedder: EmbeddingProvider | None = None) -> None:
        self.embedder = embedder or EmbeddingProvider()
        self.client = QdrantClient(location=config.QDRANT_LOCATION)
        self._by_id: dict[str, CatalogItem] = {}
        self._seeded = False

    def seed(self) -> int:
        """Build the collection. Idempotent: safe to call on every reload."""
        validate_seed()

        # recreate-by-delete keeps re-seeding clean without relying on upsert
        # semantics for a collection whose vector width may have changed.
        if self.client.collection_exists(config.COLLECTION_NAME):
            self.client.delete_collection(config.COLLECTION_NAME)
        self.client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=config.EMBEDDING_DIM, distance=Distance.COSINE
            ),
        )

        vectors = self.embedder.embed([i.embed_text() for i in SEED_ITEMS])
        points = [
            PointStruct(
                # Qdrant needs an int or UUID id; the human id lives in payload.
                id=idx,
                vector=vector,
                payload={
                    "item_id": item.id,
                    "role": item.role.value,
                    "price_cents": item.price_cents,
                    "width_cm": item.dimensions.width_cm,
                    "depth_cm": item.dimensions.depth_cm,
                    "in_stock": item.in_stock,
                    "merchant": item.merchant,
                    "style_tags": item.style_tags,
                },
            )
            for idx, (item, vector) in enumerate(zip(SEED_ITEMS, vectors))
        ]
        self.client.upsert(collection_name=config.COLLECTION_NAME, points=points)

        self._by_id = {i.id: i for i in SEED_ITEMS}
        self._seeded = True
        log.info(
            "seeded %d items (embeddings: %s)", len(points), self.embedder.source
        )
        return len(points)

    def get(self, item_id: str) -> CatalogItem | None:
        return self._by_id.get(item_id)

    def search(
        self,
        query: str,
        role: Role | None = None,
        max_price_cents: int | None = None,
        max_width_cm: float | None = None,
        max_depth_cm: float | None = None,
        limit: int = 5,
    ) -> list[CatalogItem]:
        """Vector search narrowed by payload filters.

        Dimension filters matter as much as price: handing the solver a 226cm
        sofa for a 250cm room guarantees a skip, so oversized pieces are
        excluded before they ever reach placement.
        """
        if not self._seeded:
            raise RuntimeError("CatalogIndex.seed() must run before search()")

        must: list[FieldCondition] = [
            FieldCondition(key="in_stock", match=MatchValue(value=True))
        ]
        if role is not None:
            must.append(
                FieldCondition(key="role", match=MatchValue(value=role.value))
            )
        if max_price_cents is not None:
            must.append(
                FieldCondition(key="price_cents", range=Range(lte=max_price_cents))
            )
        if max_width_cm is not None:
            must.append(FieldCondition(key="width_cm", range=Range(lte=max_width_cm)))
        if max_depth_cm is not None:
            must.append(FieldCondition(key="depth_cm", range=Range(lte=max_depth_cm)))

        vector = self.embedder.embed([query])[0]
        # NOTE: client.search() was removed in qdrant-client 1.x; query_points
        # is the supported call and returns a wrapper whose .points holds hits.
        response = self.client.query_points(
            collection_name=config.COLLECTION_NAME,
            query=vector,
            query_filter=Filter(must=must),
            limit=limit,
            with_payload=True,
        )

        results: list[CatalogItem] = []
        for point in response.points:
            item_id = (point.payload or {}).get("item_id")
            item = self._by_id.get(item_id) if item_id else None
            if item is not None:
                results.append(item)
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    index = CatalogIndex()
    index.seed()
    print(f"embedding source: {index.embedder.source}\n")
    for role in (Role.SOFA, Role.RUG, Role.FLOOR_LAMP):
        hits = index.search(
            "japandi minimalist natural wood", role=role, max_price_cents=130_000, limit=3
        )
        print(f"{role.value}:")
        for h in hits:
            print(f"  {h.title:<28} {h.merchant:<12} ${h.price_cents/100:>8,.2f}")
        print()
