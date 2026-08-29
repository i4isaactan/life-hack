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

import numpy as np

from . import clip_engine, config
from .clip_engine import ClipEmbedder, embed_catalog
from .models import (
    CameraCalibration,
    CatalogItem,
    Confidence,
    DetectedMatch,
    Detection,
    FloorQuad,
    ReverseSearchResult,
    RoomAnalysis,
    Role,
    Wall,
)
from .seed_data import SEED_ITEMS, validate_seed

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Named vectors on the catalog collection. "text" always exists; "image" only
# when CLIP is installed, so every query that ranks by it must check first.
TEXT_VECTOR = "text"
IMAGE_VECTOR = "image"


# --- Embeddings ------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def hash_embed(text: str) -> list[float]:
    """Deterministic offline embedding: hashed bag of words, L2-normalized.

    Not semantically comparable to a real model — "sofa" and "couch" land in
    unrelated buckets — but it is stable, dependency-free, and good enough for
    lexical matching over a catalog this size. It works better than it should
    here because the embedded text states colours and materials in plain words
    ("beige", "bamboo"), so a query naming one matches literally. Retrieval
    also applies payload filters (role, price, dimensions), which carry most of
    the selection work, so the vector only has to break ties sensibly.
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
        # What the LAST embed() call actually used. `source` says which
        # embedder was configured; this says which one produced the vectors in
        # hand. They diverge whenever an API call fails and degrades to the
        # hash embedder, and callers that compare scores against a calibrated
        # threshold must gate on this one - a hash vector scored against
        # OpenAI catalog vectors is noise, not a weak match.
        self.last_source = "mock"
        if config.HAS_OPENAI:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=config.OPENAI_API_KEY)
                self.source = "openai"
                self.last_source = "openai"
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
                self.last_source = "openai"
                return [d.embedding for d in resp.data]
            except Exception as exc:
                # A mid-demo API failure degrades to offline rather than 500ing.
                log.warning("embedding call failed, falling back offline: %s", exc)
        self.last_source = "mock"
        return [hash_embed(t) for t in texts]


# --- Label vocabulary ------------------------------------------------------

# Free-text detector labels mapped onto the roles we sell. Order matters:
# "coffee table" must beat a bare "table", so specific needles come first.
#
# A label that matches nothing here is not an error - it is a bookshelf, and we
# keep it as a Detection with role=None. This list decides what can be
# REPLACED, not what can be SEEN.
_ROLE_NEEDLES: list[tuple[tuple[str, ...], Role]] = [
    (("coffee table", "cocktail table"), Role.COFFEE_TABLE),
    (("side table", "end table", "nesting table"), Role.COFFEE_TABLE),
    (("armchair", "accent chair", "lounge chair", "recliner"), Role.ACCENT_CHAIR),
    (("floor lamp", "standing lamp", "reading lamp"), Role.FLOOR_LAMP),
    (("sofa", "couch", "loveseat", "settee", "sectional"), Role.SOFA),
    (("rug", "carpet"), Role.RUG),
]


def role_for_label(label: str) -> Role | None:
    """Map a detector's free-text label onto our role vocabulary, or None."""
    text = label.lower()
    for needles, role in _ROLE_NEEDLES:
        if any(n in text for n in needles):
            return role
    return None


def _loads_loose(raw: str) -> dict | None:
    """Parse JSON from model output, tolerating fences and stray prose.

    Models wrap JSON in code fences and explanatory sentences often enough that
    strict parsing throws away good answers, so take the outermost object.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


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

The room's compass is fixed by the camera, not by the building: SOUTH is
always the wall behind the camera, NORTH the wall you are looking at, WEST the
left of frame and EAST the right. Answer in that frame.

focal_wall is the wall the seating should sit against — the wall a sofa's back
would go to. Because the camera defines the compass, this is almost always
"north": the far wall, seen face-on, is where a sofa reads correctly and where
the whole piece stays in frame. Choose "east" or "west" only if the far wall is
genuinely unusable - a full-width window or door - since a sofa against a side
wall is seen edge-on and runs out of the picture. Never choose "south": that is
the camera's own wall and nothing can be placed there.

A large window is a reason NOT to pick that wall, not a reason to pick it -
seating goes against a solid wall and faces the light, it does not block it.

floor_quad traces the visible floor as a quadrilateral, in NORMALIZED image
coordinates where [0,0] is the top-left of the photo and [1,1] the
bottom-right. "near" is the floor edge closest to the camera; "far" is the
edge meeting the back wall.

Two rules that are easy to get backwards - check your answer against both:
  1. near_left[1] and near_right[1] must be LARGER than far_left[1] and
     far_right[1]. Image y grows DOWNWARD, so the near edge is at the BOTTOM
     of the photo (y closer to 1.0) and the far edge is higher up (y closer
     to 0.0). Typical values: near y around 0.9, far y around 0.45.
  2. The near edge must be WIDER than the far edge, because parallel floor
     edges converge with distance.

If the floor is mostly hidden or you cannot trace it, omit floor_quad entirely
rather than guessing.

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
        # Two different reasons to skip the model, reported differently. No
        # photo is the ordinary case - there is nothing to analyse, and the
        # design proceeds from a stated or default room. No client is a
        # misconfiguration the operator probably wants to know about, and it
        # is the only one of the two that means "mock".
        if not image_b64:
            return _MOCK_ROOM.model_copy(update={"source": "default"})
        if self._client is None:
            log.warning(
                "vision provider unavailable; falling back to the default room "
                "despite a photo being supplied"
            )
            return _MOCK_ROOM.model_copy(update={"source": "mock"})
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
            return _MOCK_ROOM.model_copy(update={"source": "mock"})

    def _parse(self, raw: str) -> RoomAnalysis:
        """Parse the model's JSON, tolerating code fences and stray prose."""
        data = _loads_loose(raw)
        if data is None:
            log.warning("vision returned no JSON object; using offline analysis")
            return _MOCK_ROOM.model_copy(update={"source": "mock"})
        try:
            room = RoomAnalysis(**{**data, "source": "openai"})
        except Exception as exc:
            log.warning("vision JSON invalid (%s); using offline analysis", exc)
            return _MOCK_ROOM.model_copy(update={"source": "mock"})

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

    # The near edge must sit LOWER in the frame than the far edge: image y
    # grows downward, and the floor closest to the camera is at the bottom of
    # the photo. Models get this backwards often enough that it is worth
    # checking - an inverted quad still yields a valid, invertible homography,
    # so nothing downstream would catch it. The symptom is furniture at the
    # front of the room rendering at the top of the image.
    near_y = (corners["near_left"][1] + corners["near_right"][1]) / 2
    far_y = (corners["far_left"][1] + corners["far_right"][1]) / 2
    if near_y <= far_y:
        log.warning(
            "floor_quad vertically inverted (near y=%.2f <= far y=%.2f); ignoring",
            near_y,
            far_y,
        )
        return None

    # Perspective also requires the near edge to be no narrower than the far
    # one. A near edge narrower than the far edge means the corners are
    # mislabelled, which produces a mirrored render.
    if near_w < far_w * 0.9:
        log.warning(
            "floor_quad perspective reversed (near %.2f < far %.2f); ignoring",
            near_w,
            far_w,
        )
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


# --- Intent ----------------------------------------------------------------

_INTENT_PROMPT = """You are the intent parser for a furniture shopping
assistant. The user is designing a room. Read their latest message in the
context of the conversation and return ONLY a JSON object.

{
  "kind": "design"|"refine"|"replace"|"explain"|"measure"|"chitchat",
  "budget_cents": <int, only if they state or change a budget>,
  "aesthetic": "<style name, only if they name or change one>",
  "style_note": "<short phrase capturing a vague steer like 'warmer', 'less busy'>",
  "reroll_roles": ["sofa"|"rug"|"coffee_table"|"accent_chair"|"floor_lamp"],
  "remove_roles": [same vocabulary],
  "role_counts": {"accent_chair": 2},
  "max_width_cm": {"sofa": 180},
  "explain_role": "<role they are asking about>",
  "reply": "<your answer, ONLY for explain or chitchat>",
  "reasoning": "<one short sentence on why you read it this way>"
}

Rules:
- kind "design" = a fresh brief. "refine" = adjust what exists (cheaper,
  warmer, bigger). "replace" = swap a specific piece. "explain" = they asked
  why something was chosen. "measure" = they gave room dimensions or door
  positions. "chitchat" = anything else.
- Only include a field if the message actually says so. Omit everything else -
  unmentioned constraints carry forward and must not be reset.
- "make it cheaper" with a known current total means budget_cents roughly 70%
  of that total. "much cheaper" means roughly half.
- "I don't like the X" / "show me another X" -> reroll_roles: ["x"].
- A number of something -> role_counts. "I only want one chair" ->
  {"accent_chair": 1}. "seating for four" -> {"accent_chair": 3} alongside the
  sofa. "two lamps" -> {"floor_lamp": 2}. "one fewer chair" -> the current
  count minus one, which you can read off the current design.
- Use remove_roles ONLY when they want none of something at all ("no rug",
  "drop the lamp"). If they name a number, even zero, use role_counts instead -
  "only one chair" is a count, not a removal, and removing the role would
  delete the chair they asked to keep.
- "the sofa is too big" -> max_width_cm, estimating a sensible ceiling from
  the room width if you know it.
- For kind "explain", ALWAYS set explain_role if the message names or clearly
  implies a piece ("why is the sofa there" -> "sofa", "why that rug" -> "rug").
  Leave it null only when they ask about the design as a whole.
- Set "reply" ONLY for explain and chitchat. For every other kind leave it
  empty - the pipeline generates the response itself. Note that for "explain"
  the pipeline appends the real placement rationale after your reply, so keep
  it to one sentence and never invent positions or measurements."""


# --- Furniture detection ---------------------------------------------------

# One call does both jobs. Asking separately - boxes, then a caption per crop -
# costs one request per object and loses context: a model that can see the
# whole room writes "matching pair of dining chairs" where a lone crop writes
# "wooden chair". The box is what makes the caption checkable, so they belong
# in the same answer.
_DETECT_PROMPT = """You are cataloguing the furniture visible in a room photo.

List every distinct piece of furniture you can see. For each one give a tight
bounding box and a description precise enough to find that piece in a shop.

Respond with ONLY a JSON object, no prose or code fences:
{
  "detections": [
    {
      "label": "<the object's common name, e.g. sofa, armchair, coffee table>",
      "box": [<x1>, <y1>, <x2>, <y2>],
      "confidence": <0-1>,
      "caption": "<appearance: form, colour, material, legs, distinguishing detail>"
    }
  ]
}

box is NORMALIZED to the image: [0,0] is the top-left corner and [1,1] the
bottom-right. x1 < x2 and y1 < y2. Box the object only - not the wall behind
it, not the rug underneath it.

caption is what makes this useful, so make it specific and visual. Write what
you would say to someone finding this exact piece in a catalogue:
  GOOD: "low two-seat sofa in oatmeal boucle, rounded arms, tapered oak legs"
  BAD:  "a nice comfortable sofa"
Name the silhouette, the colour, the material and the legs. Do not guess a
brand, and do not mention the room, the lighting or the photo.

confidence is how sure you are the object is really there and really is what
you called it. Use below 0.4 for anything partly hidden, reflected, or so
small you are inferring it.

Include furniture you cannot name precisely - a bookshelf, a bed, a cabinet -
using its ordinary name. Do NOT include walls, floors, windows, doors, plants,
cushions, throws, books, or anything hanging on a wall.

If the photo shows no furniture at all, return {"detections": []}."""


def _clamp01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _parse_detections(data: dict) -> list[Detection]:
    """Build Detections from model JSON, dropping anything malformed.

    Every rejection here is silent by design: one unparseable entry in a list
    of eight is not worth failing a room analysis over, and the alternative -
    a Detection with a garbage box - would erase the wrong pixels.
    """
    out: list[Detection] = []
    for raw in data.get("detections") or []:
        if not isinstance(raw, dict):
            continue
        box = raw.get("box") or raw.get("bbox")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            x1, y1, x2, y2 = (_clamp01(v) for v in box)
        except (TypeError, ValueError):
            continue
        # A model that returns corners in the wrong order means a box, not a
        # failure; normalising is kinder than dropping it.
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        # Zero-area boxes carry no information and would crop to nothing.
        if x2 - x1 < 1e-3 or y2 - y1 < 1e-3:
            continue

        label = str(raw.get("label", "")).strip().lower()
        if not label:
            continue
        try:
            score = _clamp01(raw.get("confidence", raw.get("score", 0.0)))
        except (TypeError, ValueError):
            score = 0.0

        out.append(
            Detection(
                role=role_for_label(label),
                label=label,
                score=score,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                caption=str(raw.get("caption", "")).strip(),
            )
        )
    return out


class DetectionProvider:
    """Finds and describes the furniture already in a room photo.

    Uses the same multimodal model as VisionProvider rather than a dedicated
    detector. A specialist like Grounding DINO localises better, but it only
    emits boxes for a fixed prompt vocabulary - it cannot say "oatmeal boucle,
    tapered oak legs", which is the whole input to reverse search. The box only
    has to be good enough to crop by; the caption is what gets matched.

    Offline there is nothing to fall back to. A canned detection list would be
    a lie about a specific photo - unlike a canned *room*, which is an honest
    default when no photo exists - so this returns nothing and says so.
    """

    def __init__(self) -> None:
        self._client = None
        self.source = "mock"
        if config.HAS_OPENAI:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=config.OPENAI_API_KEY)
                self.source = "openai"
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("OpenAI detection unavailable: %s", exc)

    @property
    def available(self) -> bool:
        return self._client is not None

    def detect(self, image_b64: str | None) -> list[Detection]:
        if not image_b64 or self._client is None:
            return []
        try:
            resp = self._client.chat.completions.create(
                model=config.VISION_MODEL,
                max_tokens=config.DETECTION_MAX_TOKENS,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _DETECT_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}",
                                    # "high" where room analysis uses "low":
                                    # a box drawn on a downsampled thumbnail is
                                    # too coarse to crop a chair out of, and the
                                    # crop is what gets matched.
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
            )
            raw = (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            log.warning("detection call failed, continuing without it: %s", exc)
            return []

        data = _loads_loose(raw)
        if data is None:
            log.warning("detection returned no JSON object")
            return []

        found = _parse_detections(data)
        kept = [d for d in found if d.score >= config.DETECTION_THRESHOLD]
        # Biggest first, so "the sofa" means the one that dominates the photo.
        kept.sort(key=lambda d: d.area, reverse=True)
        if len(found) != len(kept):
            log.info(
                "detection: kept %d of %d above threshold %.2f",
                len(kept),
                len(found),
                config.DETECTION_THRESHOLD,
            )
        return kept[: config.MAX_DETECTIONS]


def crop_detection(image_b64: str, det: Detection) -> str | None:
    """The detected object alone, as base64 JPEG. None if it cannot be cut."""
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        crop = img.crop(det.crop_box(img.width, img.height))
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=88)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as exc:
        log.warning("could not crop detection %s: %s", det.label, exc)
        return None


def _mock_intent(message: str, budget_cents: int | None) -> "Intent":
    """Keyword intent parsing for the offline path.

    Deliberately crude - it exists so the demo still responds to "cheaper" and
    "different rug" without a key, not to compete with the model.
    """
    from .models import Intent, IntentKind, Role

    global _ROLE_SYNONYMS
    if not _ROLE_SYNONYMS:
        _ROLE_SYNONYMS = _role_synonyms()

    text = (message or "").lower()

    if any(w in text for w in ("why", "how come", "explain")):
        role = next((r for r in Role if r.value.replace("_", " ") in text), None)
        return Intent(
            kind=IntentKind.EXPLAIN,
            explain_role=role,
            reasoning="offline keyword match on 'why'",
        )

    intent = Intent(kind=IntentKind.REFINE, reasoning="offline keyword match")

    if "cheaper" in text or "less expensive" in text or "budget" in text:
        if budget_cents:
            factor = 0.5 if ("much" in text or "way" in text) else 0.7
            intent.budget_cents = int(budget_cents * factor)

    for role in Role:
        # The plain word people actually use as well as the enum's own name:
        # nobody types "accent chair" when they mean "chair".
        names = [role.value.replace("_", " "), role.value, *_ROLE_SYNONYMS.get(role, ())]
        if not any(n in text for n in names):
            continue
        count = _spoken_count(text, *names)
        if count is not None:
            intent.role_counts[role.value] = count
        elif any(w in text for w in ("another", "different", "don't like", "hate", "swap")):
            intent.reroll_roles.append(role)
        elif any(w in text for w in ("no ", "without", "remove", "don't need")):
            intent.remove_roles.append(role)

    # Nothing recognised: treat it as a fresh brief rather than a no-op refine,
    # which would silently ignore the user.
    if not (
        intent.budget_cents
        or intent.reroll_roles
        or intent.remove_roles
        or intent.role_counts
    ):
        intent.kind = IntentKind.DESIGN
    return intent


# The everyday word for a role, where it differs from the enum's name. Only
# words that are unambiguous within this vocabulary: "table" is left out
# because it could equally mean a side table.
_ROLE_SYNONYMS: dict["Role", tuple[str, ...]] = {}


def _role_synonyms() -> dict:
    """Built lazily: Role is imported inside the functions that use it."""
    from .models import Role

    return {
        Role.ACCENT_CHAIR: ("chair", "armchair"),
        Role.FLOOR_LAMP: ("lamp",),
        Role.COFFEE_TABLE: ("coffee table",),
        Role.RUG: ("carpet",),
        Role.SOFA: ("couch",),
    }


# Words for the small numbers a room's worth of furniture ever involves.
_NUMBER_WORDS: dict[str, int] = {
    "no": 0, "zero": 0, "one": 1, "a": 1, "an": 1, "two": 2, "three": 3,
    "four": 4, "five": 5, "six": 6,
}


def _spoken_count(text: str, *names: str) -> int | None:
    """The count attached to a role in a plain sentence, or None.

    Offline-path only, and deliberately narrow: it reads the word immediately
    before the role ("only one chair", "2 lamps") and nothing cleverer. The
    model path handles real phrasing; this exists so the keyless demo does not
    silently ignore "I only want one chair".
    """
    import re

    for name in names:
        for match in re.finditer(rf"(\w+)\s+{re.escape(name)}s?\b", text):
            word = match.group(1)
            if word.isdigit():
                return int(word)
            if word in _NUMBER_WORDS:
                return _NUMBER_WORDS[word]
    return None


class IntentProvider:
    """Turns a chat message into a structured Intent."""

    def __init__(self) -> None:
        self._client = None
        self.source = "mock"
        if config.HAS_OPENAI:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=config.OPENAI_API_KEY)
                self.source = "openai"
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("OpenAI intent unavailable, using offline: %s", exc)

    def parse(
        self,
        message: str,
        history: list | None = None,
        budget_cents: int | None = None,
        current_items: list | None = None,
    ) -> "Intent":
        from .models import Intent

        if self._client is None or not (message or "").strip():
            return _mock_intent(message, budget_cents)

        # Context the parser needs to resolve relative asks: "cheaper" than
        # what, "another one" instead of which.
        context = []
        if budget_cents:
            context.append(f"Current budget: S${budget_cents / 100:,.0f}")
        if current_items:
            context.append(
                "Currently in the design: "
                + ", ".join(
                    f"{i.title} ({i.role.value}, S${i.price_cents / 100:,.0f})"
                    for i in current_items
                )
            )
        turns = [
            {"role": m.role, "content": m.content} for m in (history or [])[-6:]
        ]

        try:
            resp = self._client.chat.completions.create(
                model=config.INTENT_MODEL,
                max_tokens=400,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _INTENT_PROMPT},
                    *turns,
                    {
                        "role": "user",
                        "content": ("\n".join(context) + "\n\n" if context else "")
                        + f"Latest message: {message}",
                    },
                ],
            )
            raw = (resp.choices[0].message.content or "").strip()
            data = json.loads(raw)
            # Drop nulls so unset fields fall back to the model defaults rather
            # than overwriting carried-forward state with None.
            return Intent(**{k: v for k, v in data.items() if v not in (None, "")})
        except Exception as exc:
            log.warning("intent parse failed, using offline reading: %s", exc)
            return _mock_intent(message, budget_cents)


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
        # Image search is optional; when open_clip is absent this stays empty
        # and every image-ranked path degrades to text alone.
        self.clip = ClipEmbedder()
        self._image_vectors: dict[str, np.ndarray] = {}

    def seed(self) -> int:
        """Build the collection. Idempotent: safe to call on every reload."""
        validate_seed()

        # Image vectors, when CLIP is installed. Built before the collection so
        # its presence decides the schema: a collection created without the
        # named image vector cannot gain one without a rebuild.
        self._image_vectors = embed_catalog(SEED_ITEMS, self.clip)

        # recreate-by-delete keeps re-seeding clean without relying on upsert
        # semantics for a collection whose vector width may have changed.
        if self.client.collection_exists(config.COLLECTION_NAME):
            self.client.delete_collection(config.COLLECTION_NAME)
        # Two named vectors in one collection: "text" carries the description
        # and attributes, "image" the product photograph. Named rather than
        # separate collections so a single query can filter on shared payload
        # (role, price, stock) whichever vector it ranks by.
        vectors_config = {
            TEXT_VECTOR: VectorParams(
                size=config.EMBEDDING_DIM, distance=Distance.COSINE
            )
        }
        if self._image_vectors:
            vectors_config[IMAGE_VECTOR] = VectorParams(
                size=clip_engine.EMBED_DIM, distance=Distance.COSINE
            )
        self.client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=vectors_config,
        )

        text_vectors = self.embedder.embed([i.embed_text() for i in SEED_ITEMS])
        points = []
        for idx, (item, text_vec) in enumerate(zip(SEED_ITEMS, text_vectors)):
            named: dict[str, list[float]] = {TEXT_VECTOR: text_vec}
            img = self._image_vectors.get(item.id)
            if img is not None:
                named[IMAGE_VECTOR] = img.tolist()
            points.append(
                PointStruct(
                    # Qdrant needs an int or UUID id; the human id lives in payload.
                    id=idx,
                    vector=named,
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
            )
        self.client.upsert(collection_name=config.COLLECTION_NAME, points=points)

        self._by_id = {i.id: i for i in SEED_ITEMS}
        self._seeded = True
        log.info(
            "seeded %d items (embeddings: %s)", len(points), self.embedder.source
        )
        return len(points)

    def add_items(self, items: list[CatalogItem], *, with_images: bool = True) -> int:
        """Add merchant-published products to the live index.

        Ids are assigned above the seeded range so an ingested product cannot
        overwrite a seeded one.

        Image vectors are built when CLIP is available, so a merchant's
        products are findable by "find one that looks like this" the same way
        seeded ones are - a product that only ever ranked on text is invisible
        to the reverse-image path, which is the feature most likely to be
        demonstrated right after an upload.

        This does mean fetching each product photo. `embed_catalog` caps size,
        honours the SSRF allowlist and caches by (id, url), and an image that
        cannot be fetched is simply absent rather than fatal - so the worst
        case is the previous behaviour: the item is published and ranks on
        text. Pass `with_images=False` to skip the fetch entirely when the
        caller cannot afford the latency.

        NOTE ON THE ALLOWLIST. A merchant's CDN is almost certainly NOT in
        `IMAGE_FETCH_ALLOWED_HOSTS`, so in a default deployment these fetches
        are refused and merchant products still rank on text alone. That is
        the safe default and it is deliberate: the allowlist is what stops a
        catalog entry from making this server fetch an arbitrary URL, and
        merchant-supplied `image_url`s are exactly the untrusted input it
        exists for. Enabling image search for a merchant is therefore an
        operator decision - add their image host to IMAGE_FETCH_ALLOWED_HOSTS -
        rather than something a merchant can grant themselves by uploading a
        feed.
        """
        if not items:
            return 0
        if not self._seeded:
            raise RuntimeError("catalog not seeded yet")

        text_vectors = self.embedder.embed([i.embed_text() for i in items])

        # Only worth fetching if the collection can actually store the result.
        # A collection seeded without CLIP has no "image" vector in its schema,
        # and Qdrant rejects a point carrying one - so check the schema rather
        # than just whether CLIP is importable.
        image_vectors: dict[str, np.ndarray] = {}
        if with_images and self.clip.available and self._has_image_vector():
            try:
                image_vectors = embed_catalog(items, self.clip)
            except Exception as exc:  # noqa: BLE001 - never fail a publish
                # A merchant's catalogue push must not fail because their CDN
                # was slow or their image 404'd.
                log.warning("image embedding failed during ingest: %s", exc)

        # Continue past whatever ids exist rather than restarting at 0.
        base = self.client.count(config.COLLECTION_NAME, exact=True).count
        points = []
        for offset, (item, text_vec) in enumerate(zip(items, text_vectors)):
            named: dict[str, list[float]] = {TEXT_VECTOR: text_vec}
            img = image_vectors.get(item.id)
            if img is not None:
                named[IMAGE_VECTOR] = img.tolist()
            points.append(
                PointStruct(
                    id=base + offset,
                    vector=named,
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
            )
        self.client.upsert(collection_name=config.COLLECTION_NAME, points=points)
        self._by_id.update({i.id: i for i in items})
        # reverse_search gates the entire image branch on this map being
        # non-empty, so a vector that reached Qdrant but not here would be
        # stored and never queried.
        self._image_vectors.update(image_vectors)
        log.info(
            "ingested %d merchant items (%d with image vectors)",
            len(points),
            len(image_vectors),
        )
        return len(points)

    def _has_image_vector(self) -> bool:
        """Whether the live collection's schema carries the image vector.

        Decided by asking Qdrant rather than by trusting `self._image_vectors`:
        the collection is created once, at seed time, with whichever named
        vectors CLIP could supply then. If CLIP was unavailable at seed the
        schema has no "image" vector at all, and a point carrying one is
        rejected - which would turn a merchant's publish into a 500 for a
        reason they could do nothing about.
        """
        try:
            info = self.client.get_collection(config.COLLECTION_NAME)
            params = info.config.params.vectors
            return isinstance(params, dict) and IMAGE_VECTOR in params
        except Exception as exc:  # noqa: BLE001 - absence is a valid answer
            log.info("could not read collection schema: %s", exc)
            return False

    def get(self, item_id: str) -> CatalogItem | None:
        return self._by_id.get(item_id)

    def all_items(self) -> list[CatalogItem]:
        """The live catalog - seeded items plus anything merchants published.

        For consumers that scan rather than search. Resolving an item id must
        go through here (or `get`) rather than through SEED_ITEMS: a product a
        merchant pushed is orderable and can be chosen into a design, and a
        lookup that misses it drops the piece instead of failing loudly.
        """
        return list(self._by_id.values())

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
            using=TEXT_VECTOR,
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

    # Fusion weight. Reciprocal-rank fusion needs a constant that damps the
    # contribution of low-ranked hits; 60 is the value from the original RRF
    # paper and is not tuned here, because tuning it against 175 items would
    # be fitting noise.
    _RRF_K = 60

    def _rank_scores(self, ordered_ids: list[str]) -> dict[str, float]:
        """Reciprocal-rank contribution for one ranked list."""
        return {
            item_id: 1.0 / (self._RRF_K + rank)
            for rank, item_id in enumerate(ordered_ids, start=1)
        }

    def _query_ids(
        self, vector, using: str, must: list[FieldCondition], limit: int
    ) -> dict[str, float]:
        """Run one vector query, returning {item_id: raw cosine} in rank order."""
        try:
            response = self.client.query_points(
                collection_name=config.COLLECTION_NAME,
                query=vector,
                using=using,
                query_filter=Filter(must=must),
                limit=limit,
                with_payload=True,
            )
        except Exception as exc:
            log.warning("%s query failed: %s", using, exc)
            return {}
        out: dict[str, float] = {}
        for point in response.points:
            item_id = (point.payload or {}).get("item_id")
            if item_id:
                # Cosine over normalized vectors is [-1,1]; negative means
                # unrelated, and the model field is [0,1].
                out[item_id] = max(0.0, min(float(point.score or 0.0), 1.0))
        return out

    def search_by_detection(
        self,
        det: Detection,
        limit: int | None = None,
        image_b64: str | None = None,
    ) -> ReverseSearchResult:
        """Find catalog items that look like a detected object.

        Two independent signals, fused:

          IMAGE  CLIP embeds the cropped object (when `image_b64` is supplied)
                 or the caption, and ranks against the product photographs.
                 This is the signal that actually answers "looks like this" -
                 silhouette, proportion and material read directly off the
                 pixels, which no description reproduces. Two sofas both
                 described as "beige fabric two-seat" can look nothing alike.

          TEXT   The caption against the item descriptions. Weaker at
                 appearance, but it knows words CLIP does not ground well -
                 series names, materials, the vendor's own prose.

        Fused with reciprocal-rank fusion rather than by averaging the scores.
        CLIP cosines and text-embedding cosines are on different scales (CLIP
        image-text similarity clusters around 0.2-0.35; text-text runs 0.5-0.7),
        so averaging them would let the text side dominate purely through
        scale. RRF uses only the ordering, which is the part that is
        comparable.

        Falls back to text alone when CLIP is unavailable, so this path works
        without torch installed - it is simply less good.
        """
        if not self._seeded:
            raise RuntimeError("CatalogIndex.seed() must run before search()")

        limit = limit or config.REVERSE_SEARCH_LIMIT

        must: list[FieldCondition] = [
            FieldCondition(key="in_stock", match=MatchValue(value=True))
        ]
        # Filtered by role when we recognise one, because a bookshelf should
        # not come back as a sofa - and unfiltered when we do not, so an object
        # outside our catalog can still surface its nearest relatives.
        if det.role is not None:
            must.append(
                FieldCondition(key="role", match=MatchValue(value=det.role.value))
            )

        # Pull deeper than `limit` from each signal: fusion can only rank what
        # both lists contain, and a strong image match sitting at text rank 12
        # is exactly the result worth surfacing.
        depth = max(limit * 3, 15)

        image_scores: dict[str, float] = {}
        if self._image_vectors and self.clip.available:
            probe = None
            if image_b64:
                # The cropped object itself is the best possible probe: it is
                # the actual thing the user is pointing at, not a description
                # of it.
                try:
                    probe = self.clip.embed_image(base64.b64decode(image_b64))
                except Exception as exc:
                    log.info("could not embed detection crop: %s", exc)
            if probe is None and det.caption:
                # CLIP aligns text and images in one space, so a caption still
                # searches the photographs - just less precisely than pixels.
                probe = self.clip.embed_text(f"{det.label}. {det.caption}")
            if probe is not None:
                image_scores = self._query_ids(
                    probe.tolist(), IMAGE_VECTOR, must, depth
                )

        text_scores: dict[str, float] = {}
        if det.caption:
            # The label is prepended so the object's category carries weight
            # even when the caption is all adjectives.
            query = f"{det.label}. {det.caption}"
            text_scores = self._query_ids(
                self.embedder.embed([query])[0], TEXT_VECTOR, must, depth
            )

        if not image_scores and not text_scores:
            # Without a caption AND without a crop there is nothing to match
            # on: the label alone ("sofa") describes every sofa we sell equally
            # well, so ranking on it would return an arbitrary five and dress
            # them up as an answer.
            return ReverseSearchResult(detection=det, matches=[], confident=False)

        fused = self._rank_scores(list(image_scores))
        for item_id, contribution in self._rank_scores(list(text_scores)).items():
            fused[item_id] = fused.get(item_id, 0.0) + contribution

        ordered = sorted(fused, key=lambda i: -fused[i])[:limit]
        # RRF scores are tiny and unitless; rescaling against the best result
        # keeps `score` in the 0-1 the model documents without implying it is a
        # probability.
        best = max(fused.values()) if fused else 1.0

        matches: list[DetectedMatch] = []
        for item_id in ordered:
            item = self._by_id.get(item_id)
            if item is None:
                continue
            img = image_scores.get(item_id)
            txt = text_scores.get(item_id)
            matches.append(
                DetectedMatch(
                    item_id=item.id,
                    title=item.title,
                    merchant=item.merchant,
                    price_cents=item.price_cents,
                    currency=item.currency,
                    image_url=item.image_url,
                    checkout_url=item.checkout_url,
                    score=max(0.0, min(fused[item_id] / best, 1.0)),
                    image_score=img,
                    text_score=txt,
                    matched_by=(
                        "both" if img is not None and txt is not None
                        else "image" if img is not None
                        else "text"
                    ),
                )
            )

        # Confident when the leading result is genuinely close, not merely the
        # nearest of whatever we stock. The image bar is lower than the text
        # one because CLIP image-text cosines simply do not reach 0.6.
        top = matches[0] if matches else None
        confident = bool(
            top
            and (
                (top.image_score or 0.0) >= config.REVERSE_IMAGE_CONFIDENT_AT
                or (top.text_score or 0.0) >= config.REVERSE_TEXT_CONFIDENT_AT
            )
        )
        return ReverseSearchResult(
            detection=det, matches=matches, confident=confident
        )

    def _is_confident(
        self, det: Detection, matches: list[DetectedMatch]
    ) -> bool:
        """Whether the top match is an identification or just a nearest neighbour.

        Score alone decides this, and one alternative is worth ruling out
        explicitly because it looks more principled than it is: requiring the
        top match to beat the runner-up by some margin. Measured against this
        catalog, margin does not separate the two cases - true matches lead by
        0.01-0.07 and unrelated ones by 0.00-0.02, overlapping ranges. Worse,
        it inverts on the clearest evidence: the catalog stocks the same sofa
        in several finishes, so recognising a LANDSKRONA produces near-tied
        top scores and a margin test reads that as uncertainty. Absolute score
        has no such failure, separating 0.73-0.83 from 0.20-0.41.
        """
        # The hash embedder shares no semantic space with the catalog prose -
        # "boucle" and "bouclé" land in unrelated buckets - so its scores are
        # not comparable to this threshold and must never read as confident.
        #
        # Gated on last_source, not source: a rate-limited or timed-out
        # embedding call degrades to the hash embedder silently, and the
        # resulting scores are noise. Checking the configured provider instead
        # would let that noise through as an identification.
        if not matches or self.embedder.last_source != "openai":
            return False
        # An unrecognised role means the search ran unfiltered across the whole
        # catalog. The nearest sofa to a bookshelf is still a sofa, and calling
        # that an identification is exactly the overclaim this flag prevents.
        if det.role is None:
            return False
        return matches[0].score >= config.REVERSE_SEARCH_MIN_SCORE

    def identify_room(
        self, detections: list[Detection], limit: int | None = None
    ) -> list[ReverseSearchResult]:
        """Reverse-search every detected object in one photo."""
        return [self.search_by_detection(d, limit=limit) for d in detections]


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
