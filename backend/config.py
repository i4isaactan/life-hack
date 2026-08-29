"""Configuration and environment loading.

Import-safe: no network calls, no client construction, no side effects beyond
reading .env. Everything here is a plain value so that importing this module
during tests or the solver selftest costs nothing.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# --- Providers -------------------------------------------------------------
# The single switch that decides real-vs-mock for the whole app. When false,
# every provider falls back to a deterministic offline implementation and the
# app still runs end to end.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
HAS_OPENAI = bool(OPENAI_API_KEY)

VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o")
# Intent parsing is short, structured and on the critical path of every turn,
# so it uses a smaller model than vision.
INTENT_MODEL = os.getenv("INTENT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# text-embedding-3-small's native width. The offline hash embedder emits the
# same width so the Qdrant collection schema is provider-independent and a
# collection seeded in one mode is readable in the other.
EMBEDDING_DIM = 1536

# --- Qdrant ----------------------------------------------------------------
# ":memory:" is per-process. Seeding happens in the FastAPI lifespan handler,
# never at import, because uvicorn --reload executes module import twice.
QDRANT_LOCATION = os.getenv("QDRANT_LOCATION", ":memory:")
COLLECTION_NAME = "merchant_inventory"

# --- HTTP ------------------------------------------------------------------
# "null" is the Origin a browser sends for a page opened over file://, which is
# how test_client.html is normally opened. Without it the test page cannot call
# the API at all. Credentials stay off so this explicit list is permitted.
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "null",
]

# Base64 inflates payloads by ~33%, so an 8MB file becomes ~11MB in memory.
MAX_IMAGE_BYTES = 8 * 1024 * 1024
# Room analysis does not need full phone-camera resolution; downscaling cuts
# both vision latency and token cost substantially.
MAX_IMAGE_EDGE_PX = 1024

# Idle proxies and load balancers commonly drop a connection at 30-60s.
SSE_HEARTBEAT_SECONDS = 15.0

# --- Sessions --------------------------------------------------------------
# In-process and lost on restart. Adequate for a demo; the reason it is a plain
# dict rather than Redis is that nothing here needs to outlive the process.
MAX_HISTORY_MESSAGES = 20

# --- Solver ----------------------------------------------------------------
# Distance kept between the room walls and any placed item.
WALL_MARGIN_CM = 5.0
# Clearance degrades through this ladder before an item is skipped outright.
# A strict 40cm is not satisfiable in small rooms: 90cm sofa + 40 + 60cm table
# needs 190 of a 200cm depth, so an absolute rule would skip usable layouts.
CLEARANCE_LADDER_CM = [40.0, 30.0, 20.0, 10.0]
# Ring search: how far to nudge off the ideal anchor, and at what resolution.
SEARCH_STEP_CM = 5.0
SEARCH_SPAN_CM = 60.0

# --- Budget ----------------------------------------------------------------
DEFAULT_BUDGET_CENTS = 150_000


# --- Rendering -------------------------------------------------------------
# Two backends, tried in order. Unset both and the schematic renderer runs,
# exactly as an unset OPENAI_API_KEY means mock vision.
#
# Gemini is preferred: one multi-image call composes the whole room from the
# room photo plus the catalog product shots, so the recommended furniture is
# rendered from its actual photograph rather than from a text description of
# it. Replicate's four-model chain gives mask-level control the single call
# cannot, at roughly 25 predictions and 60-120s per room.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
HAS_GEMINI = bool(GEMINI_API_KEY)

# "Nano Banana 2 Lite": ~4s, $0.034 per 1K image, up to 14 object references.
# Google's stated drop-in replacement for gemini-2.5-flash-image, which retires
# 2 October 2026 and costs more - do not pin the older model in new code.
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-lite-image")

# The model composes up to 14 object references, but Google's own guidance is
# that 3-5 focused references control far better than 14 competing ones. A full
# design is 5-6 pieces, so this caps references at the useful end of that range
# and drops the lowest-priority roles first.
GEMINI_MAX_REFERENCES = int(os.getenv("GEMINI_MAX_REFERENCES", "6"))

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "").strip()
HAS_REPLICATE = bool(REPLICATE_API_TOKEN)

# Pinned model versions. Replicate resolves a bare "owner/name" to whatever is
# newest, which would let a silent upstream change alter output; pinning keeps
# renders reproducible.
GROUNDING_DINO_MODEL = os.getenv(
    "GROUNDING_DINO_MODEL",
    "adirik/grounding-dino:efd10a8ddc57ea28773327e881ce95e20cc1d734c589f7dd01d2036921ed78aa",
)
SAM2_MODEL = os.getenv(
    "SAM2_MODEL",
    "meta/sam-2:fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83",
)
LAMA_MODEL = os.getenv(
    "LAMA_MODEL",
    "allenhooo/lama:e2eb2eb2e12b1e5b0e39e4b3b9e0dcbe2b0f4e0a6d6e7d0f2c8b9a0e1d2c3b4a5",
)
INPAINT_MODEL = os.getenv(
    "INPAINT_MODEL",
    "lucataco/sdxl-inpainting:a5b13068cc81a89a4fbeefeccc774869fcb34df4dbc92c1555e0f2771d49dde7",
)

# Detection confidence floor. Below this a "sofa" is usually a cushion or a
# reflection, and erasing it damages the plate.
DETECTION_THRESHOLD = float(os.getenv("DETECTION_THRESHOLD", "0.35"))
# A busy room photo yields a long tail of increasingly marginal objects. The
# cap keeps one crowded photo from turning into dozens of catalog searches.
MAX_DETECTIONS = int(os.getenv("MAX_DETECTIONS", "12"))
# Detection returns a JSON object per object found, each with a sentence-long
# caption, so it needs far more room than the fixed-shape room analysis.
DETECTION_MAX_TOKENS = int(os.getenv("DETECTION_MAX_TOKENS", "1500"))

# Masks are dilated before inpainting so the erase covers contact shadows and
# soft edges the segmenter cuts too tightly.
MASK_DILATE_PX = 12
# Render resolution. SDXL is trained at 1024; going higher costs time without
# improving furniture-scale detail.
RENDER_EDGE_PX = 1024
# One item's full render. Generous: a cold Replicate container can take a while.
RENDER_TIMEOUT_SECONDS = float(os.getenv("RENDER_TIMEOUT_SECONDS", "120"))

# --- Reverse search --------------------------------------------------------
# How many catalog matches to return per detected object.
REVERSE_SEARCH_LIMIT = int(os.getenv("REVERSE_SEARCH_LIMIT", "5"))
# Cosine similarity below which a match is returned but NOT called an
# identification. Vector search always yields a nearest neighbour, so without
# this floor the closest of eight rugs would be presented as an answer however
# little it resembles the photo.
#
# Measured against this catalog with text-embedding-3-small, querying with the
# detector's own captions: pieces the catalog really holds score 0.73-0.83,
# while objects it has nothing like ("inflatable octopus lamp", "carved stone
# throne") score 0.20-0.41. The gap between those bands is wide and empty,
# which is what makes a single threshold workable; 0.60 sits in the middle of
# it. Re-measure this if the embedding model or the caption prompt changes -
# it is calibrated to both, not a universal constant.
REVERSE_SEARCH_MIN_SCORE = float(os.getenv("REVERSE_SEARCH_MIN_SCORE", "0.60"))

# --- Product image fetching ------------------------------------------------
# Catalog image_urls point at the vendor's CDN, so fetching one is an outbound
# request to a third party on the request path.

# Sent on every product image fetch. A default "Python-urllib/3.x" is the first
# thing a CDN rate-limits or blocks outright, and a silent block degrades every
# render to a text-conditioned guess with no obvious cause.
IMAGE_FETCH_USER_AGENT = os.getenv(
    "IMAGE_FETCH_USER_AGENT",
    "Room Hack/1.0 (+https://example.invalid/roomhack)",
)
IMAGE_FETCH_TIMEOUT_SECONDS = float(os.getenv("IMAGE_FETCH_TIMEOUT_SECONDS", "15"))
# Ceiling on a single product image. The catalog's largest is ~690KB; this
# leaves headroom without letting a redirect to something enormous through.
IMAGE_FETCH_MAX_BYTES = int(os.getenv("IMAGE_FETCH_MAX_BYTES", str(8 * 1024 * 1024)))
# Hosts allowed to serve product imagery. image_url reaches this process from
# the catalog today, but it is data rather than code: an allowlist keeps a
# future catalog edit or a scrape of a different site from turning the server
# into an open fetcher for arbitrary URLs.
IMAGE_FETCH_ALLOWED_HOSTS = tuple(
    h.strip().lower()
    for h in os.getenv("IMAGE_FETCH_ALLOWED_HOSTS", "www.ikea.com,ikea.com").split(",")
    if h.strip()
)
# Renders run concurrently, but a burst of parallel GPU calls is the fastest
# way to hit a rate limit, so keep it modest.
RENDER_CONCURRENCY = int(os.getenv("RENDER_CONCURRENCY", "2"))
