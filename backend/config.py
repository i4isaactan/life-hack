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
# Replicate hosts the Tier 2 stack. Unset means the schematic renderer runs
# instead, exactly as an unset OPENAI_API_KEY means mock vision.
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
# Masks are dilated before inpainting so the erase covers contact shadows and
# soft edges the segmenter cuts too tightly.
MASK_DILATE_PX = 12
# Render resolution. SDXL is trained at 1024; going higher costs time without
# improving furniture-scale detail.
RENDER_EDGE_PX = 1024
# One item's full render. Generous: a cold Replicate container can take a while.
RENDER_TIMEOUT_SECONDS = float(os.getenv("RENDER_TIMEOUT_SECONDS", "120"))
# Renders run concurrently, but a burst of parallel GPU calls is the fastest
# way to hit a rate limit, so keep it modest.
RENDER_CONCURRENCY = int(os.getenv("RENDER_CONCURRENCY", "2"))
