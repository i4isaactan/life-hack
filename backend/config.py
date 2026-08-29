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

# --- Catalog ingest: dimension recovery ------------------------------------
# A real merchant feed is missing the dimension that is hardest for them to
# publish, not a random one - HipVan's export has depth and height for most
# upholstery and no width. Requiring three columns rejects most of a real
# catalog, so a small model estimates the missing ones from the dimensions
# that ARE present plus the product's category and title.
#
# Small and cheap on purpose: this runs once per upload over a whole feed, the
# task is interpolation between known anchors rather than reasoning, and the
# result is always labelled as an estimate. With no OPENAI_API_KEY set, an
# offline lookup table runs instead - the same degradation as vision.
DIMENSION_MODEL = os.getenv("DIMENSION_MODEL", "gpt-4o-mini")
# One call covers the whole feed, so the ceiling scales with rows, not items.
DIMENSION_MAX_TOKENS = int(os.getenv("DIMENSION_MAX_TOKENS", "4000"))
# Estimating is opt-out: set to 0 to require merchants to supply real numbers.
DIMENSION_ESTIMATE = os.getenv("DIMENSION_ESTIMATE", "1") not in ("0", "false", "no")
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

# --- Visa Agentic Payments Stack -------------------------------------------
#
# The WebAuthn relying party. RP_ID is a *domain*, never an origin: it carries
# no scheme and no port, and a credential created for one RP ID cannot be used
# by another. That is the property that makes a passkey unphishable, so it is
# configuration rather than something derived from an incoming request - a
# server that trusted the Host header here would let an attacker nominate
# their own relying party.
WEBAUTHN_RP_ID = os.getenv("WEBAUTHN_RP_ID", "localhost")
WEBAUTHN_RP_NAME = os.getenv("WEBAUTHN_RP_NAME", "Room Hack")

# Origins whose passkey assertions we accept. Distinct from CORS_ORIGINS,
# which governs who may call the API; this governs what the *authenticator*
# signed over, and widening it wrongly is what turns a passkey back into a
# password. "null" is deliberately absent: a file:// page has no origin, and
# accepting an unauthenticated one would defeat the check entirely.
WEBAUTHN_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "WEBAUTHN_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,"
        "http://localhost:8080,http://127.0.0.1:8080",
    ).split(",")
    if o.strip()
]

# Default mandate the UI offers when a user first grants agent authority.
# Deliberately modest: a default the user does not think about should be the
# small one, and raising it is a decision they make explicitly.
DEFAULT_AGENT_PER_TXN_CAP_CENTS = int(
    os.getenv("DEFAULT_AGENT_PER_TXN_CAP_CENTS", "200000")
)
DEFAULT_AGENT_TOTAL_CAP_CENTS = int(
    os.getenv("DEFAULT_AGENT_TOTAL_CAP_CENTS", "500000")
)
DEFAULT_AGENT_MANDATE_HOURS = int(os.getenv("DEFAULT_AGENT_MANDATE_HOURS", "24"))

# Hard ceiling on any mandate, whatever the client asks for. Without it the
# caps are entirely client-declared: an agent could request a mandate far
# larger than the user would ever approve, and the only thing standing in the
# way would be the UI it is bypassing.
# A merchant catalogue upload. Large enough for a real feed of a few thousand
# products, small enough that an unbounded upload cannot exhaust memory.
MAX_CATALOG_UPLOAD_BYTES = int(os.getenv("MAX_CATALOG_UPLOAD_BYTES", str(8 * 1024 * 1024)))

MAX_AGENT_MANDATE_CENTS = int(os.getenv("MAX_AGENT_MANDATE_CENTS", "1000000"))

# --- Visa Developer Platform (real sandbox) --------------------------------
#
# The only credentials in this project that authenticate to something outside
# it. Live mode is OFF by default and must be turned on explicitly: VDP
# routinely refuses products a project is not entitled to, so an integration
# that assumed access would break the demo for anyone without it.
VISA_LIVE = os.getenv("VISA_LIVE", "").strip().lower() in ("1", "true", "yes")

# Mutual TLS material. Paths, never contents - a private key in an env var
# ends up in shell history, process listings and crash dumps.
VISA_CERT_PATH = os.getenv("VISA_CERT_PATH", "./secrets/visa_cert.pem")
VISA_KEY_PATH = os.getenv("VISA_KEY_PATH", "./secrets/visa_private_key.pem")
# The sandbox chain (intermediate + root) for OUR client certificate. Note it
# is NOT used to verify Visa's server: sandbox.api.visa.com is DigiCert-signed
# and verifies against the system's public CA store. Loading this bundle as the
# server trust store is what produces "unable to get local issuer certificate".
VISA_CA_PATH = os.getenv("VISA_CA_PATH", "./secrets/visa_ca_bundle.pem")

# HTTP Basic, checked in addition to the certificates.
VISA_USER_ID = os.getenv("VISA_USER_ID", "").strip()
VISA_PASSWORD = os.getenv("VISA_PASSWORD", "").strip()

# Click to Pay / VDES x-pay-token signing.
VISA_API_KEY = (
    os.getenv("VISA_API_KEY") or os.getenv("VISA_X_PAY_TOKEN") or ""
).strip()
VISA_SHARED_SECRET = os.getenv("VISA_SHARED_SECRET", "").strip()

# Message Level Encryption. Visa Direct requires the request body to be
# JWE-encrypted with Visa's public key, on top of mutual TLS. The key ID is
# returned when you register your public key in the VDP dashboard and travels
# in the `keyId` JWE header - Visa uses it to pick which key to decrypt with.
VISA_MLE_KEY_ID = os.getenv("VISA_MLE_KEY_ID", "").strip()
VISA_MLE_PRIVATE_KEY_PATH = os.getenv(
    "VISA_MLE_PRIVATE_KEY_PATH", "./secrets/visa_mle_private.pem"
)
# Visa's own MLE certificate - downloaded from the dashboard beside the key ID.
# Used to ENCRYPT requests; our private key above DECRYPTS the responses.
VISA_MLE_SERVER_CERT_PATH = os.getenv(
    "VISA_MLE_SERVER_CERT_PATH", "./secrets/visa_mle_server.pem"
)

# Visa Direct originator identifiers. Issued with a Visa Direct agreement and
# specific to the originating entity, so they cannot be defaulted to anything
# meaningful - a payout with someone else's BIN is not a payout.
VISA_DIRECT_ACQUIRING_BIN = os.getenv("VISA_DIRECT_ACQUIRING_BIN", "").strip()
VISA_DIRECT_ACQUIRER_COUNTRY = os.getenv("VISA_DIRECT_ACQUIRER_COUNTRY", "702").strip()
VISA_DIRECT_SENDER_ACCOUNT = os.getenv("VISA_DIRECT_SENDER_ACCOUNT", "").strip()

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
# that 3-5 focused references control far better than 14 competing ones.
#
# Set to the largest design the selection step can now produce - one sofa, one
# rug, one table, two chairs, one lamp - so a piece the user is being billed
# for is never silently left out of the picture of what they are buying. That
# was the failure this fixes: a 7-piece design against a 6-reference budget
# rendered six pieces and reported the seventh as "not shown".
#
# The right way to keep this number small is to put less furniture in the room,
# which ROLE_COUNTS now does, rather than to render less of what was chosen.
GEMINI_MAX_REFERENCES = int(os.getenv("GEMINI_MAX_REFERENCES", "8"))

# How freely the composer may deviate from its references. Variety in the
# design now comes from solving several genuinely different layouts, so
# sampling temperature no longer has to supply it - and every bit of it that
# remains is licence to redraw the customer's walls.
GEMINI_COMPOSE_TEMPERATURE = float(os.getenv("GEMINI_COMPOSE_TEMPERATURE", "0.15"))


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
# Castlery serves its product photography from Cloudinary rather than its own
# domain, so the CDN host is what has to be allowed - `castlery.com` never
# appears in an image_url. Cloudinary is multi-tenant, so this is a broader
# grant than the IKEA entries: it permits any Cloudinary-hosted image, not
# only Castlery's. That is acceptable because the fetch is still bounded by
# IMAGE_FETCH_MAX_BYTES and the timeout, and the URLs come from our own
# scrape - but it is the reason to keep this list short and reviewed.
# imgix, like Cloudinary, is multi-tenant - but this entry is a single tenant's
# subdomain (hipvan-images-production), so it grants only that merchant's
# images. Added so the HipVan demo feed can demonstrate reverse image search;
# a real merchant's host is an operator decision, made the same way.
IMAGE_FETCH_ALLOWED_HOSTS = tuple(
    h.strip().lower()
    for h in os.getenv(
        "IMAGE_FETCH_ALLOWED_HOSTS",
        "www.ikea.com,ikea.com,res.cloudinary.com,"
        "hipvan-images-production.imgix.net",
    ).split(",")
    if h.strip()
)
# Renders run concurrently, but a burst of parallel GPU calls is the fastest
# way to hit a rate limit, so keep it modest.
RENDER_CONCURRENCY = int(os.getenv("RENDER_CONCURRENCY", "2"))


# --- Reverse image search ---------------------------------------------------
# When a match is close enough to say "this looks like the LANDSKRONA" rather
# than "here are the nearest sofas we sell". The two bars differ because the
# score scales differ: CLIP image-to-text cosines cluster around 0.2-0.35 and
# essentially never reach 0.6, while text-to-text similarity runs much higher.
# A single threshold would make one signal permanently unconfident.
REVERSE_IMAGE_CONFIDENT_AT = float(os.getenv("REVERSE_IMAGE_CONFIDENT_AT", "0.75"))
REVERSE_TEXT_CONFIDENT_AT = float(os.getenv("REVERSE_TEXT_CONFIDENT_AT", "0.62"))
