# Room Hack

Agentic spatial commerce. Send a room photo, a budget and an aesthetic; get back
a room analysis, furniture retrieved from a vector catalog, a collision-free
floor plan, and a bill of materials — streamed over SSE. Then refine it in
conversation: *"make it cheaper"*, *"different rug"*, *"why is the sofa there?"*

Then buy it. The agent prices a multi-merchant basket; a Face ID passkey
authorizes it; merchants are paid on Visa Direct.

```
backend/    FastAPI + LangGraph API, the deliverable
frontend/   three pages over that API — landing, design workspace, merchant console
```

**[PAYMENTS.md](PAYMENTS.md)** — how the agent is authorized to spend, and how
merchants get paid. Read that one if you only read one.

---

## Quick start

```bash
uv venv --python 3.11
uv pip install -r backend/requirements.txt
python -m backend.mock_assets          # generate room fixtures, once
.venv/bin/uvicorn backend.main:app --port 8000 --workers 1 --reload
python3 -m http.server 8080 --directory frontend   # → http://localhost:8080
```

| Page | Path |
|---|---|
| Landing | `/` |
| Design workspace | `/app/` |
| Merchant console | `/merchant/` |

**No API key is required.** With `OPENAI_API_KEY` unset the app runs fully
offline: canned room analysis, keyword intent parsing, deterministic local
embeddings, schematic renders. Both paths return identical schemas. Set keys to
upgrade any subsystem independently — `GET /api/health` reports which are live
per subsystem (`vision`, `intent`, `embeddings`, `renderer`), because a key can
be present while one subsystem still falls back.

`localhost:8080` and `:3000` are pre-approved WebAuthn origins, so Face ID
checkout works with no setup. Passkeys need a secure context — elsewhere set
`WEBAUTHN_RP_ID` / `WEBAUTHN_ORIGINS`, or checkout falls back to OTP.

> `--workers 1` matters: Qdrant runs `:memory:`, so each worker would hold a
> separate catalog.

---

## SSE contract

`POST /api/chat` accepts `multipart/form-data`, returns `text/event-stream`.

| Field | Notes |
|---|---|
| `message` | user prompt |
| `budget` | SGD; defaults to 1500 |
| `aesthetic` | e.g. `Japandi` |
| `session_id` | omit on first turn; reuse the one `done` returns |
| `image` | optional, ≤8MB, downscaled to 1024px server-side |
| `room_width_cm` / `room_depth_cm` | real measurements; promote the room to *measured* |
| `openings` | `[{"kind":"door","wall":"north","offset_cm":20,"width_cm":85,"swing_cm":85}]` |
| `irregular` | true for L-shaped rooms, alcoves |

Frames are `event: <type>` / `data: <single-line JSON>` / blank line.

| Event | Meaning |
|---|---|
| `intent` | how the turn was read; fires first |
| `text_delta` | append to the assistant message |
| `room_analysis` | dimensions in cm, finishes, `source` |
| `layout_update` | `placements[]`, `skipped[]`, `withheld[]` |
| `clarification_needed` | measurements that would unlock withheld pieces |
| `alternatives` | per role: the pick, plus other items that fit |
| `bundles` | sets that extend the design |
| `cart_update` | bill of materials |
| `error` / `done` | `done` always fires last, even after a fatal error |

**State accumulates per session.** Measurements, budget, aesthetic, rejected
items and dropped roles all carry forward — a client only sends what changed.

**`layout_update` and `cart_update` are full snapshots, not patches.** Replace
state wholesale; they are idempotent and order-independent.

Coordinates are **centimetres**, origin top-left, x→right, y→down. `z: 0`
renders beneath `z: 1` (rugs under furniture).

### Writing a client

Three things reliably break:

1. **`EventSource` will not work** — it is GET-only and cannot send multipart.
   Use `fetch()` + `res.body.getReader()`.
2. **Never set `Content-Type` on a `FormData` body** — the browser must set the
   multipart boundary.
3. **Normalize `\r\n` → `\n` before splitting on `\n\n`**, or a CRLF anywhere in
   the chain hangs your client forever with no error.

`frontend/app.js` is the reference implementation.

### Endpoints

| Endpoint | Does |
|---|---|
| `POST /api/chat` | design + refine, SSE |
| `POST /api/swap` | replace one item, **re-solve and re-bill**, SSE |
| `POST /api/render` | visualize the design in the user's photo, SSE |
| `POST /api/detect` | identify furniture already in a photo |
| `POST /api/shop-the-look` | reverse image search — upload a crop, get lookalikes |
| `POST /api/checkout/simulate` | order grouped by merchant |
| `GET /api/health` `GET /api/catalog` | status; seeded catalog |
| `POST /api/payment/*` | authorization rail — see [PAYMENTS.md](PAYMENTS.md) |
| `POST /api/passkey/*` `POST /api/agent-token/*` | Visa Agentic stack — see [PAYMENTS.md](PAYMENTS.md) |
| `POST /api/merchant/*` | onboarding, catalog upload, payouts |

**A swap is never merely a re-render.** A different sofa has different
dimensions, so the layout is re-solved — the replacement may not fit, or may
displace the coffee table — and the cart re-billed. The outgoing item is offered
back first, so undo is one click. `layout_only: true` skips the render.

---

## How it works

```
parse_intent ─┬─ explain/chitchat ─→ answer_directly ─→ END
              │
              └─→ analyze_room → build_query → retrieve_items → select_items
                              → solve_layout → build_cart → narrate
```

A LangGraph workflow with one branch. **Every turn is parsed into a structured
`Intent` first** — this is the difference between a search box and a
conversation. Without it *"make it cheaper"* is three tokens in an embedding
query with no more force than *"oak"*.

| `kind` | Effect |
|---|---|
| `design` / `refine` / `replace` | full pipeline, constraints updated |
| `explain` | **answered from state — no retrieval, no solve, no spend** |
| `measure` | dimensions stored |
| `chitchat` | answered directly |

The parser only sets a field the message actually mentions, so unstated
constraints carry forward. *"Make it cheaper"* reads against the current total
(~70%), not a fixed number. *"Different rug"* adds the current pick to the
session's reject list — applied **after** retrieval, not as a search filter, so
a near-duplicate cannot rank into the space it vacated.

`explain` is grounded: the model supplies one sentence, then the pipeline
appends the solver's **actual** `rationale`, `confidence` and `tolerance_cm`.
The model is never the source of a position or a measurement.

### The solver is deterministic

No LLM produces coordinates. It anchors by design intent (sofa on the focal
wall, table in front, chairs flanking, lamps in corners), then resolves
collisions with inflated-AABB clearance. Three invariants, enforced by
`python -m backend.solver`:

1. every placement lies inside the room;
2. no two colliding pieces overlap;
3. an item fitting neither is **skipped with a reason, never forced** — and
   excluded from the cart.

Clearance degrades 40→30→20→10cm before dropping an item, because a strict 40cm
rule is unsatisfiable in small rooms. Rugs are exempt from collision (furniture
sits *on* them) but door swings are hard exclusions for every piece, rugs
included — a rug under a door still stops it opening.

**Precision tiers.** A rug 30cm off-centre is invisible; a sofa must sit flush
to a wall and clear of door swings. So exact-tier pieces are **withheld, not
guessed**, until the room is actually measured. Each withheld item reports what
would unlock it, and `clarification_needed` collects those into questions.

### Rendering into the photo

```
detect → segment → erase          once per photo, cached as an empty-room plate
depth-condition → inpaint         per item, against that plate
```

**Gemini (`GEMINI_API_KEY`) — one call per room, preferred.** Composes the room
photo plus catalog product photos into one image, so each piece renders *from
its actual product photograph*. ~$0.034/room, and lighting is solved once.

**Replicate (`REPLICATE_API_TOKEN`) — one call per item.** Grounding DINO → SAM 2
→ LaMa erase → SDXL inpaint. Slower and dearer, but the only path that
constrains placement with a real mask.

**The image model never decides composition.** A `Placement` already carries
collision-checked coordinates; `geometry.py` fits a homography from the photo's
floor plane and projects them into image space. Generative placement
hallucinates furniture into walls; a solver that already proved its layout is a
better source of truth. A composed render is capped one `confidence` notch below
a masked one, since description is a weaker constraint than a mask.

Vision returns a `floor_quad` for calibration, and **a returned quad is checked
before it is trusted** — vision models get floor orientation backwards often
enough to matter, and an inverted quad still yields a valid homography, so
nothing downstream would catch it. If the floor cannot be traced, rendering
declines rather than guessing.

Without either key you get **schematic** renders — footprints drawn over the
photo in swatch colours. Not pretending to be photorealistic, but they run the
identical projection maths, so a geometry bug shows up with no key and no GPU.

### Reverse image search (CLIP)

The text index answers *"Japandi sofa under S$800"* well but cannot answer
*"find one that looks like this"*. Before CLIP, two opposite captions returned
the same wrong answer:

```
"beige … slim tapered legs, tight back"        -> LANDSKRONA, dark grey
"beige … chunky rounded arms, deep cushions"   -> LANDSKRONA, dark grey
```

Same winner for opposite silhouettes, and a grey sofa for a caption saying
*beige* twice. Given a real product photo and no caption, image vectors identify
the exact item at cosine 1.000, then the same model in other colours, then
genuine lookalikes.

Qdrant holds **two named vectors per item** — `text` (1536-dim) and `image`
(512-dim CLIP ViT-B-32) — so one query filters on shared payload whichever
vector it ranks by. They fuse by **reciprocal-rank fusion, not score averaging**:
CLIP image-text cosines cluster at 0.2–0.35 while text-text runs 0.5–0.7, so
averaging would let text win purely through scale. Each match reports
`image_score`, `text_score` and `matched_by`, so a bad result is attributable.

Embedding is one-time — 175 images in ~38s on CPU, cached to a 405KB `.npz`.
`open_clip_torch` is optional (~2GB); without it every text path still works and
`image_search: false` says which mode produced the results.

### Confidence, not just matches

Vector search always returns a nearest neighbour, so `matches` is never empty —
the closest of eight rugs comes back whether or not it resembles the photo. Only
`confident` distinguishes *"this looks like the LANDSKRONA"* from *"here are the
closest sofas we sell"*. Measured against this catalog, pieces it really holds
score **0.73–0.83** and objects it has nothing like score **0.20–0.41** — a gap
wide enough for a single threshold at 0.60. Both numbers are calibrated to this
embedding model **and** this caption prompt; change either and re-measure.

`/api/detect` returns 503 with no key: "we cannot look" and "we looked and found
nothing" are different answers, and conflating them tells the user their room is
empty.

### Bundles

After a design solves, the agent suggests sets that extend it. **There is no
"customers also bought", deliberately** — the catalog is a product scrape with
no order history, so inventing counts would present fabricated social proof as
real shopper behaviour. Every basis is a checkable property instead:
`same_series` (one product line — 9 series span more than one role),
`completes_room` (a missing role), `style_match` (shared tags and colour family).

A role already in the cart is never suggested again — that would be a swap.
`fits_room` is a floor-area check, not a re-solve: a cheap *necessary* condition
that never claims a placement is guaranteed.

---

## Merchants

`POST /api/merchant/*` turns the catalog's `merchant` string into an account
that can authenticate, publish products, and be paid. Three steps, and **no Visa
relationship is required** — Visa sits between the shopper and their card, not
between us and the merchant.

1. **Register** — `POST /api/merchant/onboard`, secret shown once.
2. **Upload a catalogue** — CSV or JSON, whatever their system already exports.
3. **Keep selling as they do today** — their checkout URL is where shoppers go.

**Ingestion normalizes aggressively but never guesses a value that affects what
a shopper pays or receives** (`ingest/normalizer.py`, five deliberately
inconsistent sample feeds in [`ingest/README.md`](ingest/README.md)). Unit
conversion and field aliasing are safe. Money units are read from the *field
name* (`amount` is cents, `price` is dollars),
never inferred from magnitude — a S$2,000 sofa and a 2000-cent cushion are
equally plausible and guessing wrong is a 100x error.

**Upload is two-phase.** `publish=false` validates and returns exactly what
*would* publish plus every problem found, changing nothing. Silently publishing
28 of 30 rows would leave a merchant believing their catalogue was live when
part of it was not.

**Dimension recovery** (`backend/dimension_ai.py`). A real feed is not missing
dimensions at random — it is missing the one measurement hardest to publish.
HipVan's export omits width on most upholstery, so only 102 of 592 products have
all three and a three-column requirement rejects 83% of a real catalog. Missing
values are estimated from the ones present plus category and seat count, in one
batched call per upload — taking the demo feed from 18 of 54 published rows to
47. An estimate is never laundered into merchant data: recovered rows carry
`estimated_dims` and `estimate_source`, values outside 5–400cm are dropped, and
a row with **no** dimensions keeps its rejection because there is nothing to
anchor to.

**Requests are HMAC-signed, not bearer-authenticated.** A leaked bearer token is
directly replayable; a signature covers method, path, timestamp, nonce and raw
body, so a captured request cannot be redirected or replayed.

```
signature = HMAC-SHA256(secret, "METHOD\npath\ntimestamp\nnonce\nbody")
```

Sign the **raw bytes** — re-serializing parsed JSON changes whitespace and key
order and breaks every signature. A merchant controls neither its own name on a
product (taken from the credential) nor its MCC (platform-assigned, since it
drives the agent mandate's category lock).

Merchants are paid by **Visa Direct** push, live against Visa's sandbox — see
[PAYMENTS.md](PAYMENTS.md#part-4--paying-merchants-visa-direct). Settlement
re-checks KYC at payout rather than trusting it from order time, and payouts are
idempotent because Visa deduplicates on trace numbers.

**Where this stops.** Accepting other people's merchants and settling money to
them makes an operator a **payment facilitator** — KYC/AML, an acquirer or
PayFac licence, chargeback liability, PCI scope. None of that is code, and none
of it is here. The technical half is complete; the regulatory boundary is marked
rather than faked.

---

## Catalog data

**228 real Singapore products from three merchants**, built at import time.
Names, dimensions, photos and links come from the scrapes; nothing is invented.

| Merchant | Source | Listings | Items |
| --- | --- | --- | --- |
| IKEA SG | `products.json` | 1,579 | 175 |
| Castlery SG | `castlery_products.json` | 80 | 52 |
| YEN KAI | hand-entered, no feed | 1 | 1 |

```
accent_chair 88   sofa 79   rug 25   coffee_table 21   floor_lamp 15
```

**Why a second merchant.** IKEA SG alone topped out near S$1,699, so a "premium"
budget selected the same pieces as a mid one. Castlery sits above it, giving the
budget logic a real range. **YEN KAI** is the third shape a merchant takes: a
local supplier with no website, whose catalog is a photograph and a measurement.
Its height is an estimate and its price a placeholder, both labelled as such.

Item ids are merchant-prefixed so no two merchants collide on a shared SKU.

### What the scrape needed first

The raw 1,579 IKEA listings are not a catalog (`ikea_import.py`):

- **Mostly irrelevant** — 146 categories, mostly towels and plant pots. Mapped
  by explicit allowlist, not keywords: *"Bedside tables"* contains "table".
- **Wildly unbalanced** — 701 sofas, but only 28 actual products; the rest are
  colour variants. Capped at **3 per family**, cheapest first, or retrieval
  returns nine shades of the same UPPÅKRA and calls it choice.
- **Some dimensions are wrong in ways the solver cannot survive** — lighting got
  **cord length** as its footprint (a "380cm-wide" lamp is a 3.8m cable); rugs
  got length in the height field, arriving 2.4m tall. Both detectable, both
  repaired from the field the scrape recorded or the product name. Anything
  still out of band is dropped rather than guessed at.

**Style tags are inferred and deliberately sparse.** The scrape's `style` field
is empty on every listing. A first pass put *"Scandinavian"* on **87%** of the
catalog — a tag seven items in eight carry cannot discriminate. Each rule now
demands a specific signal, max three tags, most common now 38%.

**Embedded text is ordered most- to least-discriminating**, because an embedding
weights the whole string and filler dilutes signal. The vendor's own description
leads; `materials` is capped at four and comes last (a fabric sofa listing
`polyurethane, fibreboard, plywood` describes its inside and matches nothing
anyone asks for).

**Prices are SGD cents**, unconverted — the scrape carries no exchange rate.
Castlery's own quirks (a dimension string to parse, colour in the image
filename) are documented in [`scrapers/README.md`](scrapers/README.md).

**Images are hotlinked from third-party CDNs.** Fetches send an explicit
User-Agent, check the host against an allowlist (`image_url` is data, not code —
without this anything that writes a catalog entry can make the server fetch a
private-network URL), and enforce an 8MB ceiling read one byte past the cap.
Neither IKEA nor Castlery is affiliated with this project; mirror the images
before relying on them.

---

## Verification

```bash
python -m backend.seed_data     # catalog integrity, price and size spread
python -m backend.solver        # layout invariants across 3 room sizes
python -m backend.geometry      # floor-plane projection invariants
python -m backend.render_engine # render pipeline, offline
python -m backend.agent         # full graph, offline
python -m backend.ikea_import   # (also castlery_import, yenkai_import)
```

```bash
# -N disables curl buffering; without it you see nothing and conclude wrongly
curl -N -X POST localhost:8000/api/chat \
  -F "message=Design my living room" -F "budget=2500" \
  -F "aesthetic=Japandi" -F "image=@room.jpg"

# refine — neither budget nor aesthetic is resent; both carry forward
curl -N -X POST localhost:8000/api/chat -F "message=make it cheaper" -F "session_id=s_..."

# explain: answered from state, so no retrieval and no solve
curl -N -X POST localhost:8000/api/chat -F "message=why is the sofa there?" -F "session_id=s_..."
```

> macOS has no `timeout`; use `curl --max-time`.

The payment rail walkthrough is in
[PAYMENTS.md](PAYMENTS.md#verifying-it-yourself) — five commands proving an
intent is inert, authorization requires verification, a changed total is
refused, and a replayed idempotency key yields one order rather than two.

---

## Demo data disclaimer

**The catalog is real, and that cuts both ways.** Real scraped products — but a
**snapshot, not a live feed**. Prices and stock go stale, and neither IKEA nor
Castlery is affiliated with or has endorsed this project. Some dimensions are
repaired rather than measured (lamp bases, rug sizes), honest for layout but not
vendor spec sheets. Prices are SGD, unconverted.

**Payment is entirely simulated.** No HTTP request leaves `backend/payments.py`,
no payment SDK is installed, no card number exists in this codebase. Stored
cards are fictional last-fours; the one ending `5454` always declines so the
partial-failure path is demonstrable.

**The Visa Agentic Stack is simulated, but the cryptography is not.** No Visa
endpoint is contacted for tokens or mandates. What *is* real is the FIDO2
verification in `backend/webauthn.py` — real `navigator.credentials`, a real
Touch ID / Face ID prompt, genuine ES256/RS256 checking with origin binding,
challenge single-use, UV enforcement and counter-rollback detection. Two honest
limits: attestation is not checked against the FIDO Metadata Service, and
mandates are signed with a process-local HMAC key rather than an HSM — so this
server *could* mint a mandate the user never approved, which production prevents
with asymmetric keys in tamper-resistant hardware.

**Visa Direct payouts are live against Visa's sandbox** — real MLE-encrypted
requests, real approval codes. Without credentials, `settle` returns
`simulated: true`, names what is missing, and never marks a payout paid that did
not happen.

**Renders are visualizations, not photographs.** A generative render is
conditioned on the product image but reconstructs pixels, so grain, weave and
hardware can differ from what ships. Every `RenderResult` carries
`simulated: true` and a disclaimer string, and a client should show it.
