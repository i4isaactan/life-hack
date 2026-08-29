# Room Hack

Agentic spatial commerce. Send a room photo, a budget and an aesthetic; get
back a room analysis, furniture retrieved from a vector catalog, a
collision-free floor plan, and a bill of materials — streamed over SSE. Then
refine it in conversation: *"make it cheaper"*, *"different rug"*, *"why is the
sofa there?"*

```
backend/    FastAPI + LangGraph API, the deliverable
frontend/   two pages over that API — a landing page and the design workspace
```

---

## Quick start

```bash
uv venv --python 3.11
uv pip install -r backend/requirements.txt
python -m backend.mock_assets          # generate room fixtures, once
.venv/bin/uvicorn backend.main:app --port 8000 --workers 1 --reload
```

Then serve the frontend from any static host:

```bash
python3 -m http.server 8080 --directory frontend   # → http://localhost:8080
```

`localhost:8080` and `localhost:3000` are pre-approved WebAuthn origins, so the
Face ID / Touch ID checkout works with no extra setup. Serving from any other
host needs `WEBAUTHN_RP_ID` and `WEBAUTHN_ORIGINS` set, and a secure context
(HTTPS) — passkeys are unavailable over plain HTTP off localhost, and the
checkout falls back to the OTP step-up there.

```
frontend/
  index.html        /       landing page. No JavaScript at all.
  app/index.html    /app/   the design workspace
  shared.css                design tokens + primitives, imported by both
  app.js                    API client and workspace UI
```

`/app/` is a real directory, so this works on any static host with no routing
config and no build step. It talks to `http://127.0.0.1:8000` by default;
override with `localStorage.setItem("roomhack_api", "https://…")`.

No build step, no dependencies. The workspace exercises the whole API — chat,
swap, render, payment and health — and renders the layout as an SVG floor plan
straight from the solver's centimetres.

**No API key is required.** With `OPENAI_API_KEY` unset the app runs fully
offline: canned room analysis, keyword intent parsing and deterministic local
embeddings. Set the key (in the environment or a `.env`) to use `gpt-4o`,
`gpt-4o-mini` for intent, and `text-embedding-3-small`. Both paths return
identical schemas, so nothing downstream changes.

`GET /api/health` reports which providers are live, **broken out per
subsystem** — `vision`, `intent`, `embeddings` and `renderer` each report their
own source. A rolled-up flag is not enough: a key can be present while one
subsystem still falls back, and the rolled-up version sends you to the wrong
place. The same breakdown is logged at startup, with a warning naming any
subsystem that fell back despite a key being set.

**"source": "default" on a room analysis is not a fallback.** Vision only runs
when a photo is supplied; without one there is nothing to analyse, so the
design proceeds from stated or default dimensions and says `default`. `mock`
means the vision provider was genuinely unavailable or its call failed — those
are the ones worth investigating.

Likewise `renderer.method: "schematic"` alongside `can_compose: true` is
expected: `method` describes the *per-item* path, which needs
`REPLICATE_API_TOKEN`. With only `GEMINI_API_KEY` set you get the composed
whole-room path, which is the preferred one.

> `--workers 1` matters: Qdrant runs in `:memory:` mode, so each worker would
> otherwise hold a separate catalog.

---

## SSE contract

`POST /api/chat` accepts `multipart/form-data` and returns `text/event-stream`.

| Field | Type | Notes |
|---|---|---|
| `message` | string | user prompt |
| `budget` | number | SGD; defaults to 1500 |
| `aesthetic` | string | e.g. `Japandi`, `Industrial` |
| `session_id` | string | omit on the first turn; reuse the one `done` returns |
| `image` | file | optional, ≤8MB, downscaled to 1024px server-side |
| `room_width_cm` | number | actual measurement; promotes the room to *measured* |
| `room_depth_cm` | number | must be sent with `room_width_cm` |
| `openings` | JSON | `[{"kind":"door","wall":"north","offset_cm":20,"width_cm":85,"swing_cm":85}]` |
| `irregular` | bool | true for L-shaped rooms, alcoves, chimney breasts |

Measurements accumulate per session — give dimensions on one turn and door
positions on the next; both persist.

**So do preferences.** `budget` and `aesthetic` are authoritative only on the
turn that sets them: omit them on a follow-up and the conversation's own values
carry forward rather than resetting to the default. Rejected items, dropped
roles and size ceilings persist the same way, so a client only ever sends what
actually changed.

Frames are `event: <type>` / `data: <single-line JSON>` / blank line.

| Event | Payload | Meaning |
|---|---|---|
| `intent` | `{intent}` | how the turn was read; fires first, before any work |
| `text_delta` | `{text}` | append to the assistant message |
| `room_analysis` | `{room}` | dimensions in cm, finishes, `source: openai\|mock` |
| `layout_update` | `{layout}` | `placements[]`, `skipped[]`, `withheld[]` |
| `clarification_needed` | `{questions[], withheld[]}` | measurements that would unlock withheld pieces |
| `alternatives` | `{options[]}` | per role: the pick, plus other catalog items that fit |
| `bundles` | `{bundles[]}` | suggested sets that extend the design; absent when there is nothing to add |
| `cart_update` | `{cart, subtotal_cents, budget_cents, over_budget}` | bill of materials |
| `error` | `{message, code, fatal}` | failure inside the stream |
| `done` | `{session_id, elapsed_ms}` | always last |

```
event: layout_update
data: {"type":"layout_update","layout":{"room_width_cm":420,"room_depth_cm":330,
  "placements":[{"item_id":"ikea-59516727","name":"LANDSKRONA 2-seat sofa",
    "role":"sofa","x_cm":128,"y_cm":237,"w_cm":164,"d_cm":89,"rotation":0,"z":1,
    "swatch":"#A9BBA2","price_cents":66900,"merchant":"IKEA"}],
  "skipped":[]}}
```

Two guarantees worth designing against:

- **`layout_update` and `cart_update` are full snapshots, not patches.** Replace
  your state wholesale — they are idempotent and order-independent.
- **`done` always fires**, including after a fatal `error`, so you can always
  re-enable the composer when the stream ends.

Coordinates are **centimetres**, origin top-left, x→right, y→down. Map them
through an SVG `viewBox` and aspect ratio takes care of itself. `z: 0` renders
beneath `z: 1` (rugs sit under furniture).

### Reverse image search (CLIP)

`POST /api/shop-the-look` takes a cropped photo of one piece of furniture and
returns catalog items that look like it.

**Why a second vector.** The text index answers *"Japandi sofa under S$800"*
well, but it cannot answer *"find one that looks like this"*. Descriptions blur
exactly what furniture shopping turns on. Before CLIP, two opposite captions
returned the same wrong answer:

```
"beige … slim tapered legs, tight back"        -> LANDSKRONA, dark grey
"beige … chunky rounded arms, deep cushions"   -> LANDSKRONA, dark grey
```

Same winner for opposite silhouettes, and a grey sofa for a caption saying
*beige* twice. With image vectors the two queries separate, and both return
actually-beige sofas — VIMLE (which genuinely has chunky arms) for the second.

Given a real product photo and **no caption at all**, it identifies the exact
item at cosine 1.000, then the same model in other colours, then genuine
lookalikes:

```
1.000  POÄNG Armchair - birch/Knisa light beige   <- the query image
0.937  POÄNG Armchair - birch/Knisa black
0.924  ÅRSUNDA Armchair - Knisa light grey
```

**How it is stored.** Qdrant holds two named vectors per item: `text`
(1536-dim, the description and attributes) and `image` (512-dim, CLIP ViT-B-32
over the product photograph). Named vectors rather than two collections, so a
single query filters on shared payload — role, price, stock — whichever vector
it ranks by.

**How the two are fused.** Reciprocal-rank fusion, not score averaging. CLIP
image-text cosines cluster around 0.2–0.35 while text-text similarity runs
0.5–0.7, so averaging would let the text side win purely through scale. RRF
uses only the ordering, which is the part that is comparable. Each match
reports `image_score`, `text_score` and `matched_by` (`image` / `text` /
`both`) so a bad result can be attributed to a signal rather than guessed at.

**Cost.** Embedding is one-time: 175 images download and embed in ~38s on CPU,
then cache to `backend/assets/clip_cache.npz` (405KB), making restarts
instant. The cache key includes the image URL, so repointing an item at a new
photo re-embeds it automatically.

**It is optional.** `open_clip_torch` is ~2GB installed. Without it the catalog
still seeds, every text path works, and reverse search falls back to
caption-only matching — the response's `image_search: false` says which mode
produced the results. Verified by simulating the missing import.

### Bundles and sets

After a design is solved, the agent suggests sets that extend it — emitted as a
`bundles` frame, and absent entirely when the design is already complete.

**There is no "customers also bought" here, and that is deliberate.** The
catalog is a product scrape: attributes, prices, dimensions. It carries no
order history, no baskets and no view logs, so there is no co-purchase signal
to compute. Inventing counts would present fabricated social proof as real
shopper behaviour, which pushes real spending decisions with invented
evidence. Every basis below is instead a checkable property of the products:

| `basis` | Label | Meaning |
|---|---|---|
| `same_series` | Matching set | Both pieces are one IKEA product line — same frame, fabric and proportions |
| `completes_room` | Finishes the room | A role the design lacks, filled to match what is already chosen |
| `style_match` | Completes the look | Different lines sharing style tags and a colour family |

`same_series` is the strongest signal the data contains and ranks first. The
catalog has **9 series spanning more than one role** — LANDSKRONA, SÖDERHAMN,
VIMLE and others where the sofa and armchair are genuinely the same range. That
is a fact parsed from the product names, not an inference.

Each bundle carries the `reason` shown to the user, stating the actual basis,
plus `added_cents` (only the pieces not already owned), `affordable` judged on
that added spend, and `fits_room`.

Two rules worth naming:

- **A role already in the cart is never suggested again.** A room needs one
  sofa; offering a second as an "addition" is a swap, which `/api/swap`
  already does.
- **`fits_room` is a floor-area check, not a re-solve.** Re-solving every
  candidate would cost far more than a suggestion is worth, so this is a cheap
  *necessary* condition — it rules out the obviously impossible and never
  claims a placement is guaranteed.

Style agreement adapts to how much vocabulary each piece has: two shared tags
when both have two or more, one when either is single-tagged. Tags are inferred
and deliberately sparse (96 of 175 items carry exactly one), so a flat
two-tag rule is unsatisfiable for most of the catalog and silently yields no
bundles at all.

### Conversational refinement

Every turn is parsed into a structured `Intent` before anything else runs. This
is the difference between a search box and a conversation: without it *"make it
cheaper"* is three tokens in an embedding query with no more force than *"oak"*,
and the design comes back at the same budget.

| `kind` | Means | Effect |
|---|---|---|
| `design` | a fresh brief | full pipeline |
| `refine` | adjust what exists — cheaper, warmer, bigger | full pipeline, constraints updated |
| `replace` | swap a specific piece | full pipeline, that role re-picked |
| `explain` | *"why is the sofa there?"* | **answered from state — no retrieval, no solve, no spend** |
| `measure` | dimensions or door positions | measurements stored |
| `chitchat` | anything else | answered directly |

The parser only sets a field when the message actually says so, so an
unmentioned constraint carries forward instead of being reset each turn. What
it can express:

- **`budget_cents`** — *"make it cheaper"* is read against the current total
  (roughly 70%; *"much cheaper"* roughly half), not against a fixed number.
- **`reroll_roles`** — *"show me a different rug"*. The item currently in that
  role is added to the session's reject list, so the next search cannot hand
  back the same piece.
- **`remove_roles`** — *"I don't need a rug"*. Dropped from the design.
- **`max_width_cm`** — *"the sofa is too big"* narrows that one role below what
  the room alone would allow.
- **`style_note`** — vague steers like *"warmer"* are folded into the retrieval
  query rather than replacing the aesthetic, since they refine a style rather
  than naming a new one.

Rejections are applied **after** retrieval, not as a search filter. Filtering
the search would let a rejected item's near-duplicate rank into the space it
vacated, which is not what the user asked for.

`explain` is the one that pays for itself. It answers from the existing design,
and what it says is grounded: the language model supplies one sentence, then the
pipeline appends the solver's **actual** `rationale`, `confidence` and
`tolerance_cm`. The model is never the source of a position or a measurement.

```
> why is the sofa there?

The Mika Settee was chosen for its sleek design and comfort, aligning well
with the Japandi style you requested.
• Mika Settee sits positioned against the wall from measured dimensions
  — high confidence, ±15cm.
```

Withheld pieces are the most interesting thing to explain, and are reported as
what they are: a deliberate refusal, not a failure.

Offline, intent falls back to keyword matching — deliberately crude. It exists
so the demo still responds to *"cheaper"* and *"different rug"* without a key,
not to compete with the model.

### Writing a client

Three things reliably break here:

1. **`EventSource` will not work.** It is GET-only and cannot send
   `multipart/form-data`. Use `fetch()` + `res.body.getReader()`.
2. **Never set `Content-Type` on a `FormData` body.** The browser must set it
   to inject the multipart boundary.
3. **Normalize `\r\n` → `\n` before splitting on `\n\n`.** If anything in the
   chain emits CRLF, a naive `indexOf("\n\n")` never matches and your client
   hangs forever with no error.

Also decode with `decoder.decode(value, {stream: true})` so a multi-byte
character split across chunks is not corrupted, and consume only *complete*
frames — keep the remainder buffered. `frontend/app.js` is the reference
implementation.

### Other endpoints

- `GET /api/health` — status and active providers
- `GET /api/catalog` — the seeded catalog, for debugging retrieval
- `POST /api/checkout/simulate` — `{item_ids[], session_id?}` → order grouped by merchant
- `POST /api/render` — visualize the design in the user's photo; see below
- `POST /api/swap` — replace one item with an alternative and re-solve; see below
- `POST /api/shop-the-look` — reverse image search: upload a cropped object, get visual lookalikes; see below
- `POST /api/detect` — identify the furniture already in a photo; see below
- `POST /api/payment/*` — the simulated authorization rail; see below

### Image URLs

Catalog `image_url`s are IKEA's own `https://` product photos, which a browser
loads directly. Every outbound payload still passes through a single
normalisation hook: `http(s)://`, `data:` and `/assets/…` pass through, and
anything else (a `file://` path, say) is dropped rather than sent, since a
client can handle a missing image but not a broken one.

### `POST /api/swap`

`{"session_id": "s_…", "role": "sofa", "item_id": "ikea-59516727", "layout_only": false}`
→ `text/event-stream`.

Every `alternatives` frame carries catalog items the user could pick instead —
retrieval already found them, and discarding them would present one choice as
though it were the only one. Clicking one calls this endpoint.

| Event | Payload | Meaning |
|---|---|---|
| `swap_started` | `{role, from:{item_id,name}, to:{item_id,name}}` | what is being exchanged |
| `layout_update` | `{layout}` | re-solved around the new footprint |
| `cart_update` | `{cart, …}` | re-billed |
| `alternatives` | `{options[]}` | re-priced against the new pick |
| `room_render` / `render_failed` | | unless `layout_only` |
| `done` | `{session_id, elapsed_ms}` | always last |

**A swap is never merely a re-render.** A different sofa has different
dimensions, so the layout is re-solved — the replacement may not fit, or may
displace the coffee table that was positioned against the old one — and the
cart re-billed. Returning a new picture over a stale plan would show the user
something the solver never agreed to.

The outgoing item is offered back first in the new `alternatives`, so undo is
one click and never requires replaying the conversation. `layout_only: true`
skips the render, which is how a client previews what a swap costs before
paying for an image.

`Alternative.affordable` accounts for the swap being a *replacement*: the
outgoing item's price returns to the budget before the incoming one is charged,
so a like-priced option stays affordable even at budget. Unaffordable options
are still offered — the user may want to spend more — but flagged.

### `POST /api/detect`

Multipart: `image` (required), `match` (bool, default false), `limit` (int).
Identifies the furniture already in a photo, and optionally looks each piece up
in the catalog. Boxes are normalized `[0,1]`, so a client can draw them over the
photo at any display size.

```json
{"count": 2, "results": [
  {"detection": {"role": "sofa", "label": "sofa", "score": 0.9,
                 "x1": 0.0, "y1": 0.3, "x2": 0.6, "y2": 0.7,
                 "caption": "low green velvet sofa, tufted back, tapered oak legs"},
   "matches": [{"item_id": "…", "title": "LANDSKRONA 2-seat sofa …",
                "price_cents": 66900, "score": 0.64}],
   "confident": true}]}
```

`role` is `null` for furniture we recognise but do not sell — a bookshelf, a
bed. Those stay in the list, because a user comparing this against their own
room should see that we spotted the bookshelf; they simply cannot be replaced.

**`confident` is the flag to gate wording on.** Vector search always returns a
nearest neighbour, so `matches` is never empty for a role we stock — the closest
of eight rugs comes back whether or not it resembles the photo. Only `confident`
distinguishes *"this looks like the LANDSKRONA"* from *"here are the closest
sofas we sell"*. It requires a recognised role and a top score above
`REVERSE_SEARCH_MIN_SCORE`, and is always false on the offline embedder, whose
scores are not comparable. See **Reverse search** under *How it works*.

Returns 503 when `OPENAI_API_KEY` is unset: "we cannot look" and "we looked and
found nothing" are different answers, and a client that conflated them would
tell the user their room is empty.

### `POST /api/render`

`{"session_id": "s_…", "item_ids": []}` → `text/event-stream`. Same framing as
`/api/chat`. Renders every placed item, one frame each as it finishes; an empty
`item_ids` means all of them.

| Event | Payload | Meaning |
|---|---|---|
| `render_started` | `{total, method, erased}` | `method` is `composed`, `generative` or `schematic` |
| `room_render` | `RoomRender` + `{progress:{done,total}}` | the whole design in one image (`composed`) |
| `render_update` | `RenderResult` + `{progress:{done,total}}` | one finished visualization (per-item) |
| `render_failed` | `RenderFailure` | that item could not be rendered, and why |

`RenderFailure.reason` distinguishes `no_photo`, `no_calibration` (the floor
plane could not be traced), `out_of_frame` (the piece sits in front of the
photographed floor — a wider shot would fix it), `not_placed` and
`provider_error`.
| `done` | `{session_id, rendered, total, elapsed_ms}` | always last |

Call `/api/chat` first: rendering needs the photo, room and layout from a
completed design. An unknown session is `404`, a session with no design `409`.

**Two shapes of response.** With a composing backend configured the whole design
comes back as a single `room_render` frame — every piece in one image, `total: 1`.
Otherwise each item is rendered separately and arrives as its own
`render_update`. Pass `"per_item": true` to force the per-item path when a
composer is available (useful for comparing them, or re-rendering one swap).
`RoomRender.omitted` names any placed item the model was not shown, so a client
can say what is missing rather than leaving the user to notice.

Frames arrive in **painter's order** — rugs first, then back of the room
forward — so a client can composite them in arrival order and get correct
occlusion. Every item yields exactly one `render_update` *or* one
`render_failed`; nothing is silently dropped.

`RenderResult.image_url` is a data URI offline and a provider URL when
generative. It also carries `confidence`, `method`, the `replaced` piece it
stands in for, and a `simulated: true` disclaimer — **a render is an
approximation of the product in the room, not a photograph of it**, and clients
should label it that way.

---

### Reverse search

`analyze_room` does two things with a photo: estimate the room, and catalogue
the furniture already in it. The second is what `POST /api/detect` exposes, and
it feeds two things that used to run blind.

**Retrieval.** What someone already owns is a stronger style signal than the
adjective they typed, so the detections' captions are appended to the embedding
query — capped and last, so "something brighter" is not drowned out by the beige
they are trying to replace.

**Matching.** Each detection carries a `caption` written in the same visual
vocabulary as `CatalogItem.embed_text` — *"low two-seat sofa in oatmeal bouclé,
tapered oak legs"* — so both sides embed into one space and the existing text
index answers "do we sell this?" with no second vector per item.

The obvious upgrade is a real image index: CLIP over the 174 product shots, a
named `image` vector in Qdrant, and the detection crop (`Detection.crop_box`,
already implemented) embedded into the same space. That matches silhouette and
texture directly rather than through prose. It needs a GPU-backed embedder —
`REPLICATE_API_TOKEN` or a local `torch` — which is why the caption bridge is
what ships: it works on the providers already configured.

**On the confidence threshold.** Vector search always returns a nearest
neighbour, so the question is not *what* matched but whether it means anything.
Measured against this catalog with `text-embedding-3-small`, pieces it really
holds score **0.73–0.83** and objects it has nothing like score **0.20–0.41**.
The gap is wide and empty, so a single threshold at 0.60 works.

A margin test — requiring the top match to beat the runner-up — looks more
principled and is worse: the margins overlap between the two cases, and it
inverts on the clearest evidence, since the catalog stocks the same sofa in
several finishes and *recognising* one produces near-tied scores. Both numbers
are calibrated to this embedding model **and** this caption prompt; change
either and re-measure.

---

## How it works

```
parse_intent ─┬─ explain/chitchat ─→ answer_directly ─→ END
              │
              └─→ analyze_room → build_query → retrieve_items → select_items
                              → solve_layout → build_cart → narrate

              ↓ /api/render                    ↓ /api/swap
        one composed image              re-solve + re-bill + re-render
```

A LangGraph workflow with one branch. `parse_intent` runs first and routes:
a turn that only asks *why* is answered from existing state, skipping
retrieval, solving and any API spend. Everything else takes the full pipeline.
Nodes stream via `get_stream_writer()` with `stream_mode="custom"` — chosen
over `astream_events`, which on a graph this small emits only generic
`on_chain_*` frames with no domain meaning.

**Retrieval** filters by role, price, stock and *room dimensions*, so oversized
pieces never reach the solver, and keeps six candidates per role. Items the
user has rejected are dropped from the shortlist afterwards, and a role they
asked to remove is skipped entirely. **Selection**
buys in order of how much each piece makes the room usable (seating first),
reserving budget for seating so a premium rug cannot price out the sofa — then
**offers the candidates it did not buy** as `alternatives`, since they are
equally valid picks and the user is the one with the taste.

### Replacing furniture in the photo

`/api/render` erases the furniture already in the room and puts the recommended
pieces where the solver put them:

```
detect → segment → erase          once per photo, cached as an empty-room plate
depth-condition → inpaint         per item, against that plate
```

Two backends, either of which may be configured:

**Gemini (`GEMINI_API_KEY`) — one call per room, preferred.**
`gemini-3.1-flash-lite-image` ("Nano Banana 2 Lite") composes up to 14 object
references, so the room photo and the catalog product photos go in together and
one image comes back. Each recommended piece is rendered *from its actual
product photograph*, which is the fidelity the per-item path could only
approximate. It also costs ~$0.034 a room instead of ~$0.20, and lighting is
solved once for the whole scene rather than six times independently.

**Replicate (`REPLICATE_API_TOKEN`) — one call per item.** Grounding DINO finds
the existing furniture, SAM 2 cuts masks, LaMa erases them into a clean plate,
and SDXL inpaints each catalog item back in — conditioned on the product image
(IP-Adapter) and the item's own catalog fields. Slower and dearer (~25
predictions, 60–120s), but it is the only path that constrains placement with a
real mask.

**The image model never decides composition.** A `Placement` already carries
collision-checked floor coordinates; `geometry.py` fits a homography from the
photo's floor plane and projects them into image space. On the Replicate path
the renderer cuts the inpaint mask from that projection, so the model only
supplies appearance — materials, lighting, contact shadow. This is the whole
reason for doing it in this order: generative placement hallucinates furniture
into walls, and a solver that has already proven its layout is a better source
of truth.

A composing model takes no mask, so on the Gemini path the same geometry travels
as description — *"in the foreground, centred, spanning 45% of the room's
width; directly in front of the sofa, front legs on the rug"* — generated from
the solver's real coordinates and relationships. That is a strong constraint but
not a guarantee, which is why a composed render is capped one notch below a
masked one in `confidence`.

Calibration rides along with the room analysis: the vision prompt returns a
`floor_quad` and horizon alongside the dimensions. If the floor cannot be
traced, calibration is `None` and rendering declines rather than guessing —
the same instinct as withholding an exact-tier placement.

**A returned quad is checked before it is trusted.** Vision models get floor
orientation backwards often enough to matter, and an inverted quad still yields
a valid homography, so nothing downstream would catch it. A quad is discarded
unless its near edge sits lower in the frame than the far edge, and is no
narrower than it — the two ways a backwards quad renders furniture at the top
of the image, or mirrored.

The quad covers only the floor the camera saw, never the whole room, so
calibration carries `near_depth_cm`/`far_depth_cm` — the slice of depth it
spans. Fitting as though it covered everything slides near-wall furniture out
of frame. Visibility is therefore judged on a piece's **back edge**: a sofa
against the near wall has most of its floor cropped away yet dominates the
photo, and judging by footprint would drop the piece the user most wants to
see. The crop affects confidence instead, which is **capped by its weakest
input** — placement, room measurement and calibration alike.

Set `GEMINI_API_KEY` (or `REPLICATE_API_TOKEN`) for a real render. Without
either the endpoint returns **schematic** renders: the item's projected footprint and volume drawn
over the photo in its swatch colour. Not photorealistic and not pretending to
be, but it runs the identical projection maths, so a geometry bug shows up with
no key and no GPU.

> Every catalog item has a fetchable image, so both paths have something to
> condition on — but see **Mock data** below for what those images actually
> are. A piece with no fetchable image is still handled: on the Replicate path
> renders fall back to text conditioning; on the Gemini path it is **omitted
> and named in `omitted`**, since a composed render's whole value is fidelity
> to the real product.

### Precision tiers

Not every placement needs the same accuracy, so the solver does not pretend it
does. Each role carries a precision tier:

- **Approximate** (rug, coffee table, chair, lamp) — centred or free-floating.
  A rug 30cm off-centre is invisible. These are placed from a photo estimate.
- **Exact** (sofa) — must sit flush to a wall and clear of door swings. If the
  room is 40cm narrower than estimated, the piece does not fit.

**Exact-tier pieces are withheld, not guessed**, until the room is `measured`
(real dimensions supplied) and regular. Each withheld item reports what would
unlock it, and `clarification_needed` collects those into questions. Supplying
`room_width_cm` + `room_depth_cm` on any turn promotes the room and places them.
A coffee table cascades — it is positioned relative to the sofa, so it waits on
the same measurements rather than being stranded mid-floor.

Every placement also carries `confidence`, a `tolerance_cm` (how far it could
move before the layout reads as wrong) and a one-phrase `rationale`. A rug
reports ±40cm; a chair wedged into a corner reports ±5cm and drops to `medium`.
Wall contact is deliberately excluded from that slack calculation — a sofa flush
against its wall has a zero gap there by design, not by constraint.

Known door swings are hard exclusions for **every** piece, rugs included: a rug
under a door still stops it opening.

**The solver is deterministic** — no LLM produces coordinates. It anchors pieces
by design intent (sofa on the focal wall, table in front of it, chairs
flanking it, lamps in corners), then resolves collisions with inflated-AABB clearance tests.
Three invariants are enforced by `python -m backend.solver`:

1. every placement lies inside the room;
2. no two colliding pieces overlap;
3. an item that fits neither is **skipped with a reason, never forced** — and
   skipped items are excluded from the cart.

Rugs are deliberately exempt from collision: the plan is 2D but the room is 3D,
and furniture sits *on* a rug. Clearance degrades 40→30→20→10cm before an item
is dropped, because a strict 40cm rule is unsatisfiable in small rooms (a 90cm
sofa plus 40cm plus a 60cm table needs 190cm of a 200cm depth).

---

## Authorizing an agent to spend

The agent assembles a purchase across several merchants. The question that
raises is not whether the money moves — it never does — but **how a user grants
and withholds permission for it to move**. One invariant answers it:

> The agent may **price** a purchase. Only a human may **authorize** one.

Every endpoint below exists to keep those two acts separate.

```
GET  /api/payment/methods        stored cards (last-four only)
POST /api/payment/intent         agent prices it → preview. CHARGES NOTHING.
POST /api/payment/verify/start   issue a step-up challenge (OTP fallback)
POST /api/payment/verify         answer it
POST /api/payment/authorize      user releases the charge. The only endpoint
                                 that moves money.
POST /api/payment/cancel         user declines
GET  /api/payment/intent/{id}    re-read a preview

Visa Agentic Payments Stack
POST /api/passkey/register/options   begin FIDO2 enrolment
POST /api/passkey/register           verify + store the public key
GET  /api/passkey/credentials        registered passkeys (public data only)
POST /api/passkey/challenge          challenge bound to one intent AND amount
POST /api/passkey/verify             verify the assertion → single-use id
GET  /api/agent-token/defaults       suggested mandate limits
POST /api/agent-token/provision      mint a scoped AI_AGENT token
GET  /api/agent-token                tokens + spend history
POST /api/agent-token/revoke         kill the mandate. THE CARD KEEPS WORKING.
```

`POST /api/payment/intent` is the boundary of the agent's authority. It returns
a `PaymentIntent` that is fully priced and completely inert: holding one
confers no ability to charge anything. Reaching any later state requires a user
action the agent cannot synthesise.

### What the user sees before agreeing

A single "total" hides how many separate charges the user is authorizing, so
the preview breaks them out per merchant — **which shop, which items, which
card, shipping, tax, delivery window** — because a purchase you cannot itemise
is one you cannot meaningfully consent to. Prices are resolved server-side
against the live catalog rather than trusted from the client, so the total
shown cannot differ from the total charged.

> The rail is built for a multi-merchant basket. This catalog is entirely
> IKEA, so today every intent groups into one charge; the per-merchant
> machinery is what a second supplier would need.

### Why it asks you to verify

Every intent carries `risk[]`: plain-language signals explaining why this
purchase is or is not routine. Signals appear **every time, including when
nothing is wrong** — a warning that only shows on bad news trains people to
click past it.

| code | triggers step-up | meaning |
|---|---|---|
| `agent_initiated` | no | assembled by the agent, not typed by the user |
| `multi_merchant` | no | *n* separate statement lines, named |
| `routine` | no | within your usual limits and merchants |
| `token_presented` | no | paying via a scoped agent token, not a card number |
| `mandate_scoped` | no | inside the mandate, with remaining headroom named |
| `amount_over_threshold` | **yes** | over the per-card limit the user set |
| `new_merchant` | **yes** | first purchase from this shop |
| `over_budget` | **yes** | over the budget set for this room |
| `mandate_violation` | **blocks** | outside the mandate — verification cannot clear it |

### The Visa Agentic Payments Stack

Three layers sit on top of that rail. Each answers a question the rail alone
cannot, and each is enforced server-side — a client that skips the UI gets the
same refusals.

```
[ User Chat UI ]
       │ 1. Passkey / Face ID (VPP / FIDO2)
       ▼
[ Visa Token Service ] ──> scoped AI_AGENT token + signed mandate
       │
       ▼
[ AI Commerce Agent ] ──> bundles a multi-merchant order
       │ 2. Intent mandate & spend limits
       ▼
[ Visa Payment Rails ] ──> split settlement, one cryptogram per merchant
```

**Visa Payment Passkey (FIDO2)** — `backend/webauthn.py`. Step-up is a real
WebAuthn assertion, not an SMS code. The private key lives in the device's
secure enclave, the biometric never leaves the device, and the server verifies
an ES256/RS256 signature over the exact bytes the spec defines. What that buys
over an OTP:

- **Unphishable.** `clientDataJSON.origin` is checked against an allowlist, so
  a signature produced on a lookalike domain is rejected. A code can simply be
  read aloud to an attacker; a passkey cannot.
- **Transaction-bound.** The challenge record stores the intent *and* the
  amount, and the amount comes from the server's intent, never the client. A
  signature obtained for S$40 cannot authorize S$4,000.
- **Single-use, minutes-long.** An assertion is consumed on the charge it
  authorized. It is proof someone was present *now*, not a session token.
- **Clone detection.** The signature counter is checked for rollback.
- **UV required.** A credential registered without user verification is
  refused at enrolment rather than accepted and failed at checkout.

**Visa Token Service (`presentationType: AI_AGENT`)** — `backend/vts.py`. The
agent never sees a PAN. A card is enrolled once and the service issues a
network token whose last four digits deliberately differ from the card's, so a
statement line can be traced back to the agent that created it.

**Agent mandate.** The token carries a scope the agent holds as a signed
(JWS-shaped, HMAC-SHA256) bearer credential it cannot forge or widen:

| guardrail | enforced as |
|---|---|
| Category lock | MCC allowlist — furniture & home decor (5712/5713/5719/5200/5065). An unknown category **fails**, rather than passing |
| Per-purchase cap | checked against the *final* total, shipping and tax included |
| Cumulative cap | tracked across transactions, so splitting an order does not defeat the cap |
| Merchant allowlist | optional, tighter than the category lock |
| Merchant count | caps how many shops one order may bundle |
| Expiry | a standing permission expires on its own |
| Revocability | **revoking touches only the mandate — the card keeps working** |

Revocation is the property that makes delegation reversible, so the mandate is
checked **twice**: once at pricing, and again inside `authorize()`. A mandate
revoked in the seconds between reading the preview and pressing the button
stops the charge — a revocation that only took effect on the next purchase
would not be a revocation. Revoking deliberately requires no step-up: taking
authority away should always be easier than granting it.

A mandate violation is **not** a step-up. Verification cannot clear it, because
the user already said no in advance.

**Omitting the credential is not a bypass.** While a mandate is live, an intent
priced without one is still evaluated against it — otherwise every cap would be
advisory, defeated by an agent that simply left the header off. Once the mandate
is revoked or expires, checkout falls back to ordinary human-authorized payment:
the mandate constrains the *agent*, never the person.

**`require_user_presence` is enforced, not just displayed.** When a mandate sets
it — every mandate does by default — `authorize()` demands a consumed passkey
assertion or a completed step-up, *regardless of whether any risk signal fired*.
Without that, an agent-initiated order at a familiar merchant, under every
threshold and inside budget, would charge with no human in the loop while the UI
claimed otherwise. A guarantee shown to the user and not enforced at the rail is
worse than no guarantee.

**Assertions are purpose-bound.** A biometric performed to approve a *payment*
cannot be spent to *provision* a mandate, or vice versa. The two ceremonies ask
the user for an identical gesture but mean very different things, so the banked
proof records which one it was — otherwise the Face ID a user gave for a S$40
side table could mint a standing mandate they never agreed to.

**Caps are bounded by the deployment, not the caller.** `MAX_AGENT_MANDATE_CENTS`
(default S$10,000) is the ceiling on any mandate. The requested caps are
client-supplied, so without an absolute limit the "spend cap" would be whatever
the agent asked for.

**Cumulative headroom is reserved, not merely checked.** `authorize()` claims the
amount *before* the charge loop and releases whatever declines. Checking the cap
and committing afterwards leaves a window in which several intents priced against
the same budget each pass — precisely the split-the-order attack the cumulative
cap exists to stop. (Single-process; a multi-process deployment needs this
counter in shared storage with a real compare-and-set.)

**The recheck verifies the credential the agent presented.** The exact string is
persisted server-side (never on the `PaymentIntent`, which is serialized to the
client) and re-verified at authorization. Re-signing the mandate from current
server state would make the signature check a no-op — the server validating an
HMAC it computed a line earlier — and would silently pick up any later widening
of the in-memory scope instead of the one the user approved.

**Split settlement.** Each merchant leg gets its own single-use cryptogram,
bound to that leg's amount — so a cryptogram captured from one merchant cannot
be replayed against another on the same order. Only the *approved* portion is
recorded against the cumulative cap; counting a decline would be wrong twice
over.

### Safeguards on the authorization itself

- **No redirect.** The whole flow runs in one in-page sheet. The user is never
  handed to a merchant page, which is what lets the preview they read and the
  charge they approve be verifiably the same object.
- **Hold to confirm.** An 850ms deliberate press, not a click, so money cannot
  move on a stray tap. Enter/Space is equivalent — a gesture nobody can perform
  is a lockout, not a safeguard.
- **Total echo-back.** The client sends `confirmed_total_cents` from the screen
  the user actually read. If the intent has drifted, the charge is refused
  rather than billing a number nobody saw.
- **Idempotency keys.** A double-click, retry or flaky connection returns the
  original receipt instead of charging twice.
- **Previews expire** after 15 minutes; prices, stock and intent all go stale.
- **Partial failure is shown.** Per-merchant authorization means one charge can
  decline while others succeed, and the receipt says exactly which and for how
  much.
- **Audit trail.** Every receipt carries an `audit[]` narrating what was
  assembled, what was shown, what was verified and what was authorized, so an
  agent's spending is reviewable after the fact.

---

## Catalog data

The catalog is **228 real Singapore products from three merchants**, built at
import time. Names, dimensions, product photos and checkout links come from
the two scrapes; nothing in them is invented.

| Merchant | Source | Listings | Importer | Items |
| --- | --- | --- | --- | --- |
| IKEA SG | `products.json` (scrape) | 1,579 | `backend/ikea_import.py` | 175 |
| Castlery SG | `castlery_products.json` (scrape) | 80 | `backend/castlery_import.py` | 52 |
| YEN KAI | hand-entered, no feed | 1 | `backend/yenkai_import.py` | 1 |

```
               items   width        price
sofa            79    142-337cm   S$ 199-4098
accent_chair    87     45-130cm   S$ 19.90-1799
coffee_table    21     31-150cm   S$  6.90-1039
rug             25    180-244cm   S$ 13.90-809
floor_lamp      15     30-50cm    S$ 18.90-269
```

**Why a second merchant.** IKEA SG alone gave the catalog one price band —
it topped out near S$1,699, so a "premium" budget selected the same pieces as
a mid one. Castlery sits above it, which is what gives the budget logic a real
range to work across and retrieval a genuine choice between merchants. The
Castlery scrape is **sampled, not exhaustive**: 80 products drawn evenly
across the five roles, because Castlery's own catalog is 395 sofas to 9 floor
lamps and a flat slice would have returned almost no lamps. See
[`scrapers/README.md`](scrapers/README.md).

**YEN KAI is a local supplier with no website**, which is the third shape a
merchant takes: its catalog is a photograph and a measurement rather than a
feed. Its footprint (110×100cm) comes from the merchant and its photo is real,
but its **height is an estimate and its price is a placeholder**, both labelled
as such in `yenkai_import.py`. Having no CDN, its photo ships with the repo and
is emitted as a `data:` URI — a scheme the outbound image filter already
passes, so it needs no static route and no addition to the SSRF allowlist.

Item ids are merchant-prefixed (`ikea-…`, `castlery-…`, `yenkai-…`) so no two
merchants can collide on a shared SKU.

### What gets embedded

Items are embedded into Qdrant at startup by `CatalogIndex.seed()`, with the
payload carrying `role`, `price_cents`, `width_cm`, `depth_cm`, `in_stock`,
`merchant` and `style_tags` — the fields retrieval filters on, so an oversized
or over-budget piece is excluded before the vector search rather than after.

The embedded *text* is ordered most- to least-discriminating, because an
embedding weights the whole string and filler dilutes the signal:

- **The vendor's own description leads.** *"Light, airy design with high legs
  and slim lines"* is what someone asking for an airy room is actually
  searching for, and no amount of structured tagging reproduces it. It is
  present on all 1,579 listings, and its redundant repetition of the product
  name is stripped.
- **`materials` is capped at four and comes last.** IKEA publishes a full bill
  of materials, so a fabric sofa lists `polyurethane, fibreboard, plywood` —
  that describes the inside of the piece and matches nothing anyone would ask
  for.
- **Upholstery is carried separately** (`Gunnared light green`), parsed from
  the product title, since `materials` cannot distinguish the fabric on a
  piece from the foam inside it. Only upholstered roles get one; on a side
  table the same slot holds a size, which would be noise.

**Style tags are inferred, and deliberately sparse.** The scrape's own `style`
field is empty on every listing. A first pass put *"Scandinavian"* on **87%** of
the catalog — a tag carried by seven items in eight cannot discriminate. Each
rule now demands a specific signal and at most three tags survive, putting the
most common at 38%. Two rules where the naive version is wrong: leather counts
only when it is the *surface* (a fabric recliner lists leather for trim), and
metal reads industrial only when there is no wood at all.

**Prices are Singapore dollars.** `price_cents` is SGD cents and `currency`
says so on every item. The scrape carries no exchange rate, so nothing is
converted — inventing one would misstate every price in the app.

### What the IKEA scrape needed before it was usable

The raw 1,579 IKEA listings are not a catalog. Three problems, each handled by
`ikea_import.py`. (The Castlery scrape has different quirks — a dimension
string that has to be parsed, and colour that lives in the image filename —
documented in [`scrapers/README.md`](scrapers/README.md).)

**It is mostly irrelevant.** 146 categories, most of them towels, napkins,
plant pots and baby bedding. Only the ones that furnish a living room are
mapped, via an explicit allowlist rather than keyword matching — *"Bedside
tables & cabinets"* contains "table" but is not a coffee table, and *"Cabinet
lighting"* contains "cabinet" but is not furniture at all.

**It is wildly unbalanced.** 701 sofas against 8 floor lamps — but those 701
sofas are only 28 actual products, the rest colour and fabric variants. Left
alone, retrieval returns nine shades of the same UPPÅKRA and calls it choice.
So variants are capped at **3 per product family**, cheapest first, which keeps
a budget option in every family.

**Some dimensions are wrong in a way the solver cannot survive.** The scraper
recorded whichever measurement the page listed first, and the solver treats
dimensions as ground truth:

- *Lighting* got the **cord length** as its footprint — a "380cm-wide" floor
  lamp is a 3.8m cable. Detectable, because the scrape records which field it
  used, so anything sourced from `length` is rejected; lamps fall back to a
  plausible 40cm base.
- *Rugs* got the **length in the height field**, with depth copied from width:
  a 170×240cm rug arrives as 170 wide, 170 deep and **2.4m tall**. Every rug
  failed a plausibility check because of it. The real size is in the product
  name, so it is parsed back out from there.

Anything still out of band for its role after that is dropped rather than
guessed at — a wrong number here becomes a confidently wrong layout.

Modular sofa **components** (single-seat sections, armrests, corner units) are
also dropped. They are real products, but you cannot furnish a room with an
armrest.

### Product images

`image_url` points at IKEA's own CDN, and the images are used two ways: served
straight to the browser, and fetched server-side as appearance conditioning for
a render. Both work — all 174 resolve, `image/jpeg`, 23–689KB, and the CDN sets
permissive CORS with a 30-day cache and no hotlink protection.

Three things follow from these being **third-party URLs rather than local
files**, all handled in `_fetch_product_image`:

- **An explicit User-Agent is sent** — a default `Python-urllib/3.x` is the
  first thing a CDN rate-limits, and a silent block degrades every render with
  no obvious cause.
- **The host is checked against an allowlist** (`IMAGE_FETCH_ALLOWED_HOSTS`).
  `image_url` is data, not code, so without this anything that can write a
  catalog entry can make the server fetch an arbitrary URL, including one on a
  private network. The check is exact-or-subdomain.
- **A size ceiling is enforced** (`IMAGE_FETCH_MAX_BYTES`, default 8MB), read
  one byte past the cap so an oversized body is rejected rather than truncated
  into a corrupt image.

References for a composed render are fetched **concurrently**, in solver order
— the prompt refers to references by position, so shuffling would mislabel
every piece.

The real caveat is not technical: these are **hotlinks to third parties**.
Neither IKEA nor Castlery is affiliated with this project, both serve the
bandwidth, and either can change or remove any URL. A dead URL degrades gracefully (the piece is reported in
`omitted`), but anything beyond a demo should mirror the images.

### The missing role

The scrape contains **no TV units, media consoles or sideboards** — zero
matches. Rather than fill the gap with mock data, or with side tables
pretending to be consoles, the `tv_unit` role was **removed from the app**. The
catalog is now entirely real.

That costs one thing worth naming: `tv_unit` was an EXACT-precision role, so
the sofa is now the only piece exercising wall-hugging placement.

**The Castlery scrape changes what is possible here.** Castlery publishes
sideboards and media consoles, so the gap is no longer a data problem — it is
an unmapped role. Restoring it means re-adding `tv_unit` to `Role`,
`PLACEMENT_ORDER`, `ROLE_PRECISION` and `BUDGET_SHARE`, then mapping the
`Sideboards` and `TV Consoles` categories in
`castlery_import.CATEGORY_ROLE`. It is left unmapped here because adding a
role touches the solver's placement tiers, which is a change worth making
deliberately rather than as a side effect of a second scrape.

### Room fixtures

The four **rooms** in `backend/assets/rooms/` are still generated — drawn with
Pillow from known dimensions, no network and no image model, byte-identical on
every re-run. They are the only fixtures that can honestly report `measured`,
because their dimensions are not estimated from the image; they are the numbers
the image was *drawn from*. That makes them the only ones that exercise the
exact-precision tier.

---

## Verification

```bash
python -m backend.seed_data     # catalog integrity, price and size spread
python -m backend.solver        # layout invariants across 3 room sizes
python -m backend.geometry      # floor-plane projection invariants
python -m backend.render_engine # render pipeline, offline
python -m backend.agent         # full graph, offline
python -m backend.ikea_import     # catalog built from the IKEA scrape
python -m backend.castlery_import # catalog built from the Castlery scrape
python -m backend.yenkai_import   # the hand-entered YEN KAI catalog
python -m backend.mock_assets     # regenerate room fixtures
```

```bash
# -N disables curl buffering; without it you see nothing and conclude wrongly
curl -N -X POST http://localhost:8000/api/chat \
  -F "message=Design my living room" -F "budget=2500" \
  -F "aesthetic=Japandi" -F "image=@room.jpg"

# render that design into the photo (session_id from the `done` frame)
curl -N -X POST http://localhost:8000/api/render \
  -H 'Content-Type: application/json' -d '{"session_id":"s_..."}'

# refine it — neither budget nor aesthetic is resent; both carry forward
curl -N -X POST http://localhost:8000/api/chat \
  -F "message=make it cheaper" -F "session_id=s_..."

# explain a choice: answered from state, so no retrieval and no solve
curl -N -X POST http://localhost:8000/api/chat \
  -F "message=why is the sofa there?" -F "session_id=s_..."
```

> macOS has no `timeout`; use `curl --max-time`. And `-N` matters — without it
> curl buffers and you conclude the stream is broken when it is not.

---

Walk the payment rail end to end. Nothing here charges anything:

```bash
# 1. the agent prices a basket -> a preview, inert
curl -s localhost:8000/api/payment/intent -H 'content-type: application/json' \
  -d '{"item_ids":["ikea-59516727","ikea-50605257"]}' | jq '.id, .total_cents, .requires_step_up, .risk'

# 2. authorizing before verifying is refused
curl -s localhost:8000/api/payment/authorize -H 'content-type: application/json' \
  -d '{"intent_id":"<id>","idempotency_key":"k1","confirmed_total_cents":<total>}'
# {"detail":"identity verification required before authorizing"}

# 3. step up, then answer (demo_code is returned only because this is a demo)
curl -s localhost:8000/api/payment/verify/start -H 'content-type: application/json' -d '{"intent_id":"<id>"}'
curl -s localhost:8000/api/payment/verify -H 'content-type: application/json' \
  -d '{"intent_id":"<id>","challenge_id":"<cid>","code":"<demo_code>"}'

# 4. authorizing a total the user never saw is refused
curl -s localhost:8000/api/payment/authorize -H 'content-type: application/json' \
  -d '{"intent_id":"<id>","idempotency_key":"k2","confirmed_total_cents":999}'
# {"detail":"the total changed since you reviewed it ..."}

# 5. authorize for real, then replay the SAME key — one order, not two
curl -s localhost:8000/api/payment/authorize -H 'content-type: application/json' \
  -d '{"intent_id":"<id>","idempotency_key":"k3","confirmed_total_cents":<total>}' | jq '.order_id, .approved_cents, .declined_cents, .audit'
```

## Demo data disclaimer

**The catalog is real, and that cuts both ways.** Products, prices, dimensions,
photos and links are scraped IKEA Singapore listings, so they describe real
products — but they are **a snapshot, not a live feed**. Prices and stock go
stale, and IKEA is not affiliated with this project and has not endorsed it.
Anything sold on the strength of this data needs a live pricing and inventory
check first.

**Prices are SGD and are not converted.** A budget entered without a currency
in mind is being compared against Singapore dollars.

**Product images are hotlinked from IKEA's CDN.** This project serves none of
that bandwidth and has no agreement to use it; the URLs can change or vanish
without notice. Mirror them before relying on them.

**Some dimensions are repaired, not measured.** Lamp footprints fall back to a
plausible 40cm base where the scrape recorded a cord length, and rug sizes are
parsed out of product names. Both are documented in **Catalog data** above and
are honest for layout purposes, but they are not vendor spec sheets.

**Payment is entirely simulated.** The `/api/payment/*` rail models the shape
of a card transaction — intent, step-up challenge, per-merchant authorization,
receipt — but no HTTP request leaves `backend/payments.py`, no payment SDK is
installed, and no card number exists anywhere in this codebase. The stored
cards are fictional and carry only a last-four (`4242` is the universally
recognised test Visa). Auth codes are random hex. The one card ending `5454`
always declines, so the partial-failure path is demonstrable.

**The Visa Agentic Stack is simulated, but the cryptography is not.** No Visa
endpoint is contacted, nothing is enrolled with a real token service, and every
token, mandate and cryptogram is generated by this process. What *is* real is
the FIDO2 verification in `backend/webauthn.py`: real `navigator.credentials`
calls, a real Touch ID / Face ID prompt, and genuine ES256/RS256 signature
checking with origin binding, challenge single-use, UV enforcement and counter
rollback detection. Two honest limits: attestation is accepted without checking
it against the FIDO Metadata Service (normal for consumer passkeys, but it
means we know the key sits in *a* secure enclave without knowing whose), and
mandates are signed with a process-local HMAC key rather than in an HSM — so
this server could mint a mandate the user never approved, which a real
deployment prevents with asymmetric keys in tamper-resistant hardware. Both are
documented at the top of the modules that implement them.

Passkeys need a **secure context**: `localhost` or HTTPS. Set `WEBAUTHN_RP_ID`
and `WEBAUTHN_ORIGINS` when serving from anything else — the RP ID is a domain,
never derived from a request header, since a server that trusted the `Host`
header there would let an attacker nominate their own relying party.

**Renders are visualizations, not photographs.** A generative render is
conditioned on the product image but reconstructs the pixels, so it can differ
in real detail — grain, weave, hardware — from what would actually ship. Every
`RenderResult` carries `simulated: true` and a disclaimer string for this
reason, and a client should show it. Presenting a render as a photograph of the
product in the room would misrepresent what the customer is buying.
