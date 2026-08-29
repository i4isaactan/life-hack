# RoomCrafter AI — Backend

Agentic spatial commerce API. Send a room photo, a budget and an aesthetic;
get back a room analysis, furniture retrieved from a vector catalog, a
collision-free floor plan, and a bill of materials — streamed over SSE.

**Scope:** backend only. The chat client is assumed, so the deliverable is the
server plus the event contract documented below. `test_client.html` is a
throwaway harness, not the product UI.

---

## Quick start

```bash
uv venv --python 3.11
uv pip install -r backend/requirements.txt
.venv/bin/uvicorn backend.main:app --port 8000 --workers 1 --reload
```

Then open the test client:

```bash
python3 -m http.server 8080   # then visit http://localhost:8080/test_client.html
```

**No API key is required.** With `OPENAI_API_KEY` unset the app runs fully
offline: canned room analysis and deterministic local embeddings. Set the key
(in the environment or a `.env`) to use `gpt-4o` and `text-embedding-3-small`.
Both paths return identical schemas, so nothing downstream changes.

`GET /api/health` reports which providers are live.

> `--workers 1` matters: Qdrant runs in `:memory:` mode, so each worker would
> otherwise hold a separate catalog.

---

## SSE contract

`POST /api/chat` accepts `multipart/form-data` and returns `text/event-stream`.

| Field | Type | Notes |
|---|---|---|
| `message` | string | user prompt |
| `budget` | number | USD; defaults to 1500 |
| `aesthetic` | string | e.g. `Japandi`, `Industrial` |
| `session_id` | string | omit on the first turn; reuse the one `done` returns |
| `image` | file | optional, ≤8MB, downscaled to 1024px server-side |
| `room_width_cm` | number | actual measurement; promotes the room to *measured* |
| `room_depth_cm` | number | must be sent with `room_width_cm` |
| `openings` | JSON | `[{"kind":"door","wall":"north","offset_cm":20,"width_cm":85,"swing_cm":85}]` |
| `irregular` | bool | true for L-shaped rooms, alcoves, chimney breasts |

Measurements accumulate per session — give dimensions on one turn and door
positions on the next; both persist.

Frames are `event: <type>` / `data: <single-line JSON>` / blank line.

| Event | Payload | Meaning |
|---|---|---|
| `text_delta` | `{text}` | append to the assistant message |
| `room_analysis` | `{room}` | dimensions in cm, finishes, `source: openai\|mock` |
| `layout_update` | `{layout}` | `placements[]`, `skipped[]`, `withheld[]` |
| `clarification_needed` | `{questions[], withheld[]}` | measurements that would unlock withheld pieces |
| `cart_update` | `{cart, subtotal_cents, budget_cents, over_budget}` | bill of materials |
| `error` | `{message, code, fatal}` | failure inside the stream |
| `done` | `{session_id, elapsed_ms}` | always last |

```
event: layout_update
data: {"type":"layout_update","layout":{"room_width_cm":420,"room_depth_cm":330,
  "placements":[{"item_id":"sofa-linnea-2s","name":"Linnea 2-Seat Sofa","role":"sofa",
    "x_cm":128,"y_cm":237,"w_cm":164,"d_cm":86,"rotation":0,"z":1,
    "swatch":"#C9BCA6","price_cents":94900,"merchant":"Nordhaus"}],
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
frames — keep the remainder buffered. `test_client.html` is the reference
implementation; its parser is verified against byte-by-byte, mid-token, and
CRLF-split streams.

### Other endpoints

- `GET /api/health` — status and active providers
- `GET /api/catalog` — the seeded catalog, for debugging retrieval
- `POST /api/checkout/simulate` — `{item_ids[], session_id?}` → order grouped by merchant
- `POST /api/render` — visualize the design in the user's photo; see below

### `POST /api/render`

`{"session_id": "s_…", "item_ids": []}` → `text/event-stream`. Same framing as
`/api/chat`. Renders every placed item, one frame each as it finishes; an empty
`item_ids` means all of them.

| Event | Payload | Meaning |
|---|---|---|
| `render_started` | `{total, method, erased}` | `method` is `generative` or `schematic` |
| `render_update` | `RenderResult` + `{progress:{done,total}}` | one finished visualization |
| `render_failed` | `RenderFailure` | that item could not be rendered, and why |
| `done` | `{session_id, rendered, total, elapsed_ms}` | always last |

Call `/api/chat` first: rendering needs the photo, room and layout from a
completed design. An unknown session is `404`, a session with no design `409`.

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

## How it works

```
analyze_room → build_query → retrieve_items → select_items
             → solve_layout → build_cart → narrate
```

A linear LangGraph workflow. Nodes stream via `get_stream_writer()` with
`stream_mode="custom"` — chosen over `astream_events`, which on a graph this
small emits only generic `on_chain_*` frames with no domain meaning.

**Retrieval** filters by role, price, stock and *room dimensions*, so oversized
pieces never reach the solver. **Selection** buys in order of how much each
piece makes the room usable (seating first), reserving budget for seating so a
premium rug cannot price out the sofa.

### Replacing furniture in the photo

`/api/render` erases the furniture already in the room and puts the recommended
pieces where the solver put them:

```
detect → segment → erase          once per photo, cached as an empty-room plate
depth-condition → inpaint         per item, against that plate
```

Grounding DINO finds the existing furniture, SAM 2 cuts masks, LaMa erases them
into a clean plate, and SDXL inpaints each catalog item back in — conditioned on
the product image (IP-Adapter) and the item's own catalog fields.

**The image model never decides composition.** A `Placement` already carries
collision-checked floor coordinates; `geometry.py` fits a homography from the
photo's floor plane and projects them into image space. The renderer cuts the
inpaint mask from that projection, so the model only supplies appearance —
materials, lighting, contact shadow. This is the whole reason for doing it in
this order: generative placement hallucinates furniture into walls, and a solver
that has already proven its layout is a better source of truth.

Calibration rides along with the room analysis: the vision prompt returns a
`floor_quad` and horizon alongside the dimensions. If the floor cannot be traced
the calibration is `None` and rendering declines rather than guessing a
projection — the same instinct as withholding an exact-tier placement.

**Confidence is capped by its weakest input** — the placement's own confidence,
whether the room was measured, and the calibration's. A perfectly-placed sofa
projected through an estimated camera is still an estimate.

Set `REPLICATE_API_TOKEN` for the generative path. Without it the endpoint
returns **schematic** renders: the item's projected footprint and volume drawn
over the photo in its swatch colour. Not photorealistic and not pretending to
be, but it runs the identical projection maths, so a geometry bug shows up with
no key and no GPU.

> The seed catalog's `image_url`s point at a placeholder host, so appearance
> conditioning has nothing to fetch and renders fall back to text conditioning —
> a like-styled piece rather than that exact SKU. Real product imagery is what
> closes that gap.

### Precision tiers

Not every placement needs the same accuracy, so the solver does not pretend it
does. Each role carries a precision tier:

- **Approximate** (rug, coffee table, chair, lamp) — centred or free-floating.
  A rug 30cm off-centre is invisible. These are placed from a photo estimate.
- **Exact** (sofa, TV unit) — must sit flush to a wall and clear of door
  swings. If the room is 40cm narrower than estimated, the piece does not fit
  and the console fouls the door.

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
by design intent (sofa on the focal wall, table in front of it, TV opposite,
lamps in corners), then resolves collisions with inflated-AABB clearance tests.
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

## Verification

```bash
python -m backend.seed_data     # catalog integrity
python -m backend.solver        # layout invariants across 3 room sizes
python -m backend.geometry      # floor-plane projection invariants
python -m backend.render_engine # render pipeline, offline
python -m backend.agent         # full graph, offline
```

```bash
# -N disables curl buffering; without it you see nothing and conclude wrongly
curl -N -X POST http://localhost:8000/api/chat \
  -F "message=Design my living room" -F "budget=2500" \
  -F "aesthetic=Japandi" -F "image=@room.jpg"

# prove frames arrive incrementally rather than in one dump
curl -N -X POST http://localhost:8000/api/chat -F "message=hi" -F "budget=1500" \
  | while IFS= read -r l; do echo "$(date +%T) $l"; done

# render the design from a chat turn into the photo (session_id from `done`)
curl -N -X POST http://localhost:8000/api/render \
  -H 'Content-Type: application/json' -d '{"session_id":"s_..."}'
```

---

## Demo data disclaimer

**Merchants, products, prices, stock and URLs are fictional.** Nordhaus,
Cedarline, Muraya and Hesper & Co. do not exist, and the `*.invalid` links
resolve nowhere. No real retailer is named, because real pricing and inventory
cannot be verified here and attaching invented numbers to a real company would
misrepresent them.

**Checkout is entirely simulated.** `POST /api/checkout/simulate` fabricates an
order id and payment token, processes no payment, contacts no merchant, and
places no order. There is no payment SDK and no card entry anywhere in this
codebase.

**Renders are visualizations, not photographs.** A generative render is
conditioned on the product image but reconstructs the pixels, so it can differ
in real detail — grain, weave, hardware — from what would actually ship. Every
`RenderResult` carries `simulated: true` and a disclaimer string for this
reason, and a client should show it. Presenting a render as a photograph of the
product in the room would misrepresent what the customer is buying.
