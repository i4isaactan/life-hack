"""FastAPI server: multipart in, Server-Sent Events out.

The SSE contract is the product here, since the chat client is assumed rather
than built. It is documented in README.md and mirrored by test_client.html.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from . import config
from .agent import GraphState, build_graph
from .models import (
    Cart,
    CartLine,
    ChatMessage,
    CheckoutRequest,
    CheckoutResult,
    MerchantGroup,
    Opening,
    RenderFailure,
    RenderRequest,
    RenderResult,
    LayoutResult,
    RoomAnalysis,
)
from .rag_engine import CatalogIndex, VisionProvider, prepare_image
from .render_engine import RenderProvider, render_layout
from .seed_data import SEED_ITEMS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("roomcrafter")

# Populated in the lifespan handler. Seeding at import would run twice under
# uvicorn --reload and rebuild the in-memory collection needlessly.
STATE: dict[str, object] = {}

# In-process chat history. The client sends a session_id rather than replaying
# the transcript, which would mean re-uploading the room photo every turn.
SESSIONS: dict[str, list[ChatMessage]] = {}

# Room measurements accumulated per session. Kept separate from chat history
# because they are structured facts that refine across turns: a user may give
# dimensions on one turn and door positions on the next.
MEASUREMENTS: dict[str, dict[str, object]] = {}

# What /api/render needs from the chat turn that produced the design: the photo
# to paint into, the analysed room, and the solved layout. Rendering is a
# separate request because it takes tens of seconds per item and must not hold
# up the floor plan and cart, which arrive in under a second.
RENDER_CONTEXT: dict[str, dict[str, object]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    index = CatalogIndex()
    count = index.seed()
    vision = VisionProvider()
    renderer = RenderProvider()
    STATE["index"] = index
    STATE["vision"] = vision
    STATE["renderer"] = renderer
    STATE["graph"] = build_graph(index, vision)
    mode = "OpenAI" if config.HAS_OPENAI else "offline mock"
    log.info("catalog seeded with %d items | providers: %s", count, mode)
    log.info("renderer: %s (%s)", renderer.source, renderer.method.value)
    if not config.HAS_OPENAI:
        log.info("No OPENAI_API_KEY set - running fully offline with mock providers.")
    if not config.HAS_REPLICATE:
        log.info(
            "No REPLICATE_API_TOKEN set - /api/render returns schematic previews."
        )
    yield
    STATE.clear()
    RENDER_CONTEXT.clear()


app = FastAPI(title="RoomCrafter AI", version="1.0.0", lifespan=lifespan)

# Credentials stay off, which is what permits this explicit origin list.
# "null" covers test_client.html opened directly from disk over file://.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- SSE framing -----------------------------------------------------------


def sse_frame(event: str, data: dict) -> str:
    """Encode one SSE frame.

    Separators are compact and the JSON must stay on a single line: a newline
    inside a data: field silently truncates the frame at the client.
    """
    payload = json.dumps(data, separators=(",", ":"), default=str)
    return f"event: {event}\ndata: {payload}\n\n"


# --- Chat ------------------------------------------------------------------


@app.post("/api/chat")
async def chat(
    message: str = Form(""),
    budget: float | None = Form(None),
    aesthetic: str = Form(""),
    session_id: str | None = Form(None),
    image: UploadFile | None = File(None),
    # Measurements. Supplying width and depth promotes the room from an
    # estimate to measured, which unlocks wall-hugging pieces the solver
    # otherwise withholds rather than guessing at.
    room_width_cm: float | None = Form(None),
    room_depth_cm: float | None = Form(None),
    irregular: bool = Form(False),
    # JSON array of openings, e.g.
    # [{"kind":"door","wall":"north","offset_cm":10,"width_cm":80,"swing_cm":80}]
    openings: str | None = Form(None),
) -> StreamingResponse:
    graph = STATE.get("graph")
    if graph is None:  # pragma: no cover - only before lifespan completes
        raise HTTPException(503, "server still starting")

    sid = session_id or f"s_{secrets.token_hex(6)}"
    history = SESSIONS.setdefault(sid, [])

    # Read and downscale the upload before streaming starts: an oversized or
    # undecodable file should fail as a normal HTTP error, not mid-stream where
    # the status line has already been sent.
    image_b64: str | None = None
    if image is not None:
        raw = await image.read()
        if len(raw) > config.MAX_IMAGE_BYTES:
            raise HTTPException(
                413,
                f"image exceeds {config.MAX_IMAGE_BYTES // (1024 * 1024)}MB limit",
            )
        if raw:
            image_b64 = prepare_image(raw)
            if image_b64 is None:
                raise HTTPException(400, "could not decode image file")

    budget_cents = (
        int(budget * 100) if budget and budget > 0 else config.DEFAULT_BUDGET_CENTS
    )

    # Parse measurements up front so malformed input fails as a normal HTTP
    # error rather than mid-stream.
    parsed_openings: list[Opening] = []
    if openings:
        try:
            raw_openings = json.loads(openings)
            parsed_openings = [Opening(**o) for o in raw_openings]
        except Exception as exc:
            raise HTTPException(400, f"invalid openings: {exc}") from exc

    measurements: dict[str, object] | None = None
    if room_width_cm and room_depth_cm:
        if room_width_cm <= 0 or room_depth_cm <= 0:
            raise HTTPException(400, "room dimensions must be positive")
        measurements = {
            "width_cm": room_width_cm,
            "depth_cm": room_depth_cm,
            "measured": True,
            "irregular": irregular,
            "openings": parsed_openings,
        }
    elif parsed_openings or irregular:
        # Openings without dimensions still refine the room we estimated.
        measurements = {"irregular": irregular, "openings": parsed_openings}

    # Measurements persist for the session: having stated the room is 4.2×3.3m,
    # the user should not have to repeat it on every follow-up turn.
    if measurements:
        MEASUREMENTS[sid] = {**MEASUREMENTS.get(sid, {}), **measurements}

    history.append(ChatMessage(role="user", content=message))
    del history[: -config.MAX_HISTORY_MESSAGES]

    state: GraphState = {
        "messages": list(history),
        "image_b64": image_b64,
        "budget_cents": budget_cents,
        "aesthetic": aesthetic,
        "prompt": message,
        "measurements": MEASUREMENTS.get(sid) or None,
    }

    # The photo is kept for rendering. A later turn with no new photo reuses
    # the one already on the session, so a user can refine a design without
    # re-uploading it every message.
    ctx = RENDER_CONTEXT
    if image_b64:
        ctx.setdefault(sid, {})["image_b64"] = image_b64

    async def stream() -> AsyncIterator[str]:
        started = time.perf_counter()
        transcript: list[str] = []
        # Flush something immediately so proxies commit the response headers
        # and the client can distinguish "connected" from "still waiting".
        yield ": connected\n\n"

        queue: asyncio.Queue[tuple[str, dict] | None] = asyncio.Queue()

        async def run() -> None:
            try:
                async for chunk in graph.astream(state, stream_mode="custom"):
                    kind = chunk.get("type", "text_delta")
                    await queue.put((kind, chunk))
            except Exception as exc:
                log.exception("graph failed")
                # Headers are long gone by now, so an error must travel as a
                # frame; raising here would truncate the stream instead.
                await queue.put(
                    ("error", {"message": str(exc), "code": "graph_error", "fatal": True})
                )
            finally:
                await queue.put(None)

        task = asyncio.create_task(run())
        try:
            while True:
                try:
                    item = await asyncio.wait_for(
                        queue.get(), timeout=config.SSE_HEARTBEAT_SECONDS
                    )
                except asyncio.TimeoutError:
                    # Keeps idle intermediaries from dropping the connection.
                    yield ": ping\n\n"
                    continue

                if item is None:
                    break
                kind, payload = item
                if kind == "text_delta":
                    transcript.append(payload.get("text", ""))
                # Capture what /api/render will need. Taken from the frames
                # rather than the graph's final state because stream_mode
                # "custom" yields writer output, not state, and these are the
                # same objects the client is being told about.
                elif kind == "room_analysis":
                    ctx.setdefault(sid, {})["room"] = payload.get("room")
                elif kind == "layout_update":
                    ctx.setdefault(sid, {})["layout"] = payload.get("layout")
                yield sse_frame(kind, payload)
        finally:
            task.cancel()

        if transcript:
            history.append(ChatMessage(role="assistant", content="".join(transcript)))
            del history[: -config.MAX_HISTORY_MESSAGES]

        # `done` always fires, including after a fatal error, so a client can
        # unconditionally re-enable its composer when the stream ends.
        yield sse_frame(
            "done",
            {
                "session_id": sid,
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
            },
        )

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            # Tells nginx and friends not to buffer, which would otherwise
            # withhold every frame until the response completed.
            "X-Accel-Buffering": "no",
        },
    )


# --- Render ----------------------------------------------------------------


@app.post("/api/render")
async def render(req: RenderRequest) -> StreamingResponse:
    """Visualize the placed items in the user's own room photo.

    A separate endpoint rather than a graph node: a generative render is tens
    of seconds per item, and folding that into /api/chat would delay the floor
    plan and cart by minutes for a picture the user may not have asked for.

    Streams the same SSE framing as /api/chat, one `render_update` per item as
    it completes, in painter's order so a client can composite on arrival.
    """
    renderer = STATE.get("renderer")
    index = STATE.get("index")
    if renderer is None or index is None:  # pragma: no cover
        raise HTTPException(503, "server still starting")

    ctx = RENDER_CONTEXT.get(req.session_id)
    if not ctx:
        raise HTTPException(
            404,
            "unknown session; POST /api/chat first to produce a design to render",
        )

    room_raw, layout_raw = ctx.get("room"), ctx.get("layout")
    if not room_raw or not layout_raw:
        raise HTTPException(409, "this session has no completed design yet")

    # Rehydrate through the models so a malformed cache entry fails here as a
    # 4xx rather than deep inside the renderer mid-stream.
    try:
        room = RoomAnalysis(**room_raw)
        layout = LayoutResult(**layout_raw)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(500, f"corrupt session design: {exc}") from exc

    image_b64 = ctx.get("image_b64")
    items = {i.id: i for i in SEED_ITEMS}

    async def stream() -> AsyncIterator[str]:
        started = time.perf_counter()
        yield ": connected\n\n"

        total = len(
            [p for p in layout.placements if not req.item_ids or p.item_id in req.item_ids]
        )
        yield sse_frame(
            "render_started",
            {
                "total": total,
                "method": renderer.method.value,
                "erased": bool(image_b64) and renderer.method.value == "generative",
            },
        )

        queue: asyncio.Queue[tuple[str, dict] | None] = asyncio.Queue()

        async def run() -> None:
            try:
                async for result in render_layout(
                    renderer,
                    image_b64,
                    room,
                    layout,
                    items,
                    only=req.item_ids or None,
                ):
                    event = (
                        "render_update"
                        if isinstance(result, RenderResult)
                        else "render_failed"
                    )
                    await queue.put((event, result.model_dump(mode="json")))
            except Exception as exc:
                log.exception("render failed")
                await queue.put(
                    ("error", {"message": str(exc), "code": "render_error", "fatal": True})
                )
            finally:
                await queue.put(None)

        task = asyncio.create_task(run())
        done_count = 0
        try:
            while True:
                try:
                    item = await asyncio.wait_for(
                        queue.get(), timeout=config.SSE_HEARTBEAT_SECONDS
                    )
                except asyncio.TimeoutError:
                    # Renders are slow enough that heartbeats are load-bearing
                    # here, not just defensive as they are on /api/chat.
                    yield ": ping\n\n"
                    continue
                if item is None:
                    break
                kind, payload = item
                if kind == "render_update":
                    done_count += 1
                    payload["progress"] = {"done": done_count, "total": total}
                yield sse_frame(kind, payload)
        finally:
            task.cancel()

        yield sse_frame(
            "done",
            {
                "session_id": req.session_id,
                "rendered": done_count,
                "total": total,
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
            },
        )

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# --- Checkout (simulated) --------------------------------------------------


@app.post("/api/checkout/simulate", response_model=CheckoutResult)
async def checkout_simulate(req: CheckoutRequest) -> CheckoutResult:
    """Fabricate a multi-merchant confirmation. No payment is processed."""
    index = STATE.get("index")
    if index is None:  # pragma: no cover
        raise HTTPException(503, "server still starting")

    if not req.item_ids:
        raise HTTPException(400, "no items to check out")

    lines: list[CartLine] = []
    for item_id in req.item_ids:
        item = index.get(item_id)  # type: ignore[attr-defined]
        if item is None:
            raise HTTPException(404, f"unknown item: {item_id}")
        lines.append(
            CartLine(
                item_id=item.id,
                name=item.title,
                merchant=item.merchant,
                role=item.role,
                price_cents=item.price_cents,
                checkout_url=item.checkout_url,
                image_url=item.image_url,
            )
        )

    grouped: dict[str, list[CartLine]] = {}
    for line in lines:
        grouped.setdefault(line.merchant, []).append(line)

    groups = [
        MerchantGroup(
            merchant=merchant,
            lines=merchant_lines,
            subtotal_cents=sum(line.line_total_cents for line in merchant_lines),
        )
        for merchant, merchant_lines in sorted(grouped.items())
    ]

    return CheckoutResult(
        order_id=f"SIM-{secrets.token_hex(4).upper()}",
        payment_token=f"tok_sim_{secrets.token_hex(8)}",
        groups=groups,
        total_cents=sum(g.subtotal_cents for g in groups),
    )


# --- Health ----------------------------------------------------------------


@app.get("/api/health")
async def health() -> dict:
    index = STATE.get("index")
    renderer = STATE.get("renderer")
    return {
        "status": "ok" if index is not None else "starting",
        "providers": "openai" if config.HAS_OPENAI else "mock",
        "offline_mode": not config.HAS_OPENAI,
        "catalog_seeded": index is not None,
        "renderer": {
            "source": getattr(renderer, "source", "unavailable"),
            "method": getattr(getattr(renderer, "method", None), "value", "unavailable"),
        },
    }


@app.get("/api/catalog")
async def catalog(limit: int = 100) -> dict:
    """Inspect the seeded catalog. Useful when debugging retrieval."""
    from .seed_data import SEED_ITEMS

    return {
        "count": len(SEED_ITEMS),
        "items": [i.model_dump(mode="json") for i in SEED_ITEMS[:limit]],
    }
