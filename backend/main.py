"""FastAPI server: multipart in, Server-Sent Events out.

The SSE contract is the product here, since the chat client is assumed rather
than built. It is documented in README.md and mirrored by test_client.html.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import secrets
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from . import config
from .agent import GraphState, build_graph
from .models import (
    AgentTokenSummary,
    CatalogItem,
    Dimensions,
    MerchantBalance,
    MerchantCatalogPush,
    MerchantCatalogResult,
    MerchantOnboardRequest,
    MerchantOnboardResponse,
    MerchantSummary,
    MandateScope,
    PasskeyAssertionRequest,
    PasskeyCredentialSummary,
    PasskeyRegistrationRequest,
    ProvisionTokenRequest,
    RevokeMandateRequest,
    AuthorizationReceipt,
    Detection,
    Role,
    AuthorizeRequest,
    Cart,
    CartLine,
    ChatMessage,
    CheckoutRequest,
    CheckoutResult,
    DimensionSource,
    MerchantGroup,
    Opening,
    PaymentIntent,
    PaymentIntentRequest,
    PaymentIntentStatus,
    PaymentMethod,
    RenderFailure,
    RenderRequest,
    RenderResult,
    RoomRender,
    LayoutResult,
    RoomAnalysis,
    SwapRequest,
    VerificationChallenge,
    VerifyRequest,
)
from . import merchants
from . import payments
from . import visa_direct
from . import vts
from . import webauthn
from .rag_engine import (
    CatalogIndex,
    DetectionProvider,
    IntentProvider,
    VisionProvider,
    prepare_image,
)
from .render_engine import RenderProvider, render_layout, render_room
from .solver import LayoutSolver

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("roomhack")

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

# Conversational preferences that accumulate across turns: budget and aesthetic
# as last stated, items the user rejected, roles they dropped, size ceilings.
# Separate from MEASUREMENTS because these are taste, not facts about the room.
PREFERENCES: dict[str, dict[str, object]] = {}

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
    detector = DetectionProvider()
    # Constructed here rather than inside build_graph so its provider status is
    # observable on /api/health; the graph is handed the same instance.
    intent = IntentProvider()
    STATE["index"] = index
    STATE["vision"] = vision
    STATE["renderer"] = renderer
    STATE["intent"] = intent
    STATE["detector"] = detector
    STATE["graph"] = build_graph(index, vision, intent, detector)
    mode = "OpenAI" if config.HAS_OPENAI else "offline mock"
    log.info("catalog seeded with %d items | providers: %s", count, mode)
    # Per-subsystem, so a partial fallback is visible at boot rather than
    # showing up later as a mysteriously generic design.
    log.info(
        "vision: %s | intent: %s | embeddings: %s | detection: %s",
        vision.source,
        intent.source,
        index.embedder.source,
        detector.source,
    )
    if config.HAS_OPENAI:
        degraded = [
            name
            for name, provider in (
                ("vision", vision.source),
                ("intent", intent.source),
                ("embeddings", index.embedder.source),
            )
            if provider != "openai"
        ]
        if degraded:
            log.warning(
                "OPENAI_API_KEY is set but these fell back to offline: %s",
                ", ".join(degraded),
            )
    log.info(
        "renderer: %s | per-item: %s | compose: %s",
        renderer.source,
        renderer.method.value,
        config.GEMINI_IMAGE_MODEL if renderer.can_compose else "unavailable",
    )
    if not config.HAS_OPENAI:
        log.info("No OPENAI_API_KEY set - running fully offline with mock providers.")
    if not (config.HAS_GEMINI or config.HAS_REPLICATE):
        log.info(
            "No GEMINI_API_KEY or REPLICATE_API_TOKEN set - /api/render returns "
            "schematic previews."
        )
    yield
    STATE.clear()
    RENDER_CONTEXT.clear()


app = FastAPI(title="Room Hack", version="1.0.0", lifespan=lifespan)

# Credentials stay off, which is what permits this explicit origin list.
# "null" covers test_client.html opened directly from disk over file://.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Outbound image URLs ---------------------------------------------------
# Catalog image_urls are IKEA's own https:// product photos, which a browser
# loads directly, so nothing needs rewriting or re-hosting. This hook stays as
# the single place an outbound image_url is normalised, because renders arrive
# as data: URIs and any future local imagery would need mapping to a route
# rather than leaking a filesystem path to the client.


def _image_key(image_b64: str) -> str:
    """A short content hash, used to tell whether a photo is the same one.

    Hashing the base64 rather than storing it twice keeps the session dict
    small; a collision would only mean reusing detections for a different
    photo, which blake2b at this width makes irrelevant.
    """
    return hashlib.blake2b(image_b64.encode(), digest_size=16).hexdigest()


def _http_image_url(url: str) -> str:
    """An image_url the browser can load, or "" if it could not be made one."""
    if not isinstance(url, str):
        return ""
    if url.startswith(("http://", "https://", "data:", "/assets/")):
        return url
    # A file:// path would 404 in the browser and leak a local path in the
    # payload, so it is dropped rather than sent. Clients already handle a
    # missing image; they cannot handle a broken one.
    log.warning("dropping non-browser-loadable image_url: %s", url[:80])
    return ""


def _rewrite_image_urls(payload: object) -> object:
    """Recursively normalise every image_url in an outbound SSE/JSON payload."""
    if isinstance(payload, dict):
        return {
            key: (
                _http_image_url(value)
                if key == "image_url" and isinstance(value, str)
                else _rewrite_image_urls(value)
            )
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [_rewrite_image_urls(entry) for entry in payload]
    return payload


# --- SSE framing -----------------------------------------------------------


def sse_frame(event: str, data: dict) -> str:
    """Encode one SSE frame.

    Separators are compact and the JSON must stay on a single line: a newline
    inside a data: field silently truncates the frame at the client.
    """
    payload = json.dumps(
        _rewrite_image_urls(data), separators=(",", ":"), default=str
    )
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
    # "Your estimate looks right." Promotes the room to CONFIRMED, which is
    # enough to release wall-hugging pieces without demanding a tape measure.
    # Send alone to accept the proposal as-is, or with edited dimensions.
    confirm_dimensions: bool = Form(False),
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
        # confirm_dimensions marks "I looked at your estimate and it's about
        # right" - the same numbers, but now vouched for. Typed-in numbers
        # without that flag are a real measurement.
        measurements = {
            "width_cm": room_width_cm,
            "depth_cm": room_depth_cm,
            "dimension_source": (
                DimensionSource.CONFIRMED
                if confirm_dimensions
                else DimensionSource.MEASURED
            ),
            "irregular": irregular,
            "openings": parsed_openings,
        }
    elif confirm_dimensions:
        # Accepting the estimate without editing it: no numbers come back, so
        # promote whatever the last turn proposed.
        prior_dims = MEASUREMENTS.get(sid) or {}
        proposed = PREFERENCES.get(sid, {}).get("proposed_dimensions")
        if proposed:
            measurements = {
                **{k: v for k, v in proposed.items() if k in ("width_cm", "depth_cm")},
                "dimension_source": DimensionSource.CONFIRMED,
                "irregular": irregular or bool(prior_dims.get("irregular")),
                "openings": parsed_openings or prior_dims.get("openings") or [],
            }
        else:
            raise HTTPException(
                409, "nothing to confirm - no dimensions have been proposed yet"
            )
    elif parsed_openings or irregular:
        # Openings without dimensions still refine the room we estimated.
        measurements = {"irregular": irregular, "openings": parsed_openings}

    # Measurements persist for the session: having stated the room is 4.2×3.3m,
    # the user should not have to repeat it on every follow-up turn.
    if measurements:
        MEASUREMENTS[sid] = {**MEASUREMENTS.get(sid, {}), **measurements}

    history.append(ChatMessage(role="user", content=message))
    del history[: -config.MAX_HISTORY_MESSAGES]

    # Preferences carry forward. The form fields are authoritative only when
    # the caller actually sets them: a follow-up turn that omits budget or
    # aesthetic should keep what the conversation established, not silently
    # reset to the default.
    prefs = PREFERENCES.setdefault(sid, {})
    if budget is not None and budget > 0:
        prefs["budget_cents"] = budget_cents
    if aesthetic:
        prefs["aesthetic"] = aesthetic

    # The previous turn's design, so parse_intent can resolve "cheaper than
    # what" and "another one" rather than "which one".
    prior_ctx = RENDER_CONTEXT.get(sid) or {}
    prior_ids = prior_ctx.get("selected_ids") or []
    index = STATE.get("index")
    prior_items = [
        it
        for it in (index.get(i) for i in prior_ids)  # type: ignore[attr-defined]
        if it is not None
    ]
    prior_layout_raw = prior_ctx.get("layout")
    prior_layout = (
        LayoutResult(**prior_layout_raw)
        if isinstance(prior_layout_raw, dict)
        else prior_layout_raw
    )

    # A turn with no new upload reuses the photo already on the session, so a
    # follow-up like "make it cheaper" still analyses the user's real room.
    # Without this the graph re-runs with no image at all and the room silently
    # reverts to default dimensions, losing the detections and existing-style
    # signal the first turn paid for.
    if image_b64 is None:
        stored = prior_ctx.get("image_b64")
        if isinstance(stored, str) and stored:
            image_b64 = stored

    # Detections cached from a previous turn on this session, reusable only
    # while the photo they describe is still the current one. Detection is a
    # second vision call, and a follow-up like "make it cheaper" re-runs the
    # whole graph over an unchanged photo - paying for it again buys nothing.
    cached_detections = None
    if image_b64 and prior_ctx.get("detected_for") == _image_key(image_b64):
        raw_dets = prior_ctx.get("detections")
        if isinstance(raw_dets, list):
            cached_detections = [Detection(**d) for d in raw_dets]

    state: GraphState = {
        "messages": list(history),
        "image_b64": image_b64,
        "detections": cached_detections,
        "budget_cents": prefs.get("budget_cents", budget_cents),
        "aesthetic": prefs.get("aesthetic", aesthetic),
        "prompt": message,
        "measurements": MEASUREMENTS.get(sid) or None,
        "selected": prior_items,
        "layout": prior_layout,
        "rejected_ids": list(prefs.get("rejected_ids") or []),
        "excluded_roles": list(prefs.get("excluded_roles") or []),
        "role_counts": dict(prefs.get("role_counts") or {}),
        "size_caps": dict(prefs.get("size_caps") or {}),
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
                    room_payload = payload.get("room") or {}
                    ctx.setdefault(sid, {})["room"] = room_payload
                    # Cache the detections against the photo they came from, so
                    # a follow-up turn over the same image skips the second
                    # vision call. Keyed by content, not by session, because a
                    # new photo must invalidate them.
                    if image_b64:
                        ctx[sid]["detections"] = room_payload.get("detections") or []
                        ctx[sid]["detected_for"] = _image_key(image_b64)
                elif kind == "layout_update":
                    ctx.setdefault(sid, {})["layout"] = payload.get("layout")
                elif kind == "alternatives":
                    # Kept so /api/swap can validate a chosen item against what
                    # was actually offered, rather than trusting a client id.
                    ctx.setdefault(sid, {})["options"] = payload.get("options")
                elif kind == "cart_update":
                    ctx.setdefault(sid, {})["selected_ids"] = [
                        line["item_id"] for line in payload.get("cart", {}).get("lines", [])
                    ]
                    ctx.setdefault(sid, {})["budget_cents"] = payload.get("budget_cents")
                elif kind == "dimension_proposal":
                    # Remember what we offered, so a bare confirm_dimensions on
                    # the next turn can promote these numbers without the client
                    # having to echo them back.
                    prefs["proposed_dimensions"] = payload.get("proposal") or {}
                elif kind == "intent":
                    # Persist what the turn established so the next one inherits
                    # it: a rejected sofa stays rejected, a lowered budget stays
                    # lowered, without the client having to resend either.
                    parsed = payload.get("intent") or {}
                    if parsed.get("budget_cents"):
                        prefs["budget_cents"] = parsed["budget_cents"]
                    if parsed.get("aesthetic"):
                        prefs["aesthetic"] = parsed["aesthetic"]
                    if parsed.get("reject_item_ids"):
                        prefs["rejected_ids"] = sorted(
                            {*(prefs.get("rejected_ids") or []), *parsed["reject_item_ids"]}
                        )
                    if parsed.get("remove_roles"):
                        prefs["excluded_roles"] = sorted(
                            {*(prefs.get("excluded_roles") or []), *parsed["remove_roles"]}
                        )
                    if parsed.get("role_counts"):
                        counts = {
                            k: max(0, int(v))
                            for k, v in parsed["role_counts"].items()
                        }
                        prefs["role_counts"] = {
                            **(prefs.get("role_counts") or {}),
                            **counts,
                        }
                        # Kept in step with the graph's own reading of a count:
                        # zero excludes the role, and any positive number
                        # un-excludes it, so "no chairs" then "two chairs"
                        # does not leave a stale exclusion behind.
                        excluded = set(prefs.get("excluded_roles") or [])
                        excluded |= {k for k, v in counts.items() if v <= 0}
                        excluded -= {k for k, v in counts.items() if v > 0}
                        prefs["excluded_roles"] = sorted(excluded)
                    if parsed.get("max_width_cm"):
                        prefs["size_caps"] = {
                            **(prefs.get("size_caps") or {}),
                            **parsed["max_width_cm"],
                        }
                    # A reroll rejects the item currently in that role, so the
                    # next search cannot return the same piece.
                    if parsed.get("reroll_roles"):
                        rolled = {
                            it.id
                            for it in prior_items
                            if it.role.value in parsed["reroll_roles"]
                        }
                        if rolled:
                            prefs["rejected_ids"] = sorted(
                                {*(prefs.get("rejected_ids") or []), *rolled}
                            )
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
    # The live index, not SEED_ITEMS: a merchant-published product is in the
    # catalog and can be chosen into a design, but exists nowhere in the seed
    # list. Looking it up there renders the design without it, silently.
    items = {i.id: i for i in index.all_items()}  # type: ignore[attr-defined]

    # A composing backend renders the whole room in one call, which is both
    # cheaper and more coherent than six masked inpaints - and every piece
    # comes from its real product photo. Fall back to per-item when there is
    # no composer, or when the caller explicitly asks for it.
    compose = renderer.can_compose and not req.per_item

    async def stream() -> AsyncIterator[str]:
        started = time.perf_counter()
        yield ": connected\n\n"

        total = (
            1
            if compose
            else len(
                [
                    p
                    for p in layout.placements
                    if not req.item_ids or p.item_id in req.item_ids
                ]
            )
        )
        yield sse_frame(
            "render_started",
            {
                "total": total,
                "method": "composed" if compose else renderer.method.value,
                "erased": bool(image_b64) and renderer.method.value == "generative",
            },
        )

        if compose:
            result = await render_room(
                renderer,
                image_b64,
                room,
                layout,
                items,
                only=req.item_ids or None,
            )
            ok = isinstance(result, RoomRender)
            payload = result.model_dump(mode="json")
            if ok:
                payload["progress"] = {"done": 1, "total": 1}
            yield sse_frame("room_render" if ok else "render_failed", payload)
            yield sse_frame(
                "done",
                {
                    "session_id": req.session_id,
                    "rendered": 1 if ok else 0,
                    "total": 1,
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                },
            )
            return

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


# --- Swap ------------------------------------------------------------------


@app.post("/api/swap")
async def swap(req: SwapRequest) -> StreamingResponse:
    """Replace one item with an alternative, re-solve, and re-render.

    A swap is never merely a re-render. A different sofa has different
    dimensions, so the layout must be re-solved - the replacement may not fit,
    or may displace the coffee table that was positioned against the old one -
    and the cart re-billed. Returning a new picture over a stale plan would
    show the user something the solver never agreed to.
    """
    renderer = STATE.get("renderer")
    index = STATE.get("index")
    if renderer is None or index is None:  # pragma: no cover
        raise HTTPException(503, "server still starting")

    ctx = RENDER_CONTEXT.get(req.session_id)
    if not ctx:
        raise HTTPException(404, "unknown session; POST /api/chat first")

    room_raw = ctx.get("room")
    selected_ids = ctx.get("selected_ids")
    if not room_raw or selected_ids is None:
        raise HTTPException(409, "this session has no completed design yet")

    replacement = index.get(req.item_id)  # type: ignore[attr-defined]
    if replacement is None:
        raise HTTPException(404, f"unknown item: {req.item_id}")
    if replacement.role is not req.role:
        raise HTTPException(
            400,
            f"{req.item_id} is a {replacement.role.value}, not a {req.role.value}",
        )

    try:
        room = RoomAnalysis(**room_raw)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(500, f"corrupt session design: {exc}") from exc

    # Substitute by role. The outgoing item leaves the design entirely, so a
    # swap never grows the piece count - it exchanges one for one.
    #
    # Resolved against the live index rather than SEED_ITEMS. A merchant-
    # published product is orderable and can be in the design, but is absent
    # from the seed list - and because the comprehension below skips ids it
    # cannot resolve, looking it up there dropped that piece from the design
    # and the cart silently, rather than failing.
    items = {i.id: i for i in index.all_items()}  # type: ignore[attr-defined]
    current = [items[i] for i in selected_ids if i in items]
    outgoing = next((i for i in current if i.role is req.role), None)
    if outgoing is None:
        raise HTTPException(
            409, f"nothing of role {req.role.value} is currently in the design"
        )
    if outgoing.id == replacement.id:
        raise HTTPException(400, "that item is already selected")

    updated = [replacement if i.role is req.role else i for i in current]

    budget_cents = int(ctx.get("budget_cents") or config.DEFAULT_BUDGET_CENTS)
    image_b64 = ctx.get("image_b64")

    async def stream() -> AsyncIterator[str]:
        started = time.perf_counter()
        yield ": connected\n\n"
        yield sse_frame(
            "swap_started",
            {
                "role": req.role.value,
                "from": {"item_id": outgoing.id, "name": outgoing.title},
                "to": {"item_id": replacement.id, "name": replacement.title},
            },
        )

        # Re-solve. The replacement's footprint differs, so positions move and
        # a piece that no longer fits is reported rather than forced.
        layout = LayoutSolver(room).solve(updated)
        yield sse_frame(
            "layout_update", {"type": "layout_update", "layout": layout.model_dump(mode="json")}
        )

        placed_ids = {p.item_id for p in layout.placements}
        lines = [
            CartLine(
                item_id=i.id,
                name=i.title,
                merchant=i.merchant,
                role=i.role,
                price_cents=i.price_cents,
                checkout_url=i.checkout_url,
                image_url=i.image_url,
            )
            for i in updated
            if i.id in placed_ids
        ]
        cart = Cart(
            lines=lines,
            subtotal_cents=sum(line.line_total_cents for line in lines),
            budget_cents=budget_cents,
            currency=updated[0].currency if updated else Cart.model_fields["currency"].default,
        )
        yield sse_frame(
            "cart_update",
            {
                "type": "cart_update",
                "cart": cart.model_dump(mode="json"),
                "subtotal_cents": cart.subtotal_cents,
                "budget_cents": cart.budget_cents,
                "over_budget": cart.over_budget,
            },
        )

        # The swap is committed to the session before rendering, so a failed or
        # skipped render still leaves the design in its new state.
        ctx["selected_ids"] = [i.id for i in updated]
        ctx["layout"] = layout.model_dump(mode="json")

        # Offer the alternatives again, now priced against the new pick.
        alternatives = _reprice_options(
            ctx.get("options"),
            updated,
            budget_cents,
            cart,
            index.get,  # type: ignore[attr-defined]
        )
        if alternatives:
            yield sse_frame("alternatives", {"options": alternatives})

        if not req.layout_only:
            result = await render_room(renderer, image_b64, room, layout, items)
            event = "room_render" if isinstance(result, RoomRender) else "render_failed"
            yield sse_frame(event, result.model_dump(mode="json"))

        yield sse_frame(
            "done",
            {
                "session_id": req.session_id,
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


def _reprice_options(
    stored: object,
    selected: list,
    budget_cents: int,
    cart: Cart,
    lookup: Callable[[str], CatalogItem | None],
) -> list[dict]:
    """Re-point the stored alternatives at the new selection.

    Deltas and affordability were computed against the item that just left the
    design, so leaving them untouched would price every option against a piece
    the user no longer has.

    `lookup` resolves the outgoing item so it can be offered back as an
    alternative. It is passed in rather than read from SEED_ITEMS because a
    merchant-published piece is not in the seed list, and a user who swaps one
    away needs the same route back that any other piece gets.
    """
    if not isinstance(stored, list):
        return []

    by_role = {i.role.value: i for i in selected}
    spent = cart.subtotal_cents
    out: list[dict] = []
    for entry in stored:
        if not isinstance(entry, dict):
            continue
        chosen = by_role.get(entry.get("role"))
        if chosen is None:
            continue
        headroom = budget_cents - spent + chosen.price_cents
        alts = [
            {
                **alt,
                "price_delta_cents": alt["price_cents"] - chosen.price_cents,
                "affordable": alt["price_cents"] <= headroom,
            }
            for alt in entry.get("alternatives", [])
            if alt.get("item_id") != chosen.id
        ]
        # The outgoing item becomes an alternative itself - a user who dislikes
        # the swap needs a way back without replaying the conversation. It is
        # listed first, because undo is the most likely next click.
        outgoing_id = entry.get("selected_id")
        if outgoing_id and outgoing_id != chosen.id:
            if not any(a["item_id"] == outgoing_id for a in alts):
                previous = lookup(outgoing_id)
                if previous is not None:
                    alts.insert(
                        0,
                        {
                            "item_id": previous.id,
                            "name": previous.title,
                            "merchant": previous.merchant,
                            "role": previous.role.value,
                            "price_cents": previous.price_cents,
                            "price_delta_cents": previous.price_cents - chosen.price_cents,
                            "swatch": previous.swatch,
                            "image_url": previous.image_url,
                            "materials": previous.materials,
                            "primary_color": previous.primary_color,
                            "style_tags": previous.style_tags,
                            "width_cm": previous.dimensions.width_cm,
                            "depth_cm": previous.dimensions.depth_cm,
                            "height_cm": previous.dimensions.height_cm,
                            "affordable": previous.price_cents <= headroom,
                        },
                    )
        entry["selected_id"] = chosen.id
        entry["alternatives"] = alts
        out.append(
            {"role": entry["role"], "selected_id": chosen.id, "alternatives": alts}
        )
    return out


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
                # Browser-loadable; this response is rendered directly by the
                # client rather than passing through sse_frame.
                image_url=_http_image_url(item.image_url),
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


# --- Payment (simulated Visa rail) -----------------------------------------
#
# The authorization model, in one line: the agent may PRICE a purchase; only
# the user may AUTHORIZE one. Everything below enforces that split.
#
#   GET  /api/payment/methods    the user's stored cards
#   POST /api/payment/intent     agent prices it -> preview, charges nothing
#   POST /api/payment/verify/start   issue a step-up challenge
#   POST /api/payment/verify     answer it
#   POST /api/payment/authorize  the user releases the charge
#   POST /api/payment/cancel     the user declines
#   GET  /api/payment/intent/{id}  re-read a preview


def _payment_error(exc: payments.PaymentError) -> HTTPException:
    return HTTPException(exc.status, exc.detail)


@app.get("/api/payment/methods", response_model=list[PaymentMethod])
async def payment_methods() -> list[PaymentMethod]:
    """The cards on file. Fictional; only last four digits ever exist."""
    return payments.wallet()


@app.post("/api/payment/intent", response_model=PaymentIntent)
async def payment_intent(req: PaymentIntentRequest) -> PaymentIntent:
    """Price a purchase into a preview. NOTHING IS CHARGED HERE.

    This is the boundary of the agent's authority. It resolves every item
    against the live catalog rather than trusting prices the client sends,
    because a client-supplied total is a total the user could be shown while a
    different one is charged.
    """
    index = STATE.get("index")
    if index is None:  # pragma: no cover
        raise HTTPException(503, "server still starting")
    if not req.item_ids:
        raise HTTPException(400, "no items to pay for")

    lines: list[CartLine] = []
    for item_id in req.item_ids:
        item = index.get(item_id)  # type: ignore[attr-defined]
        if item is None:
            raise HTTPException(404, f"unknown item: {item_id}")
        if not item.in_stock:
            raise HTTPException(409, f"out of stock: {item.title}")
        lines.append(
            CartLine(
                item_id=item.id,
                name=item.title,
                merchant=item.merchant,
                role=item.role,
                price_cents=item.price_cents,
                checkout_url=item.checkout_url,
                image_url=_http_image_url(item.image_url),
            )
        )

    # The budget the design was actually solved against, so the preview can say
    # whether this purchase honours the constraint the user set.
    budget_cents = 0
    if req.session_id:
        prefs = PREFERENCES.get(req.session_id) or {}
        budget_cents = int(prefs.get("budget_cents") or 0)

    try:
        return payments.create_intent(
            lines,
            session_id=req.session_id,
            budget_cents=budget_cents,
            payment_method_ids=req.payment_method_ids,
            initiated_by="agent",
            mandate_credential=req.mandate_credential,
        )
    except payments.PaymentError as exc:
        raise _payment_error(exc) from exc


@app.get("/api/payment/intent/{intent_id}", response_model=PaymentIntent)
async def payment_intent_read(intent_id: str) -> PaymentIntent:
    try:
        return payments.get_intent(intent_id)
    except payments.PaymentError as exc:
        raise _payment_error(exc) from exc


@app.post("/api/payment/verify/start", response_model=VerificationChallenge)
async def payment_verify_start(body: dict) -> VerificationChallenge:
    """Issue a step-up challenge, in the shape of a 3-D Secure prompt."""
    intent_id = str(body.get("intent_id") or "")
    if not intent_id:
        raise HTTPException(400, "intent_id required")
    try:
        return payments.start_verification(intent_id)
    except payments.PaymentError as exc:
        raise _payment_error(exc) from exc


@app.post("/api/payment/verify", response_model=PaymentIntent)
async def payment_verify(req: VerifyRequest) -> PaymentIntent:
    """Answer a step-up challenge. Proves a human is present."""
    try:
        return payments.verify(req.intent_id, req.challenge_id, req.code)
    except payments.PaymentError as exc:
        raise _payment_error(exc) from exc


@app.post("/api/payment/authorize", response_model=AuthorizationReceipt)
async def payment_authorize(req: AuthorizeRequest) -> AuthorizationReceipt:
    """The user releases the charge. The only endpoint that moves money.

    Refuses if the total has drifted from what the user reviewed, if step-up
    was required and not completed, or if the preview has expired. Replaying
    an idempotency key returns the original receipt rather than charging twice.
    """
    if not req.idempotency_key:
        raise HTTPException(400, "idempotency_key required")
    try:
        return payments.authorize(
            req.intent_id,
            req.idempotency_key,
            req.confirmed_total_cents,
            assertion_id=req.assertion_id,
        )
    except payments.PaymentError as exc:
        raise _payment_error(exc) from exc


@app.post("/api/payment/cancel", response_model=PaymentIntent)
async def payment_cancel(body: dict) -> PaymentIntent:
    """Decline the purchase. Always available while nothing is charged."""
    intent_id = str(body.get("intent_id") or "")
    if not intent_id:
        raise HTTPException(400, "intent_id required")
    try:
        return payments.cancel_intent(intent_id)
    except payments.PaymentError as exc:
        raise _payment_error(exc) from exc


# --- Visa Agentic Payments Stack -------------------------------------------
#
# Three groups of endpoints, one per layer of the stack:
#
#   /api/passkey/*   FIDO2 registration and assertion. Proves the cardholder
#                    is physically present, via a device biometric bound to a
#                    hardware-held key. No biometric data ever reaches here.
#   /api/agent-token/*  Visa Token Service. Enrolls a card and mints a scoped
#                    AI_AGENT token so the agent spends against a revocable
#                    token rather than a card number.
#   revoke           The kill switch. Ends the agent's authority without
#                    touching the underlying card.


def _webauthn_error(exc: webauthn.WebAuthnError) -> HTTPException:
    return HTTPException(exc.status, exc.detail)


def _mandate_error(exc: vts.MandateViolation) -> HTTPException:
    return HTTPException(exc.status, exc.detail)


def _credential_summary(c: webauthn.StoredCredential) -> PasskeyCredentialSummary:
    return PasskeyCredentialSummary(
        credential_id=c.credential_id,
        label=c.label,
        created_at=c.created_at,
        sign_count=c.sign_count,
        backed_up=c.backed_up,
        transports=c.transports,
    )


@app.get("/api/passkey/credentials", response_model=list[PasskeyCredentialSummary])
async def passkey_credentials() -> list[PasskeyCredentialSummary]:
    """Passkeys registered on this account. Public metadata only."""
    return [_credential_summary(c) for c in webauthn.list_credentials()]


@app.post("/api/passkey/register/options")
async def passkey_register_options(body: dict | None = None) -> dict:
    """Options for navigator.credentials.create().

    The challenge is minted server-side and single-use, so the browser cannot
    choose what it signs over.
    """
    body = body or {}
    return webauthn.registration_options(
        rp_id=config.WEBAUTHN_RP_ID,
        rp_name=config.WEBAUTHN_RP_NAME,
        user_id=str(body.get("user_id") or "roomhack-demo-user"),
        user_name=str(body.get("user_name") or "demo@roomhack.local"),
        user_display=str(body.get("user_display") or "Room Hack demo"),
    )


@app.post("/api/passkey/register", response_model=PasskeyCredentialSummary)
async def passkey_register(req: PasskeyRegistrationRequest) -> PasskeyCredentialSummary:
    """Verify a create() response and store the public key."""
    try:
        credential = webauthn.verify_registration(
            credential_id=req.credential_id,
            client_data_json_b64=req.client_data_json,
            attestation_object_b64=req.attestation_object,
            transports=req.transports,
            label=req.label,
            rp_id=config.WEBAUTHN_RP_ID,
            allowed_origins=config.WEBAUTHN_ORIGINS,
        )
    except webauthn.WebAuthnError as exc:
        raise _webauthn_error(exc) from exc
    return _credential_summary(credential)


@app.delete("/api/passkey/credentials/{credential_id}")
async def passkey_delete(credential_id: str) -> dict:
    """Remove a passkey. Any agent token it provisioned keeps its own mandate,
    which is revoked separately - the two are deliberately independent."""
    if not webauthn.delete_credential(credential_id):
        raise HTTPException(404, "unknown passkey")
    return {"deleted": credential_id}


@app.post("/api/passkey/challenge")
async def passkey_challenge(body: dict) -> dict:
    """Options for navigator.credentials.get(), bound to one transaction.

    The amount is baked into the challenge record server-side, so a signature
    obtained for one total cannot authorize a different one. Passing the
    amount here is not the client asserting what it will pay - the intent
    already fixed that - it is the client naming which payment it is asking
    the user to approve.
    """
    purpose = str(body.get("purpose") or "payment")
    intent_id = body.get("intent_id")
    amount_cents = 0

    if purpose == "payment":
        if not intent_id:
            raise HTTPException(400, "intent_id required for a payment challenge")
        try:
            intent = payments.get_intent(str(intent_id))
        except payments.PaymentError as exc:
            raise _payment_error(exc) from exc
        # Bind to the SERVER's total, never a client-supplied one. Otherwise a
        # client could request a challenge for a small amount, have the user
        # approve that, and present the signature against a larger intent.
        amount_cents = intent.total_cents

    try:
        return webauthn.authentication_options(
            rp_id=config.WEBAUTHN_RP_ID,
            intent_id=str(intent_id) if intent_id else None,
            amount_cents=amount_cents,
            purpose=purpose,
        )
    except webauthn.WebAuthnError as exc:
        raise _webauthn_error(exc) from exc


@app.post("/api/passkey/verify")
async def passkey_verify(req: PasskeyAssertionRequest) -> dict:
    """Verify a get() response - the moment "it is really them" is proven.

    Returns an assertion id, not a session: it is single-use, expires in
    minutes, and is bound to one intent and one amount.
    """
    amount_cents = 0
    if req.purpose == "payment":
        if not req.intent_id:
            raise HTTPException(400, "intent_id required")
        try:
            intent = payments.get_intent(req.intent_id)
        except payments.PaymentError as exc:
            raise _payment_error(exc) from exc
        amount_cents = intent.total_cents

    try:
        result = webauthn.verify_assertion(
            credential_id=req.credential_id,
            client_data_json_b64=req.client_data_json,
            authenticator_data_b64=req.authenticator_data,
            signature_b64=req.signature,
            rp_id=config.WEBAUTHN_RP_ID,
            allowed_origins=config.WEBAUTHN_ORIGINS,
            purpose=req.purpose,
            intent_id=req.intent_id,
            amount_cents=amount_cents,
        )
    except webauthn.WebAuthnError as exc:
        raise _webauthn_error(exc) from exc

    if req.purpose == "payment" and req.intent_id:
        payments.record_assertion(result, req.intent_id, amount_cents, purpose="payment")
        # Clear the step-up so the intent is confirmable. The passkey is the
        # stronger factor, so it satisfies a requirement an OTP would have.
        try:
            intent = payments.get_intent(req.intent_id)
            if intent.status == PaymentIntentStatus.REQUIRES_VERIFICATION:
                intent.status = PaymentIntentStatus.REQUIRES_CONFIRMATION
        except payments.PaymentError:  # pragma: no cover - defensive
            pass
    else:
        # A provisioning assertion is banked too, so token minting can demand
        # the same proof of presence a payment does - tagged with its purpose
        # so it can never be spent as approval for a payment, or vice versa.
        payments.record_assertion(
            result, f"provisioning:{result.assertion_id}", 0, purpose="provisioning"
        )

    return {
        "assertion_id": result.assertion_id,
        "credential_id": result.credential_id,
        "user_verified": result.user_verified,
        "verified_at": result.verified_at,
        "amount_cents": result.amount_cents,
        "intent_id": result.intent_id,
    }


def _token_summary(token: vts.NetworkToken) -> AgentTokenSummary:
    method = payments.get_method(token.funding_method_id)
    mandate = token.mandate
    return AgentTokenSummary(
        token_id=token.token_id,
        funding_method_id=token.funding_method_id,
        funding_display=method.display if method else "",
        token_last4=token.token_last4,
        presentation_type=token.presentation_type.value,  # type: ignore[arg-type]
        status=token.status.value,  # type: ignore[arg-type]
        mandate_id=mandate.id,
        scope=MandateScope(
            per_transaction_cap_cents=mandate.per_transaction_cap_cents,
            cumulative_cap_cents=mandate.cumulative_cap_cents,
            spent_cents=mandate.spent_cents,
            remaining_cents=mandate.remaining_cents,
            allowed_mccs=sorted(mandate.allowed_mccs),
            # Derived from the mandate's ACTUAL categories rather than left at
            # the model default: a user who narrowed their agent to furniture
            # alone should not be shown the umbrella label for everything.
            category_label=vts.labels_for_mccs(mandate.allowed_mccs),
            allowed_merchants=sorted(mandate.allowed_merchants),
            max_merchants_per_transaction=mandate.max_merchants_per_transaction,
            require_user_presence=mandate.require_user_presence,
            expires_at=mandate.expires_at,
        ),
        created_at=token.created_at,
        revoked_at=mandate.revoked_at,
        revocation_reason=mandate.revocation_reason,
        assurance_level=token.assurance_level,
        assurance_method=token.assurance_method,
        uses=mandate.uses,
    )


@app.get("/api/agent-token/defaults")
async def agent_token_defaults() -> dict:
    """What the mandate dialog should suggest before the user adjusts it."""
    return {
        "per_transaction_cap_cents": config.DEFAULT_AGENT_PER_TXN_CAP_CENTS,
        "cumulative_cap_cents": config.DEFAULT_AGENT_TOTAL_CAP_CENTS,
        "ttl_hours": config.DEFAULT_AGENT_MANDATE_HOURS,
        "category_label": "Furniture & Home Decor",
        "allowed_mccs": sorted(vts.HOME_CATEGORY_MCCS),
        "known_merchant_mccs": vts.MERCHANT_MCC,
    }


@app.get("/api/agent-token", response_model=list[AgentTokenSummary])
async def agent_tokens() -> list[AgentTokenSummary]:
    """Every agent token, live or revoked. The revoked ones stay listed
    because a spending history the user cannot audit after revoking is not
    much of an audit trail."""
    return [_token_summary(t) for t in vts.list_tokens()]


@app.post("/api/agent-token/provision")
async def agent_token_provision(req: ProvisionTokenRequest) -> dict:
    """Enroll a card into the token service and mint a scoped AI_AGENT token.

    Requires a fresh passkey assertion. Granting standing spending authority
    is at least as sensitive as any single purchase it later permits, so it
    demands the same proof that the cardholder is present.

    Returns the mandate credential exactly once. It is a bearer credential:
    whoever holds it can spend within its scope, and nothing wider.
    """
    method = payments.get_method(req.funding_method_id)
    if method is None:
        raise HTTPException(404, "unknown payment method")

    if webauthn.CREDENTIALS:
        if not req.assertion_id:
            raise HTTPException(
                401,
                "verify with Face ID / Touch ID before granting the agent a mandate",
            )
        try:
            payments.consume_provisioning_assertion(req.assertion_id)
        except payments.PaymentError as exc:
            raise _payment_error(exc) from exc
        assurance = "fido2_device_biometric"
    else:
        # No authenticator on this device. The token is still scoped, but its
        # assurance level records honestly that no biometric backed it.
        assurance = "none"

    try:
        token, credential = vts.provision_token(
            funding_method_id=req.funding_method_id,
            per_transaction_cap_cents=req.per_transaction_cap_cents,
            cumulative_cap_cents=req.cumulative_cap_cents,
            allowed_merchants=frozenset(req.allowed_merchants),
            max_merchants_per_transaction=req.max_merchants_per_transaction,
            ttl_seconds=req.ttl_hours * 3600,
            assurance_method=assurance,
        )
    except vts.MandateViolation as exc:
        raise _mandate_error(exc) from exc

    method.tokenized = True
    return {
        "token": _token_summary(token).model_dump(),
        # Shown once, held by the agent thereafter.
        "mandate_credential": credential,
    }


# --- Merchant platform -----------------------------------------------------
#
# Lets third-party merchants onboard, publish products and see what they are
# owed. Requests are HMAC-signed rather than bearer-authenticated: a leaked
# bearer token is directly replayable, while a signature covers the method,
# path, timestamp, nonce and body, so a captured request cannot be redirected
# to a different endpoint or replayed at all.
#
#   POST /api/merchant/onboard        register; returns the API secret ONCE
#   POST /api/merchant/catalog        publish products (signed)
#   GET  /api/merchant/me             this merchant's account (signed)
#   GET  /api/merchant/balance        what is owed, and whether it can settle
#   GET  /api/merchant/payouts        per-order breakdown (signed)
#   POST /api/merchant/{id}/kyc       platform-side KYC outcome


def _merchant_error(exc: merchants.MerchantError) -> HTTPException:
    return HTTPException(exc.status, exc.detail)


def _merchant_summary(m: merchants.Merchant) -> MerchantSummary:
    return MerchantSummary(
        id=m.id,
        name=m.name,
        legal_name=m.legal_name,
        email=m.email,
        country=m.country,
        mcc=m.mcc,
        status=m.status.value,  # type: ignore[arg-type]
        kyc_status=m.kyc_status.value,  # type: ignore[arg-type]
        commission_bps=m.commission_bps,
        payout_account_last4=m.payout_account_last4,
        webhook_url=m.webhook_url,
        created_at=m.created_at,
        can_sell=m.can_sell,
        can_settle=m.can_settle,
    )


async def _authenticated_merchant(
    request: Request, raw_body: str | None = None
) -> merchants.Merchant:
    """Verify a signed merchant request.

    The body is signed byte-for-byte. Re-serializing parsed JSON would change
    whitespace and key order and break every signature, so the raw bytes are
    what both sides agree on.

    `raw_body` exists for multipart endpoints: FastAPI consumes the request
    stream to parse a form, so by the time this runs `request.body()` raises
    "Stream consumed". Those routes read the body themselves, before touching
    any form field, and pass it in.
    """
    headers = request.headers
    key_id = headers.get("x-merchant-key", "")
    signature = headers.get("x-merchant-signature", "")
    timestamp = headers.get("x-merchant-timestamp", "")
    nonce = headers.get("x-merchant-nonce", "")
    if not all((key_id, signature, timestamp, nonce)):
        raise HTTPException(
            401,
            "signed request required: x-merchant-key, x-merchant-signature, "
            "x-merchant-timestamp and x-merchant-nonce headers",
        )
    raw = raw_body if raw_body is not None else (await request.body()).decode()
    try:
        return merchants.authenticate(
            key_id=key_id,
            signature=signature,
            timestamp=timestamp,
            nonce=nonce,
            method=request.method,
            path=request.url.path,
            body=raw,
        )
    except merchants.MerchantError as exc:
        raise _merchant_error(exc) from exc


@app.post("/api/merchant/onboard", response_model=MerchantOnboardResponse)
async def merchant_onboard(req: MerchantOnboardRequest) -> MerchantOnboardResponse:
    """Register a merchant and issue its first API credential.

    Open by design so a merchant can self-serve, but the account starts
    PENDING with UNVERIFIED KYC: it may list products and receive orders,
    and nothing can settle to it until a human verifies who they are.
    """
    try:
        merchant, key_id, secret = merchants.onboard(
            name=req.name,
            legal_name=req.legal_name,
            email=req.email,
            country=req.country,
            payout_account_last4=req.payout_account_last4,
            payout_pan=req.payout_pan,
            webhook_url=req.webhook_url,
        )
    except merchants.MerchantError as exc:
        raise _merchant_error(exc) from exc
    merchants.SECRETS[key_id] = secret
    return MerchantOnboardResponse(
        merchant=_merchant_summary(merchant), key_id=key_id, api_secret=secret
    )


@app.get("/api/merchant/me", response_model=MerchantSummary)
async def merchant_me(request: Request) -> MerchantSummary:
    return _merchant_summary(await _authenticated_merchant(request))


@app.post("/api/merchant/catalog", response_model=MerchantCatalogResult)
async def merchant_catalog(request: Request) -> MerchantCatalogResult:
    """Publish products into the live catalog.

    The merchant name and MCC come from the authenticated credential, never
    from the payload, so a merchant cannot publish under another's name or
    assign itself a category that would place it inside an agent's mandate.
    """
    merchant = await _authenticated_merchant(request)
    index = STATE.get("index")
    if index is None:  # pragma: no cover
        raise HTTPException(503, "server still starting")

    try:
        push = MerchantCatalogPush.model_validate_json(await request.body())
    except ValueError as exc:
        raise HTTPException(400, f"invalid catalog payload: {exc}") from exc

    result = MerchantCatalogResult()
    added: list[CatalogItem] = []
    seen_skus: set[str] = set()
    for product in push.products:
        # Per-product validation, collecting errors rather than failing the
        # whole push: a merchant with 200 products and one bad URL should be
        # told which one, not have the batch rejected.
        if product.sku in seen_skus:
            result.rejected += 1
            result.errors.append(
                {"sku": product.sku, "error": "duplicate SKU in this upload"}
            )
            continue
        seen_skus.add(product.sku)
        try:
            checkout_url = merchants.validate_checkout_url(product.checkout_url)
        except merchants.MerchantError as exc:
            result.rejected += 1
            result.errors.append({"sku": product.sku, "error": exc.detail})
            continue
        try:
            item = CatalogItem(
                # Namespaced by merchant id so two merchants can use the same
                # SKU without colliding.
                id=f"{merchant.id}:{product.sku}",
                merchant=merchant.name,
                title=product.title,
                role=product.role,
                price_cents=product.price_cents,
                currency=product.currency,
                dimensions=Dimensions(
                    width_cm=product.width_cm,
                    depth_cm=product.depth_cm,
                    height_cm=product.height_cm,
                ),
                materials=product.materials,
                primary_color=product.primary_color,
                swatch=product.swatch,
                style_tags=product.style_tags,
                image_url=product.image_url,
                checkout_url=checkout_url,
                in_stock=product.in_stock,
                description=product.description,
            )
        except ValueError as exc:
            result.rejected += 1
            result.errors.append({"sku": product.sku, "error": str(exc)})
            continue
        added.append(item)
        result.item_ids.append(item.id)
        result.accepted += 1

    if added:
        try:
            index.add_items(added)  # type: ignore[attr-defined]
        except AttributeError:
            # The index predates merchant ingestion. Report honestly rather
            # than claiming products were published into a catalog that
            # cannot accept them.
            raise HTTPException(
                501,
                "this catalog index does not support runtime ingestion yet",
            ) from None
        # Keep the mandate's category map in step, so an agent token locked to
        # furniture recognises this merchant's category.
        vts.MERCHANT_MCC.setdefault(merchant.name, merchant.mcc)
    for warning in merchants.check_domain_consistency(
        merchant, [i.checkout_url for i in added]
    ):
        result.errors.append({"sku": "", "error": f"warning: {warning}"})
    return result


@app.post("/api/merchant/catalog/upload")
async def merchant_catalog_upload(request: Request) -> dict:
    """Upload a CSV or JSON catalogue feed.

    Two-phase by default: `publish=false` (the default) normalizes and
    validates, and returns exactly what WOULD be published plus every problem
    found - without changing anything. The merchant fixes the reported rows
    and re-uploads with `publish=true`.

    A preview step matters more here than in most upload flows: these products
    become things a shopper can be charged for, and a feed that silently
    published 400 correct rows and dropped 12 would leave the merchant
    believing their catalogue was live when part of it was not.

    Authentication is the same HMAC signature as the JSON endpoint. Multipart
    bodies are signed over the raw request body like any other.
    """
    # Read the raw body BEFORE parsing the form: signature verification needs
    # the exact bytes, and parsing consumes the stream.
    raw = await request.body()
    merchant = await _authenticated_merchant(
        request, raw_body=raw.decode("utf-8", "replace")
    )
    index = STATE.get("index")
    if index is None:  # pragma: no cover
        raise HTTPException(503, "server still starting")

    form = await request.form()
    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        raise HTTPException(400, "a `file` part is required (.csv or .json)")
    publish = str(form.get("publish", "")).lower() in ("1", "true", "yes", "on")
    filename = getattr(upload, "filename", "") or "upload"
    content = await upload.read()
    if len(content) > config.MAX_CATALOG_UPLOAD_BYTES:
        raise HTTPException(
            413,
            f"catalogue file is too large "
            f"({len(content) // 1024}KB, limit {config.MAX_CATALOG_UPLOAD_BYTES // 1024}KB)",
        )

    try:
        rows = merchants.parse_upload(content, filename)
    except merchants.MerchantError as exc:
        raise _merchant_error(exc) from exc

    if not rows:
        raise HTTPException(400, "the file contained no product rows")

    products, problems = merchants.normalize_feed(rows)

    # Checkout URLs are validated after normalization, since normalization is
    # what resolves which column held the link in the first place.
    valid: list[dict] = []
    for product in products:
        try:
            product["checkout_url"] = merchants.validate_checkout_url(
                product["checkout_url"]
            )
        except merchants.MerchantError as exc:
            problems.append({"row": "", "sku": product["sku"], "error": exc.detail})
            continue
        valid.append(product)

    # Rows that only reached the catalog because a dimension was estimated.
    # Surfaced separately from the accepted count so the merchant can see that
    # some of their products carry numbers they did not supply - and by which
    # source, since a model estimate and a lookup-table one are not equally
    # good. Never folded into `normalized`: an estimate is a published product
    # AND a caveat, not one or the other.
    estimated = [p for p in valid if p.get("estimated_dims")]
    result = {
        "filename": filename,
        "rows_read": len(rows),
        "normalized": len(valid),
        "rejected": len(problems),
        "errors": problems[:100],
        "published": False,
        "preview": valid[:5],
        "dimensions_estimated": len(estimated),
        "estimate_sources": sorted({
            p.get("estimate_source", "") for p in estimated if p.get("estimate_source")
        }),
    }

    if not publish:
        result["next_step"] = (
            "Nothing was published. Fix the rows listed in `errors`, then "
            "re-upload with publish=true."
        )
        return result

    added: list[CatalogItem] = []
    for product in valid:
        role = _role_from_category(product["category"], product["title"])
        if role is None:
            label = product["category"] or product["title"] or "unknown"
            problems.append(
                {
                    "row": "",
                    "sku": product["sku"],
                    "error": (
                        f"category {label!r} does not map to a placeable role. "
                        "This catalog places sofas, accent chairs, coffee "
                        "tables, floor lamps and rugs."
                    ),
                }
            )
            continue
        try:
            item = CatalogItem(
                id=f"{merchant.id}:{product['sku']}",
                merchant=merchant.name,
                title=product["title"],
                role=role,
                price_cents=product["price_cents"],
                currency=product["currency"],
                dimensions=Dimensions(
                    width_cm=product["width_cm"],
                    depth_cm=product["depth_cm"],
                    height_cm=product["height_cm"],
                ),
                materials=product["materials"],
                primary_color=product["primary_color"],
                swatch="#cccccc",
                image_url=product["image_url"],
                checkout_url=product["checkout_url"],
                in_stock=product["in_stock"],
                description=product["description"],
            )
        except ValueError as exc:
            problems.append({"row": "", "sku": product["sku"], "error": str(exc)})
            continue
        added.append(item)

    with_images = 0
    if added:
        try:
            index.add_items(added)  # type: ignore[attr-defined]
        except AttributeError:
            raise HTTPException(
                501, "this catalog index does not support runtime ingestion yet"
            ) from None
        vts.MERCHANT_MCC.setdefault(merchant.name, merchant.mcc)
        # How many products ended up with an image vector, so the merchant is
        # told whether "find one that looks like this" will surface them.
        # Reported rather than assumed: their image host is very likely not on
        # IMAGE_FETCH_ALLOWED_HOSTS, in which case these products are live and
        # text-searchable but invisible to reverse image search - a difference
        # they would otherwise only discover by noticing an absence.
        vectors = getattr(index, "_image_vectors", {})
        with_images = sum(1 for i in added if i.id in vectors)

    result["published"] = True
    result["image_searchable"] = with_images
    if added and with_images < len(added):
        result["image_search_note"] = (
            f"{len(added) - with_images} of {len(added)} published products have no "
            "image vector, so they rank on text only and will not appear in "
            "reverse image search. Usually the image host is not on the "
            "platform's image allowlist, or the image could not be fetched."
        )
    result["accepted"] = len(added)
    result["rejected"] = len(problems)
    result["errors"] = problems[:100]
    result["item_ids"] = [i.id for i in added]
    return result


def _role_from_category(category: str, title: str = "") -> Role | None:
    """Map a merchant's own category vocabulary onto our Role enum.

    Substring matching rather than exact: feeds say "3-seater sofas",
    "Living / Seating" and "SOFA_LARGE" for the same thing. The title is a
    fallback because plenty of feeds pair a useless category column
    ("Furniture") with a precise name ("Fjord Coffee Table").

    Returns None when nothing matches. The Role enum is deliberately small -
    it drives the layout solver, which only knows how to place these five
    pieces - so a product outside it cannot be placed, and saying so is more
    useful to the merchant than filing it under a role it does not fill.
    """
    haystack = f"{category} {title}".lower()
    table: list[tuple[tuple[str, ...], Role]] = [
        # Most specific first: "coffee table" must beat a bare "table", and
        # "floor lamp" must not be swallowed by the chair rule.
        (("coffee table", "cocktail table", "centre table", "center table"), Role.COFFEE_TABLE),
        (("floor lamp", "standing lamp", "arc lamp", "torchiere"), Role.FLOOR_LAMP),
        (("sofa", "couch", "settee", "loveseat", "sectional"), Role.SOFA),
        (("armchair", "accent chair", "lounge chair", "occasional chair", "chair"), Role.ACCENT_CHAIR),
        (("rug", "carpet", "runner"), Role.RUG),
    ]
    for needles, role in table:
        if any(n in haystack for n in needles):
            return role
    return None


@app.get("/api/merchant/balance", response_model=MerchantBalance)
async def merchant_balance(request: Request) -> MerchantBalance:
    """What this merchant is owed, and whether it can actually be paid."""
    merchant = await _authenticated_merchant(request)
    return MerchantBalance(**merchants.balance_for(merchant.id))


@app.get("/api/merchant/payouts")
async def merchant_payouts(request: Request) -> dict:
    """Per-order breakdown. Every record is `pending_settlement`: this
    codebase computes what is owed but moves no money."""
    merchant = await _authenticated_merchant(request)
    return {
        "merchant_id": merchant.id,
        "payouts": merchants.payouts_for(merchant.id),
        "disclaimer": (
            "SIMULATED - amounts are computed and reconciled, but no funds are "
            "transferred. Real settlement requires an acquirer relationship."
        ),
    }


@app.post("/api/merchant/{merchant_id}/settle")
async def merchant_settle(merchant_id: str, body: dict | None = None) -> dict:
    """Pay a merchant's pending balance via Visa Direct.

    Platform-side, and behind operator auth in a real deployment - this is the
    endpoint that moves money. `dry_run` reports what would be paid without
    attempting anything.

    When live payouts are not configured this returns `simulated: true` and
    marks nothing paid, rather than reporting a settlement that did not
    happen.
    """
    dry_run = bool((body or {}).get("dry_run"))
    try:
        return merchants.settle(merchant_id, dry_run=dry_run)
    except merchants.MerchantError as exc:
        raise _merchant_error(exc) from exc


@app.get("/api/payouts/status")
async def payout_status() -> dict:
    """Whether live Visa Direct payouts are configured, and what is missing."""
    return visa_direct.status()


@app.post("/api/merchant/{merchant_id}/kyc", response_model=MerchantSummary)
async def merchant_kyc(merchant_id: str, body: dict) -> MerchantSummary:
    """Record a KYC outcome.

    Platform-side, and in a real deployment this sits behind operator auth -
    it is the switch that decides whether money may move to an account.
    """
    status = str(body.get("kyc_status") or "")
    try:
        kyc = merchants.KycStatus(status)
    except ValueError as exc:
        raise HTTPException(
            400,
            f"kyc_status must be one of: "
            f"{', '.join(k.value for k in merchants.KycStatus)}",
        ) from exc
    try:
        return _merchant_summary(merchants.set_kyc(merchant_id, kyc))
    except merchants.MerchantError as exc:
        raise _merchant_error(exc) from exc


@app.get("/api/merchant", response_model=list[MerchantSummary])
async def merchant_list() -> list[MerchantSummary]:
    """Every registered merchant. Platform-side view."""
    return [_merchant_summary(m) for m in merchants.list_merchants()]


@app.post("/api/agent-token/revoke", response_model=AgentTokenSummary)
async def agent_token_revoke(req: RevokeMandateRequest) -> AgentTokenSummary:
    """Revoke the agent's spending mandate. THE CARD IS UNAFFECTED.

    Deliberately requires no step-up. Taking authority away should always be
    easier than granting it - a revocation gated behind a biometric is one a
    user cannot perform in the moment they most need to.
    """
    try:
        token = vts.revoke_token(req.token_id, req.reason)
    except vts.MandateViolation as exc:
        raise _mandate_error(exc) from exc
    return _token_summary(token)



# --- Health ----------------------------------------------------------------


@app.get("/api/health")
async def health() -> dict:
    index = STATE.get("index")
    renderer = STATE.get("renderer")
    vision = STATE.get("vision")
    return {
        "status": "ok" if index is not None else "starting",
        "providers": "openai" if config.HAS_OPENAI else "mock",
        "offline_mode": not config.HAS_OPENAI,
        "catalog_seeded": index is not None,
        # Broken out per provider, because "providers: openai" only says a key
        # was present - it does not say which subsystems actually built a
        # client. A single rolled-up flag sends you to the wrong place when one
        # of them silently falls back.
        "vision": {
            "source": getattr(vision, "source", "unavailable"),
            "model": config.VISION_MODEL if config.HAS_OPENAI else None,
            # Vision only runs when a photo is supplied. Without one the room
            # comes back with source "default", which is not a fallback.
            "requires_photo": True,
        },
        "intent": {
            "source": getattr(STATE.get("intent"), "source", "unavailable"),
            "model": config.INTENT_MODEL if config.HAS_OPENAI else None,
        },
        "embeddings": {
            "source": getattr(getattr(index, "embedder", None), "source", "unavailable"),
            "model": config.EMBEDDING_MODEL if config.HAS_OPENAI else None,
        },
        "renderer": {
            "source": getattr(renderer, "source", "unavailable"),
            "method": getattr(getattr(renderer, "method", None), "value", "unavailable"),
            "can_compose": bool(getattr(renderer, "can_compose", False)),
            "compose_model": config.GEMINI_IMAGE_MODEL if config.HAS_GEMINI else None,
        },
    }


@app.post("/api/detect")
async def detect(
    image: UploadFile = File(...),
    # Off by default: detection alone is fast and cheap, while matching costs
    # one embedding call per object found.
    match: bool = Form(False),
    limit: int = Form(config.REVERSE_SEARCH_LIMIT),
) -> dict:
    """Identify the furniture in a room photo, and optionally price it.

    Two things in one call because they answer one question: "what is in this
    room, and can I buy it here?" The boxes are normalized [0,1] so a client
    can draw them over the photo at any display size.

    Reverse search is a *suggestion*, not an identification: see `confident` on
    each result for whether the top match actually resembles the object or is
    merely the closest thing in a small catalog.
    """
    index = STATE.get("index")
    detector = STATE.get("detector")
    if index is None or detector is None:  # pragma: no cover - pre-lifespan
        raise HTTPException(503, "server still starting")

    raw = await image.read()
    if len(raw) > config.MAX_IMAGE_BYTES:
        raise HTTPException(
            413, f"image exceeds {config.MAX_IMAGE_BYTES // (1024 * 1024)}MB limit"
        )
    if not raw:
        raise HTTPException(400, "empty image file")
    image_b64 = prepare_image(raw)
    if image_b64 is None:
        raise HTTPException(400, "could not decode image file")

    # 503 rather than an empty list: "we cannot look" and "we looked and found
    # nothing" are different answers, and a client that conflated them would
    # tell the user their room is empty.
    if not detector.available:  # type: ignore[attr-defined]
        raise HTTPException(
            503,
            "furniture detection needs OPENAI_API_KEY; the server is running "
            "with offline providers",
        )

    found = await asyncio.to_thread(detector.detect, image_b64)  # type: ignore[attr-defined]
    if not match:
        return {
            "count": len(found),
            "detections": [d.model_dump(mode="json") for d in found],
        }

    limit = max(1, min(limit, 20))
    results = await asyncio.to_thread(index.identify_room, found, limit)  # type: ignore[attr-defined]
    return {
        "count": len(results),
        "results": [
            _rewrite_image_urls(r.model_dump(mode="json")) for r in results
        ],
    }


@app.post("/api/shop-the-look")
async def shop_the_look(
    image: UploadFile = File(...),
    role: str = Form(""),
    caption: str = Form(""),
    limit: int = Form(5),
) -> dict:
    """Find catalog items that look like the uploaded object.

    The image is the query, not a description of it: CLIP ranks against the
    product photographs, so silhouette and material match directly. Upload a
    crop of one piece of furniture, not a whole room - a wide shot averages
    every object in it into one vector and matches nothing in particular.

    `caption` and `role` are optional. A role narrows the search to that
    category; a caption adds a text signal fused with the image one.
    """
    index = STATE.get("index")
    if index is None:
        raise HTTPException(503, "catalog is still seeding")

    raw = await image.read()
    if not raw:
        raise HTTPException(400, "empty image")
    if len(raw) > config.MAX_IMAGE_BYTES:
        raise HTTPException(
            413, f"image exceeds {config.MAX_IMAGE_BYTES // (1024 * 1024)}MB limit"
        )

    parsed_role: Role | None = None
    if role:
        try:
            parsed_role = Role(role)
        except ValueError:
            raise HTTPException(422, f"unknown role {role!r}")

    det = Detection(
        role=parsed_role,
        label=role or "furniture",
        score=1.0,
        # The whole upload is the object: the caller cropped it, so there is no
        # sub-box to report.
        x1=0.0, y1=0.0, x2=1.0, y2=1.0,
        caption=caption,
    )
    result = await asyncio.to_thread(
        index.search_by_detection,
        det,
        max(1, min(limit, 20)),
        base64.b64encode(raw).decode(),
    )
    payload = result.model_dump(mode="json")
    payload["image_search"] = bool(
        getattr(index, "_image_vectors", None) and index.clip.available
    )
    return _rewrite_image_urls(payload)


@app.get("/api/catalog")
async def catalog(limit: int = 100) -> dict:
    """Inspect the live catalog. Useful when debugging retrieval.

    Reports what retrieval can actually return - seeded items plus anything
    merchants have published - rather than the seed list alone, so a merchant
    debugging a missing product can see whether it was ingested at all.
    """
    index = STATE.get("index")
    if index is None:  # pragma: no cover - only before lifespan completes
        raise HTTPException(503, "server still starting")

    items = index.all_items()  # type: ignore[attr-defined]
    return {
        "count": len(items),
        "items": [
            _rewrite_image_urls(i.model_dump(mode="json")) for i in items[:limit]
        ],
    }
