/* Room Hack workspace.
   Everything that talks to the API, plus the workspace UI it drives. The
   landing page does not load this file: it has no design state to hold, and
   keeping the two apart is what stops the marketing page from paying for the
   planner's weight. */

/* --- API client -----------------------------------------------------------
   The backend streams Server-Sent Events from POST endpoints, so EventSource
   is unusable (it only issues GETs). We read the response body ourselves and
   parse the `event:` / `data:` framing described in README.md. */
const API = (localStorage.getItem("roomhack_api") || "http://127.0.0.1:8000").replace(/\/$/, "");
const api = path => API + path;

// Product images come back as /assets/... paths served by the API host.
const assetUrl = url => (typeof url === "string" && url.startsWith("/assets/")) ? API + url : url;

/* Prices come from the catalog, and so does their currency - the IKEA import
   carries SGD rather than relabelling it (see ikea_import.py). Hardcoding "$"
   here printed Singapore prices in what most readers take for US dollars, so
   the symbol is read off whatever the server actually sent and only falls back
   to the catalog's own currency when nothing has arrived yet.

   `money` stays unary because it has 23 call sites, most of them inside
   template literals in the payment sheet; threading a currency argument
   through all of them would be noise for a value that is global to a session
   anyway. */
const CURRENCY_SYMBOLS = { SGD: "S$", USD: "US$", EUR: "€", GBP: "£", MYR: "RM" };
let currency = "SGD";

function setCurrency(code) {
  if (code) currency = code;
}

const currencySymbol = () => CURRENCY_SYMBOLS[currency] || (currency + " ");

const money = cents => currencySymbol() + (cents / 100).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const titleize = s => String(s ?? "").replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());

/** POST `body` and dispatch each SSE frame to handlers[event].
 *  Returns the `done` payload. Throws on a non-OK status so the caller can
 *  surface a normal HTTP error (413 oversized image, 409 no design yet). */
async function streamSSE(path, body, handlers, signal) {
  const res = await fetch(api(path), {
    method: "POST",
    body: body instanceof FormData ? body : JSON.stringify(body),
    headers: body instanceof FormData ? {} : { "Content-Type": "application/json" },
    signal,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch {}
    throw new Error(detail);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let donedata = null;

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // Frames are separated by a blank line. A partial trailing frame stays in
    // the buffer until the rest of it arrives.
    let split;
    while ((split = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);
      let event = "message";
      const dataLines = [];
      for (const line of raw.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7).trim();
        else if (line.startsWith("data: ")) dataLines.push(line.slice(6));
        // Lines starting with ":" are comments (": connected", ": ping").
      }
      if (!dataLines.length) continue;
      let payload;
      try { payload = JSON.parse(dataLines.join("\n")); } catch { continue; }
      if (event === "done") donedata = payload;
      handlers[event]?.(payload);
      handlers["*"]?.(event, payload);
    }
  }
  return donedata;
}

/* --- State ---------------------------------------------------------------- */
const $ = id => document.getElementById(id);
const state = {
  sessionId: null,   // assigned by the server on the first `done` frame
  roomPhoto: null,   // File the user uploaded
  bundles: [],       // Suggested sets from the last turn
  layout: null,
  room: null,
  cart: null,
  options: [],       // alternatives, per role
  busy: false,
  openings: [],      // doors and windows the user has described
  proposal: null,    // the estimate the server offered for confirmation
  shopResults: [],   // reverse-search matches for the pieces in the photo
  shopLoading: false,
  rendered: false,   // has a room visualization been produced this session?
  renderStale: false,// has the design changed since that visualization?
  lastRenderHTML: "",// the composed render markup, shared by layout and swap
  stage: "brief",    // current step in the four-stage sequence
  // Stages the sequence has already pulled the user forward to. Per-stage
  // rather than a single latch: identify opens first and design opens after
  // it, and BOTH should carry the user - but each only once, so a later
  // re-render of the same stage cannot yank someone back to it.
  autoAdvanced: {},
};

/* --- Chat transcript ------------------------------------------------------ */
const messageList = () => $("message-list");

function addBubble(who, html, cls = "") {
  const el = document.createElement("div");
  el.className = "bubble " + cls;
  el.innerHTML = `<small>${esc(who)}</small>${html}`;
  messageList().appendChild(el);
  messageList().scrollTop = messageList().scrollHeight;
  return el;
}

/** A bubble the assistant streams text_delta into.
 *  The reply arrives as one long string carrying "• " bullets and blank-line
 *  paragraph breaks. Rendered as raw text that reads as an undifferentiated
 *  wall, so the accumulated buffer is re-laid-out into paragraphs and lists on
 *  each delta. Re-rendering the whole buffer (rather than appending nodes) is
 *  what keeps a bullet correct when its line arrives split across deltas. */
function startAssistantBubble() {
  const el = addBubble("Room Hack", "");
  const body = document.createElement("div");
  body.className = "reply";
  el.appendChild(body);
  let buffer = "";

  const layout = () => {
    body.replaceChildren();
    // Split into blocks on blank lines, then treat runs of "• " as one list.
    for (const block of buffer.split(/\n\s*\n/)) {
      const lines = block.split("\n").map(l => l.trim()).filter(Boolean);
      if (!lines.length) continue;
      let list = null;
      for (const line of lines) {
        const bullet = line.match(/^[•\-\u2022]\s*(.+)$/);
        if (bullet) {
          if (!list) { list = document.createElement("ul"); body.appendChild(list); }
          const li = document.createElement("li");
          li.textContent = bullet[1];
          list.appendChild(li);
        } else {
          list = null;
          const para = document.createElement("p");
          para.textContent = line;
          body.appendChild(para);
        }
      }
    }
  };

  return {
    append(text) {
      buffer += text;
      layout();
      messageList().scrollTop = messageList().scrollHeight;
    },
    get empty() { return buffer.trim() === ""; },
    remove() { el.remove(); },
  };
}

function setBusy(on) {
  state.busy = on;
  $("chat-send").disabled = on;
  $("chat-send").textContent = on ? "…" : "Send";
  document.body.classList.toggle("is-busy", on);
  if (state.stage) paintStageFoot();
}

/* --- The one call that drives everything ---------------------------------- */
async function sendTurn(message, opts = {}) {
  const { includePhoto = false } = opts;
  if (state.busy) return;
  setBusy(true);

  const form = new FormData();
  form.append("message", message);
  if (state.sessionId) form.append("session_id", state.sessionId);

  // Budget and aesthetic are only sent when set: the backend treats an absent
  // field as "keep what the conversation already established".
  const budget = $("chat-budget").value.trim();
  if (budget) form.append("budget", budget);
  const style = $("chat-style").value;
  if (style) form.append("aesthetic", style);

  // The photo is uploaded once; later turns reuse it from the session.
  if (includePhoto && state.roomPhoto) form.append("image", state.roomPhoto);

  // `opts` carries the accuracy fields. They ride on a normal turn rather than
  // a separate endpoint because the backend treats them as part of the brief:
  // confirming a dimension is a message in the conversation, not a side call.
  const width = opts.width ?? $("room-width").value.trim();
  const depth = opts.depth ?? $("room-depth").value.trim();
  if (width && depth) {
    form.append("room_width_cm", width);
    form.append("room_depth_cm", depth);
  }
  // Sent alone, this accepts the last proposal as-is; sent with dimensions, it
  // marks them as confirmed rather than measured.
  if (opts.confirm) form.append("confirm_dimensions", "true");
  if (opts.irregular) form.append("irregular", "true");
  // Openings are cumulative for the session, so the full list goes every time
  // a turn carries them - a partial list would read as a deletion.
  if (opts.openings && state.openings.length) {
    form.append("openings", JSON.stringify(state.openings));
  }

  // The trace goes in first: it narrates the work, so it belongs above the
  // reply it produced.
  const think = startThinking();
  const bubble = startAssistantBubble();
  const status = setStatus("Thinking…");

  try {
    const done = await streamSSE("/api/chat", form, {
      text_delta: d => bubble.append(d.text || ""),
      room_analysis: d => {
        state.room = d.room;
        renderRoomFacts(d.room);
        const r = d.room;
        think.mark("analyze", r
          ? `${Math.round(r.width_cm) / 100} × ${Math.round(r.depth_cm) / 100} m`
          : "");
        // Detection only runs on a photo, so it is only claimed when the
        // analysis actually carried detections.
        if (r?.detections?.length) {
          think.mark("identify", `${r.detections.length} piece${r.detections.length === 1 ? "" : "s"} found`);
          // Match those pieces against the catalog. Fired once per photo, not
          // per turn: the detections are cached server-side but the matching
          // is not, and a follow-up turn re-sends the same room.
          if (!state.shopResults.length) loadShopMatches();
        }
      },
      layout_update: d => {
        state.layout = d.layout;
        renderFloorPlan(d.layout);
        const n = d.layout?.placements?.length;
        think.mark("solve", n ? `${n} placed` : "");
      },
      // `alternatives` arrives before `cart_update`, and the swap filter needs
      // the cart to know which roles were actually placed, so re-render the
      // options once the cart lands.
      cart_update: d => {
        state.cart = d.cart;
        setCurrency(d.cart?.currency);
        renderCart(d);
        renderAlternatives(state.options);
        think.mark("price", d.cart?.subtotal_cents != null ? money(d.cart.subtotal_cents) : "");
      },
      alternatives: d => {
        state.options = d.options || [];
        renderAlternatives(state.options);
        const n = state.options.reduce((sum, o) => sum + (o.alternatives?.length || 0), 0);
        think.mark("retrieve", n ? `${n} candidates` : "");
      },
      bundles: d => renderBundles(d.bundles),
      clarification_needed: d => renderClarifications(d),
      dimension_proposal: d => renderDimensionProposal(d.proposal),
      intent: () => {},
      error: d => addBubble("Error", `<span class="err">${esc(d.message)}</span>`),
    });
    if (done?.session_id) state.sessionId = done.session_id;
    think.finish();
    if (bubble.empty) bubble.remove();
    // The first visualization is automatic: a design is much easier to judge
    // as a picture of your own room than as a floor plan, and a user cannot
    // know to ask for something they have not seen. Later ones are explicit,
    // because by then they know it exists and each costs a slow generative
    // call they may not want on every tweak.
    if (!state.rendered && !state.renderPending && state.roomPhoto
        && state.cart?.lines?.length) {
      state.renderPending = true;
    }
  } catch (err) {
    // A failed turn should not leave a half-finished log standing above the
    // error - it would read as work that succeeded.
    think.remove();
    bubble.remove();
    addBubble("Error", `<span class="err">${esc(err.message)}</span>`);
  } finally {
    think.finish();
    status.clear();
    setBusy(false);
    updateActionAvailability();
  }
  // Outside the finally: doRender bails while busy, so it has to run after
  // setBusy(false) has actually landed.
  if (state.renderPending) {
    state.renderPending = false;
    setStage("design");
    doRender();
  }
}

/* --- Thinking trace -------------------------------------------------------
   A line-by-line account of what the agent is doing, rendered in the
   transcript above the reply rather than inside a bubble.

   Two sources drive it, and the split matters:

   1. REAL EVENTS. The backend already streams `room_analysis`, `alternatives`,
      `layout_update` and `cart_update` as each stage completes. Those tick
      their line and can carry what was actually found - the measured room, the
      number of candidates, the piece count - so the trace reports fact.

   2. PACING. Between those events there is dead air, and a single line sitting
      on a spinner for eight seconds reads as a hang. So a step whose event has
      not arrived yet advances on a timer to the next *unconfirmed* line, which
      keeps the trace moving without ever ticking a step that has not happened.

   A line is only marked done by its event, or when a later event proves it
   passed. Nothing here invents a result. */
const THINK_STEPS = [
  { key: "analyze",  text: "Analyzing room" },
  { key: "identify", text: "Identifying what's already there" },
  { key: "retrieve", text: "Searching the catalog" },
  { key: "consider", text: "Considering layouts" },
  { key: "solve",    text: "Placing furniture" },
  { key: "price",    text: "Pricing the bill of materials" },
];

function startThinking() {
  const wrap = document.createElement("div");
  wrap.className = "think";
  wrap.setAttribute("role", "status");
  wrap.setAttribute("aria-live", "polite");
  messageList().appendChild(wrap);

  const lines = new Map();
  let index = -1;      // last line revealed
  let timer = null;
  let stopped = false;

  const scroll = () => { messageList().scrollTop = messageList().scrollHeight; };

  /** Reveal the next line, marking the previous one done.
   *  Ticking goes through settle() so a line advanced by pacing still picks up
   *  a note whose event landed while it was on screen. */
  function reveal() {
    if (stopped || index >= THINK_STEPS.length - 1) return;
    if (index >= 0) settle(THINK_STEPS[index].key);
    index += 1;
    const step = THINK_STEPS[index];
    const el = document.createElement("div");
    el.className = "think-line doing";
    el.innerHTML = '<span class="think-mark"></span><span class="think-text"></span>';
    el.querySelector(".think-text").textContent = step.text;
    wrap.appendChild(el);
    lines.set(step.key, el);
    lastAt = performance.now();
    scroll();
  }

  /* Pacing. The backend answers fast enough that all six events can land
     inside a couple of seconds, which flickers past unread. So a line is held
     on screen for a minimum before the next one is allowed to appear, and
     confirmations queue behind that floor rather than fast-forwarding it. */
  const MIN_LINE_MS = 620;
  let lastAt = 0;
  let confirmed = -1;    // furthest step an event has actually proven
  let finishing = false;
  const notes = new Map();

  /** Tick a line and hang its note, if the event carried one. */
  function settle(key) {
    const el = lines.get(key);
    if (!el) return;
    if (el.classList.contains("doing")) el.classList.replace("doing", "done");
    const note = notes.get(key);
    if (note && !el.querySelector(".think-note")) {
      const span = document.createElement("span");
      span.className = "think-note";
      span.textContent = " · " + note;
      el.appendChild(span);
    }
  }

  /* One pump for the whole trace. It advances at most one line per
     MIN_LINE_MS, whether that line was confirmed by an event or is just the
     next unconfirmed step being paced along, so a burst of six events reads at
     the same speed as a slow one. */
  function drain() {
    clearTimeout(timer);
    if (stopped) return;

    // Settle every line an event has already proven.
    for (let i = 0; i <= confirmed && i <= index; i++) settle(THINK_STEPS[i].key);

    if (index >= THINK_STEPS.length - 1) {
      if (finishing) {
        settle(THINK_STEPS[index].key);
        wrap.querySelectorAll(".think-line.doing").forEach(el => el.classList.replace("doing", "done"));
        stopped = true;
      }
      scroll();
      return;
    }
    // Nothing more to show yet: the next step is unconfirmed and the turn is
    // still running, so pace it rather than stalling on a spinner.
    const wait = Math.max(0, MIN_LINE_MS - (performance.now() - lastAt));
    timer = setTimeout(() => {
      if (stopped) return;
      reveal();
      drain();
    }, wait || (finishing ? 0 : 260 + Math.random() * 420));
  }

  reveal();
  drain();

  return {
    /** An event landed: tick this step (and everything before it) and move on.
     *  `note` is what was actually found, shown beside the line.
     *  Paced, so a burst of events still reads line by line. */
    mark(key, note) {
      if (stopped) return;
      const target = THINK_STEPS.findIndex(st => st.key === key);
      if (target < 0) return;
      notes.set(key, note);
      confirmed = Math.max(confirmed, target);
      drain();
    },
    /** The turn is over. Let whatever is still queued finish drawing, then
     *  tick the last line - cutting the log off mid-list would lose stages
     *  that actually ran. */
    finish() {
      if (stopped) return;
      confirmed = THINK_STEPS.length - 1;
      finishing = true;
      drain();
    },
    /** The turn failed or produced nothing - drop the trace entirely rather
     *  than leaving a half-finished log above an error. */
    remove() {
      stopped = true;
      clearTimeout(timer);
      wrap.remove();
    },
    get empty() { return wrap.children.length === 0; },
  };
}

/* --- Status line ---------------------------------------------------------- */
function setStatus(text) {
  const el = $("run-status");
  el.innerHTML = `<span class="spin"></span><span>${esc(text)}</span>`;
  el.hidden = false;
  return { clear: () => { el.hidden = true; } };
}

/* --- Room facts ----------------------------------------------------------- */
function renderRoomFacts(room) {
  if (!room) return;
  const measured = room.measured
    ? `<span class="pill ok">measured</span>`
    : `<span class="pill warn">estimated from photo</span>`;
  $("room-facts").innerHTML = `
    <div class="row"><span class="muted">Size</span><span>${Math.round(room.width_cm)} × ${Math.round(room.depth_cm)} cm ${measured}</span></div>
    <div class="row"><span class="muted">Focal wall</span><span>${esc(titleize(room.focal_wall))}</span></div>
    <div class="row"><span class="muted">Flooring</span><span>${esc(room.flooring)}</span></div>
    <div class="row"><span class="muted">Light</span><span>${esc(room.lighting)}</span></div>
    ${room.notes ? `<p class="muted note-line">${esc(room.notes)}</p>` : ""}
    ${renderDetections(room.detections)}
    <p class="muted note-line">Analysis source: ${esc(room.source)}</p>`;
  // Once the room is no longer a bare estimate the proposal is answered, so
  // the card retires rather than inviting the same confirmation twice.
  if (room.dimension_source && room.dimension_source !== "estimated") {
    $("panel-confirm").hidden = true;
  }
  $("panel-room").hidden = false;
  syncStages();
}

/* What the photo already contains. Pieces we sell are marked as replaceable;
   the rest are listed anyway, because a user comparing this against their own
   room should see that we spotted the bookshelf even though we can't swap it. */
function renderDetections(detections) {
  if (!detections || !detections.length) return "";
  const rows = detections.map(d => {
    const tag = d.role
      ? `<span class="pill ok">can replace</span>`
      : `<span class="pill">not in catalog</span>`;
    return `<div class="row"><span class="muted">${esc(titleize(d.label))}</span><span>${tag}</span></div>`;
  }).join("");
  return `<div class="row"><span class="muted">Already here</span><span></span></div>${rows}`;
}

/* --- Shop the photo (reverse search) ---------------------------------------
   Every detected object, matched against the catalog. This is useful well
   beyond redesigning your own room: pointed at a photo of someone else's, it
   answers "where do I buy that?", which is the question a room photo usually
   provokes.

   Runs off /api/detect rather than the chat graph. The graph's detections come
   back without matches, and matching costs an embedding call per object, so it
   is worth doing once against the photo rather than on every design turn. */
async function loadShopMatches() {
  // Needs the original file: the endpoint takes a multipart upload, and the
  // detections on state.room carry boxes but no catalog matches.
  if (!state.roomPhoto || state.shopLoading) return;
  state.shopLoading = true;

  const body = $("shop-matches");
  $("panel-shop").hidden = false;
  // Shaped like the result: one card per piece we expect to match. The
  // detections are already known when this runs, so the count is the real
  // one rather than a guess - the skeleton is the same length as the list
  // that replaces it.
  body.innerHTML = skeletonCards(
    Math.min(Math.max(state.room?.detections?.length || 3, 1), 6),
    "Finding furniture like yours in the catalog…"
  );
  // Open the stage now rather than on completion, so the skeleton is visible
  // while it fills instead of the panel appearing only once it is done.
  syncStages();

  try {
    const form = new FormData();
    form.append("image", state.roomPhoto);
    form.append("match", "true");
    const res = await fetch(api("/api/detect"), { method: "POST", body: form });
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch {}
      throw new Error(detail);
    }
    const data = await res.json();
    state.shopResults = data.results || [];
    renderShopMatches(state.shopResults);
  } catch (err) {
    // A failed match does not invalidate the analysis, so this degrades to a
    // note in its own panel rather than an error over the whole stage.
    body.innerHTML = `<div class="skip">${esc(err.message)}</div>`;
  } finally {
    state.shopLoading = false;
    syncStages();
  }
}

function renderShopMatches(results) {
  const body = $("shop-matches");
  if (!results || !results.length) {
    body.innerHTML = `<p class="muted">No furniture found in the photo.</p>`;
    return;
  }

  body.innerHTML = results.map(r => {
    const d = r.detection || {};
    // `confident` is the server's own honesty flag: matches always come back
    // for a role we stock, so without this a nearest neighbour would read as
    // an identification. Never infer it from the score here - the threshold
    // that defines it is calibrated server-side.
    // Three states, not two. An object outside our catalog (role null) is
    // searched unfiltered, so its "matches" are whatever the index found
    // nearest - a bookshelf comes back as side tables. Saying "similar only"
    // there would still imply we sell something like it, so it gets its own
    // wording.
    const unsold = !d.role;
    const tag = unsold
      ? `<span class="pill">we don't sell these</span>`
      : r.confident
        ? `<span class="pill ok">close match</span>`
        : `<span class="pill">similar only</span>`;

    const cards = (r.matches || []).map(m => `
      <a class="alt" href="${esc(m.checkout_url || "#")}" target="_blank" rel="noopener noreferrer">
        <img src="${esc(assetUrl(m.image_url))}" alt="${esc(m.title)}" loading="lazy">
        <span class="alt-name">${esc(m.title)}</span>
        <span class="muted">${esc(m.merchant)}</span>
        <span class="delta">${money(m.price_cents)}</span>
      </a>`).join("");

    return `
      <div class="shop-group">
        <div class="shop-head"><span class="shop-what">${esc(titleize(d.label || "piece"))}</span>${tag}</div>
        ${d.caption ? `<p class="shop-seen">${esc(d.caption)}</p>` : ""}
        ${unsold
          ? `<p class="shop-none">Not something we stock — shown so you can see what we spotted.</p>`
          : cards
            ? `<div class="alt-row">${cards}</div>`
            : `<p class="shop-none">Nothing in the catalog fills this role.</p>`}
      </div>`;
  }).join("");
}

/* --- Accuracy: dimensions and openings -------------------------------------
   The solver withholds anything that sits against a wall until it trusts the
   room's dimensions, because placing a bookshelf from a photo estimate that is
   off by a metre puts it through the wall. Three tiers of trust exist server
   side - ESTIMATED, CONFIRMED, MEASURED - and only the first is reachable
   without the user saying something. These controls are how they say it. */

/* The server offers its estimate back rather than demanding a measurement.
   Rendered prefilled and editable: accepting is one tap, and correcting a
   wrong number is far easier than producing a right one from nothing. */
function renderDimensionProposal(proposal) {
  if (!proposal) return;
  state.proposal = proposal;
  $("confirm-question").textContent = proposal.question || "Is this about right?";
  $("confirm-width").value = Math.round(proposal.width_cm);
  $("confirm-depth").value = Math.round(proposal.depth_cm);

  // Naming what confirming buys is the difference between a chore and an
  // obvious win - "unlocks a bookshelf" beats "confirm dimensions".
  const unlocks = $("confirm-unlocks");
  if (proposal.unlocks?.length) {
    const names = proposal.unlocks.map(r => titleize(r));
    const list = names.length > 1
      ? names.slice(0, -1).join(", ") + " and " + names[names.length - 1]
      : names[0];
    unlocks.textContent = `Confirming lets us place your ${list}.`;
    unlocks.hidden = false;
  } else {
    unlocks.hidden = true;
  }

  $("panel-confirm").hidden = false;
  syncStages();
}

/* Send the confirmation. `measured` distinguishes "your estimate looks right"
   from "I got a tape measure out" - the backend holds tighter tolerances
   against the latter, so claiming it on the user's behalf would be a lie. */
function submitDimensions({ measured }) {
  const w = $("confirm-width").value.trim();
  const d = $("confirm-depth").value.trim();
  if (!w || !d || Number(w) <= 0 || Number(d) <= 0) {
    addBubble("Room Hack", "Please give a width and depth greater than zero.");
    return;
  }
  const irregular = $("confirm-irregular").checked;
  // Mirror the numbers into the brief so the two places never disagree.
  $("room-width").value = w;
  $("room-depth").value = d;
  briefSummary();

  const text = measured
    ? `My room measures exactly ${w}cm by ${d}cm.`
    : `Yes — about ${w}cm by ${d}cm is right.`;
  addBubble("You", esc(text), "user");
  $("panel-confirm").hidden = true;
  sendTurn(text, { width: w, depth: d, confirm: !measured, irregular, openings: true });
}

/* --- Openings --------------------------------------------------------------
   Doors and windows are the biggest accuracy win after dimensions: a door's
   swing is dead space, and a piece under a window has to clear the sill. */

const WALLS = ["north", "east", "south", "west"];

function renderOpenings() {
  const list = $("openings-list");
  if (!state.openings.length) {
    list.innerHTML = `<p class="openings-none">None added. The solver will assume clear walls.</p>`;
    $("apply-openings").hidden = true;
    return;
  }
  list.innerHTML = state.openings.map((o, i) => `
    <div class="opening" data-i="${i}">
      <span class="kind">${o.kind === "door" ? "Door" : "Window"}</span>
      <label class="mini"><span>Wall</span>
        <select data-f="wall">${WALLS.map(w =>
          `<option value="${w}" ${w === o.wall ? "selected" : ""}>${titleize(w)}</option>`).join("")}</select>
      </label>
      <label class="mini"><span>From left (cm)</span>
        <input data-f="offset_cm" type="number" min="0" step="1" value="${o.offset_cm}">
      </label>
      <label class="mini"><span>Width (cm)</span>
        <input data-f="width_cm" type="number" min="1" step="1" value="${o.width_cm}">
      </label>
      <button class="drop-it" data-drop="${i}" aria-label="Remove this ${o.kind}">✕</button>
    </div>`).join("");
  $("apply-openings").hidden = false;
}

function addOpening(kind) {
  state.openings.push({
    kind,
    wall: "north",
    offset_cm: 0,
    // A door's leaf sweeps roughly its own width, which is the clearance the
    // solver must keep free. Windows sweep nothing.
    width_cm: kind === "door" ? 80 : 120,
    swing_cm: kind === "door" ? 80 : 0,
  });
  renderOpenings();
}

/* --- Floor plan ----------------------------------------------------------- */
/* Placements are top-left origin in centimetres; the SVG viewBox is the room,
   so no manual scaling is needed. z=0 (rugs) must paint underneath. */
function renderFloorPlan(layout) {
  if (!layout) return;
  const W = layout.room_width_cm, D = layout.room_depth_cm;
  // z=0 (rugs) must paint underneath, but the legend should read in a stable
  // order regardless of stacking, so numbering is assigned before the sort.
  const placements = [...(layout.placements || [])];
  placements.forEach((p, i) => { p._n = i + 1; });
  const painted = [...placements].sort((a, b) => (a.z ?? 1) - (b.z ?? 1));

  /* A piece standing on a rug shares its centre, so centred markers collide -
     the table's number lands on top of the rug's. Each marker is nudged off any
     marker already placed, along the line between them, so both stay readable
     and each stays within its own footprint. */
  const marks = [];
  function markerAt(p, r) {
    let cx = p.x_cm + p.w_cm / 2;
    let cy = p.y_cm + p.d_cm / 2;
    for (let pass = 0; pass < 12; pass++) {
      const hit = marks.find(m => Math.hypot(m.cx - cx, m.cy - cy) < m.r + r + 4);
      if (!hit) break;
      const dx = cx - hit.cx, dy = cy - hit.cy;
      const len = Math.hypot(dx, dy) || 1;
      const push = hit.r + r + 6;
      // Push away from the marker already there; straight down when the two
      // centres coincide exactly, which is the common rug-and-table case.
      cx = hit.cx + (len < 0.01 ? 0 : dx / len) * push;
      cy = hit.cy + (len < 0.01 ? 1 : dy / len) * push;
    }
    // Keep the marker inside its own piece, so it never floats over a
    // neighbour and mislabels it.
    cx = Math.max(p.x_cm + r, Math.min(p.x_cm + p.w_cm - r, cx));
    cy = Math.max(p.y_cm + r, Math.min(p.y_cm + p.d_cm - r, cy));
    marks.push({ cx, cy, r });
    return { cx, cy };
  }

  const rects = painted.map(p => {
    const provisional = p.confidence && p.confidence !== "high";
    const label = `${p.name} — ${Math.round(p.w_cm)}×${Math.round(p.d_cm)}cm${provisional ? ` (±${Math.round(p.tolerance_cm)}cm, ${p.confidence} confidence)` : ""}${p.rationale ? `\n${p.rationale}` : ""}`;
    // One marker size for every piece. The old code scaled the text to the
    // footprint, so a rug shouted and a lamp was illegible; a fixed marker is
    // readable on both because it no longer has to contain a name.
    const r = Math.max(11, Math.min(17, Math.min(p.w_cm, p.d_cm) / 3.4));
    const { cx, cy } = markerAt(p, r);
    return `<g class="piece" data-n="${p._n}">
      <rect x="${p.x_cm}" y="${p.y_cm}" width="${p.w_cm}" height="${p.d_cm}" rx="4"
            fill="${esc(p.swatch || "#8B7355")}" fill-opacity="${p.z === 0 ? 0.5 : 0.92}"
            stroke="#26333f" stroke-width="1.5"
            stroke-dasharray="${provisional ? "7 5" : "0"}"><title>${esc(label)}</title></rect>
      <circle cx="${cx}" cy="${cy}" r="${r}" fill="#fdfcfa" fill-opacity=".92"
              stroke="#26333f" stroke-width="1.5" style="pointer-events:none"/>
      <text x="${cx}" y="${cy}" font-size="${r * 1.15}" text-anchor="middle"
            dominant-baseline="central" fill="#26333f"
            style="pointer-events:none;font-weight:700">${p._n}</text>
    </g>`;
  }).join("");

  $("floorplan").innerHTML =
    `<svg viewBox="-10 -10 ${W + 20} ${D + 20}" role="img"
          aria-label="Floor plan: ${placements.length} pieces in a ${Math.round(W)} by ${Math.round(D)} centimetre room. The numbered key is listed below.">
       <rect x="0" y="0" width="${W}" height="${D}" fill="#fdfcfa" stroke="#26333f" stroke-width="3"/>
       ${rects}
     </svg>`;

  // The legend carries every name, so nothing has to be written into the plan.
  $("plan-legend").innerHTML = `
    <div class="legend">
      ${placements.map(p => {
        const provisional = p.confidence && p.confidence !== "high";
        return `<button class="legend-item ${provisional ? "provisional" : ""}" data-n="${p._n}"
                        title="${esc(p.rationale || p.name)}">
          <span class="legend-key" style="background:${esc(p.swatch || "#8B7355")}">${p._n}</span>
          <span class="legend-name">${esc(p.name)}</span>
          <span class="legend-dim">${Math.round(p.w_cm)}×${Math.round(p.d_cm)}</span>
        </button>`;
      }).join("")}
    </div>
    <p class="legend-foot">${placements.length} piece${placements.length === 1 ? "" : "s"} placed in ${Math.round(W)}×${Math.round(D)}cm. A dashed key marks a provisional position.</p>`;

  // Hovering a legend row isolates its piece in the plan. Pointer only - the
  // pairing is already stated by the shared number for anyone not hovering.
  const svg = $("floorplan").querySelector("svg");
  const pieces = [...svg.querySelectorAll(".piece")];
  const focus = n => pieces.forEach(g => {
    const on = g.dataset.n === String(n);
    g.classList.toggle("lift", on);
    g.classList.toggle("dim", n != null && !on);
  });
  $("plan-legend").querySelectorAll(".legend-item").forEach(row => {
    row.addEventListener("pointerenter", () => focus(row.dataset.n));
    row.addEventListener("pointerleave", () => focus(null));
    row.addEventListener("focus", () => focus(row.dataset.n));
    row.addEventListener("blur", () => focus(null));
  });

  const skipped = layout.skipped || [];
  const withheld = layout.withheld || [];
  $("plan-notes").innerHTML = [
    ...skipped.map(s => `<div class="skip">Skipped ${esc(s.name)}: ${esc(s.reason)}</div>`),
    ...withheld.map(w => `<div class="skip">Withheld ${esc(w.name)}: ${esc(w.reason)}</div>`),
  ].join("");
  $("panel-plan").hidden = false;
  syncStages();
}

/* --- Cart ----------------------------------------------------------------- */
function renderCart(d) {
  const cart = d.cart || {};
  const lines = cart.lines || [];
  const subtotal = cart.subtotal_cents ?? 0;
  const budget = cart.budget_cents ?? 0;
  const over = d.over_budget ?? (budget > 0 && subtotal > budget);
  const pct = budget > 0 ? Math.min(100, (subtotal / budget) * 100) : 0;

  $("cart-lines").innerHTML = lines.map(l => `
    <div class="cart-line">
      <img src="${esc(assetUrl(l.image_url))}" alt="" loading="lazy" onerror="this.style.visibility='hidden'">
      <div class="cart-line-body">
        <strong>${esc(l.name)}</strong>
        <span class="muted">${esc(l.merchant)} · ${esc(titleize(l.role))}</span>
      </div>
      <span class="price">${money(l.price_cents * (l.qty || 1))}</span>
    </div>`).join("") || `<p class="muted">No items yet.</p>`;

  $("cart-total").innerHTML = `
    <div class="row"><span>Subtotal</span><strong>${money(subtotal)}</strong></div>
    ${budget > 0 ? `<div class="row"><span class="muted">Budget</span><span class="muted">${money(budget)}</span></div>
    <div class="bar"><i class="${over ? "over" : ""}" style="width:${pct}%"></i></div>
    ${over ? `<div class="skip">Over budget by ${money(subtotal - budget)}</div>` : ""}` : ""}`;
  $("panel-cart").hidden = false;
  updateActionAvailability();
  syncStages();
}

/* --- Bundles --------------------------------------------------------------
   Suggested sets. The `reason` comes from the server and states the actual
   basis - a shared IKEA range, shared styling - so it is shown verbatim
   rather than replaced with generic "customers also bought" copy the data
   does not support. */
function renderBundles(bundles) {
  state.bundles = bundles || [];
  if (!state.bundles.length) { $("panel-bundles").hidden = true; return; }

  $("bundles").innerHTML = state.bundles.map(b => `
    <div class="bundle">
      <div class="bundle-head">
        <span class="bundle-label">${esc(b.label)}</span>
        <span class="bundle-add">+${money(b.added_cents)}</span>
      </div>
      <p class="bundle-reason">${esc(b.reason)}</p>
      <div class="bundle-items">
        ${b.items.map(i => `
          <span class="bundle-piece ${i.is_new ? "" : "owned"}">
            <img src="${esc(assetUrl(i.image_url))}" alt="" loading="lazy" onerror="this.style.visibility='hidden'">
            <span>${i.is_new ? "" : "In cart · "}${esc(i.name)}</span>
          </span>`).join("")}
      </div>
      <div class="bundle-flags">
        ${b.affordable ? "" : `<span class="skip">Takes you over budget</span>`}
        ${b.fits_room ? "" : `<span class="skip">May not fit the room</span>`}
      </div>
    </div>`).join("");
  $("panel-bundles").hidden = false;
}

/* --- Alternatives (drive /api/swap) --------------------------------------- */
function renderAlternatives(options) {
  // A swap substitutes by role, so it only works for a role that actually made
  // it into the design. The server still offers alternatives for withheld
  // pieces, and clicking one of those could only ever 409.
  const placedRoles = new Set((state.cart?.lines || []).map(l => l.role));
  options = options.filter(o => placedRoles.has(o.role));
  if (!options.length) { $("panel-alts").hidden = true; return; }
  $("alts").innerHTML = options.map(opt => `
    <div class="alt-group">
      <h5>${esc(titleize(opt.role))}</h5>
      <div class="alt-row">
        ${(opt.alternatives || []).map(a => `
          <button class="alt ${a.affordable ? "" : "unaffordable"}"
                  data-role="${esc(opt.role)}" data-item="${esc(a.item_id)}"
                  title="${esc(a.materials?.join(", ") || "")} · ${Math.round(a.width_cm)}×${Math.round(a.depth_cm)}cm">
            <img src="${esc(assetUrl(a.image_url))}" alt="" loading="lazy" onerror="this.style.visibility='hidden'">
            <span class="alt-name">${esc(a.name)}</span>
            <span class="muted">${esc(a.merchant)}</span>
            <span class="delta ${a.price_delta_cents > 0 ? "up" : a.price_delta_cents < 0 ? "down" : ""}">
              ${a.price_delta_cents === 0 ? "same price" : (a.price_delta_cents > 0 ? "+" : "−") + money(Math.abs(a.price_delta_cents))}
            </span>
            ${a.affordable ? "" : `<span class="skip">over budget</span>`}
          </button>`).join("")}
      </div>
    </div>`).join("");
  $("panel-alts").hidden = false;
  syncStages();
}

/* --- Clarifications ------------------------------------------------------- */
/* The solver withholds wall-hugging pieces without real measurements. Asking
   here, with inputs wired to the measurement fields, is the way to unlock them. */
function renderClarifications(d) {
  const questions = d.questions || [];
  if (!questions.length) { $("panel-ask").hidden = true; return; }
  $("asks").innerHTML = questions.map(q => `
    <div class="ask">
      <p>${esc(q.question)}</p>
      ${q.affects?.length ? `<span class="muted">Affects: ${esc(q.affects.join(", "))}</span>` : ""}
    </div>`).join("");
  $("panel-ask").hidden = false;
}

/* --- Swap ----------------------------------------------------------------- */
/* A swap re-solves the layout and re-bills the cart, so it streams the same
   layout_update / cart_update frames the chat turn does. */
async function doSwap(role, itemId) {
  if (state.busy || !state.sessionId) return;
  setBusy(true);
  const status = setStatus("Swapping…");
  try {
    await streamSSE("/api/swap", {
      session_id: state.sessionId,
      role,
      item_id: itemId,
      // Skip the render here: the layout and price are what the user is
      // comparing, and a generative render costs tens of seconds.
      layout_only: true,
    }, {
      swap_started: d => {
        addBubble("Swap", `${esc(d.from.name)} → <strong>${esc(d.to.name)}</strong>`);
        // The standing render shows the OLD piece. Mark it rather than hiding
        // it: a labelled stale image is still the best reference for judging
        // the swap, and clearing it would leave the stage blank mid-decision.
        if (state.rendered) { state.renderStale = true; paintSwapRender(); }
      },
      layout_update: d => { state.layout = d.layout; renderFloorPlan(d.layout); },
      cart_update: d => { state.cart = d.cart; setCurrency(d.cart?.currency); renderCart(d); renderAlternatives(state.options); },
      alternatives: d => { state.options = d.options || []; renderAlternatives(state.options); },
      render_failed: d => addBubble("Render", `<span class="err">${esc(d.reason)}</span>`),
      error: d => addBubble("Error", `<span class="err">${esc(d.message)}</span>`),
    });
  } catch (err) {
    addBubble("Error", `<span class="err">${esc(err.message)}</span>`);
  } finally {
    status.clear();
    setBusy(false);
    paintSwapRender();
  }
}

/* --- Render --------------------------------------------------------------- */
/* Two shapes come back depending on the backend: `room_render` (one composed
   image of the whole room) or a stream of per-item `render_update` frames. */
/* --- Skeletons -----------------------------------------------------------
   A placeholder shaped like the content that is coming. Used where the wait
   is long enough that an empty panel reads as a broken one: the render is a
   model call (a minute or more on Replicate), and catalog matching is an
   embedding call per detected object.

   The point is not decoration - it is that the panel keeps its layout, so
   the arriving content does not shove the page around, and the user can see
   WHAT is loading rather than just THAT something is. */

/** Skeleton for the room visualization: one image block plus a caption line. */
function skeletonRender(label = "Painting your design into your room…") {
  return `
    <div class="sk-label muted"><span class="spin"></span> ${esc(label)}</div>
    <div class="sk sk-render"></div>
    <div class="sk sk-line w60"></div>`;
}

/** Drop any skeleton still sitting in a container, leaving real content alone.

    Needed because some streams APPEND their results (one frame per image),
    so the first real item would otherwise land underneath the placeholder
    and both would show at once. */
function clearSkeleton(id) {
  const el = $(id);
  if (!el) return;
  el.querySelectorAll(".sk, .sk-label, .sk-cards").forEach(n => n.remove());
}

/** Skeleton for a list of catalog matches: `n` thumbnail-plus-text rows. */
function skeletonCards(n = 3, label = "") {
  const card = `
    <div class="sk-card">
      <div class="sk sk-thumb"></div>
      <div>
        <div class="sk sk-line w60"></div>
        <div class="sk sk-line w40"></div>
      </div>
    </div>`;
  return `
    ${label ? `<div class="sk-label muted"><span class="spin"></span> ${esc(label)}</div>` : ""}
    <div class="sk-cards">${card.repeat(n)}</div>`;
}

async function doRender({ stay = false } = {}) {
  if (state.busy || !state.sessionId) return;
  setBusy(true);
  const status = setStatus("Rendering…");
  // A skeleton rather than an empty panel: this call runs to a minute or more,
  // and a blank stage for that long is indistinguishable from a failed one.
  // Not on a re-render from Swap - `stay` keeps the previous image up, which
  // is a better reference than a placeholder while judging a swap.
  $("renders").innerHTML = stay ? $("renders").innerHTML : skeletonRender();
  $("panel-render").hidden = false;
  // Re-rendering from the Swap stage: keep the previous image up while the new
  // one generates, so the stage does not go blank for the length of the call.
  if (stay) {
    $("rerender-swap").disabled = true;
    $("swap-render-note").textContent = "Updating…";
  }

  try {
    await streamSSE("/api/render", { session_id: state.sessionId, item_ids: [], per_item: false }, {
      render_started: d => {
        status.clear();
        setStatus(`Rendering ${d.total} ${d.total === 1 ? "image" : "images"} (${d.method})…`);
      },
      room_render: d => {
        state.rendered = true;
        state.renderStale = false;
        state.lastRenderHTML = `
          <img class="render-img" src="${esc(assetUrl(d.image_url))}" alt="Your room, visualized">
          <p class="muted note-line">${esc(d.disclaimer || "")}</p>
          ${d.omitted?.length ? `<div class="skip">Not shown: ${esc(d.omitted.join(", "))}</div>` : ""}
          ${d.replaced?.length ? `<p class="muted note-line">Removed from the photo: ${esc(d.replaced.join(", "))}</p>` : ""}`;
        $("renders").innerHTML = state.lastRenderHTML;
        paintSwapRender();
      },
      render_update: d => {
        const item = d.render || d;
        // The first per-item result replaces the skeleton; later ones append
        // to it. Without this the placeholder would sit above the real
        // images for the rest of the run.
        clearSkeleton("renders");
        $("renders").insertAdjacentHTML("beforeend", `
          <figure class="render-item">
            <img class="render-img" src="${esc(assetUrl(item.image_url))}" alt="${esc(item.name)}">
            <figcaption class="muted">${esc(item.name)}${item.replaced ? ` — replaces ${esc(item.replaced)}` : ""}</figcaption>
          </figure>`);
      },
      // no_photo is a prerequisite the user can act on, not an error in the
      // design, so it reads as an instruction rather than a failure - and it
      // is reported once for the whole design instead of once per piece.
      render_failed: d => {
        clearSkeleton("renders");
        if (d.reason === "no_photo") {
          $("renders").innerHTML =
            `<div class="skip">Add a room photo in your brief, then visualize again — `
            + `this paints your design into your own room.</div>`;
          return;
        }
        $("renders").insertAdjacentHTML("beforeend",
          `<div class="skip">${esc(d.name || "Render")} failed: ${esc(d.detail || d.reason)}</div>`);
      },
      error: d => addBubble("Error", `<span class="err">${esc(d.message)}</span>`),
    });
  } catch (err) {
    $("renders").innerHTML = `<div class="skip">${esc(err.message)}</div>`;
  } finally {
    status.clear();
    setBusy(false);
    $("swap-render-note").textContent =
      "Showing your previous design — swap something and update to see it.";
    paintSwapRender();
    syncStages();
  }
}

/* Staleness controls beside the visualization.

   A swap changes the design the render was made from, so the image on screen
   is now of the previous design. It deliberately does NOT refresh on its own:
   each render is a slow generative call and swapping is exploratory, so the
   old image stays up, is labelled as stale, and the user asks for a new one
   once they have settled. */
function paintSwapRender() {
  // Only offer the update once the design has actually moved on; otherwise the
  // button would re-render the identical scene at full generative cost.
  const stale = state.rendered && state.renderStale;
  $("rerender-swap").hidden = !stale;
  $("rerender-swap").disabled = state.busy;
  $("swap-render-note").hidden = !stale;
}

/* --- Payment (simulated Visa rail) ----------------------------------------
   The whole authorization happens in this sheet. The user is never redirected
   to a merchant, which is what lets the preview they read and the charge they
   approve be verifiably the same object: one payment intent, priced once by
   the server, displayed here, and authorized against its own total.

   The agent's authority stops at creating the intent. Everything past that
   point in this file is driven by a human gesture.                          */

const pay = {
  methods: [], intent: null, challenge: null, idem: null, busy: false,
  // merchant -> payment_method_id, as the user has overridden it. Persists
  // across re-pricing so changing one card does not reset the others.
  assignments: {},
  // --- Visa Agentic Stack ---------------------------------------------------
  // Registered passkeys, the live agent token, and the signed mandate the
  // agent presents when it prices an order. The mandate is a bearer
  // credential: holding it permits spending within its scope and nothing
  // wider, so it is kept in memory only and never persisted.
  credentials: [], token: null, mandateCredential: "",
  // The assertion id from the most recent Face ID / Touch ID check. Single-use
  // and bound to one intent and amount; cleared as soon as it is spent.
  assertion: null,
};

/* --- Visa Payment Passkey (FIDO2) ------------------------------------------
   Real WebAuthn. The private key lives in this device's secure enclave, the
   biometric never leaves the device, and what crosses the wire is a signature
   over a server-issued challenge bound to this exact amount.

   Why that matters more than the OTP it replaces: a code can be read aloud to
   someone on the phone, and it authorizes whatever the attacker is doing at
   the time. A passkey signature is bound to this origin - so it is worthless
   on a lookalike domain - and to this order's total, so it cannot be reused
   against a larger one.                                                     */

const b64urlToBuf = s => {
  const pad = "=".repeat((4 - (s.length % 4)) % 4);
  const bin = atob((s + pad).replace(/-/g, "+").replace(/_/g, "/"));
  return Uint8Array.from(bin, c => c.charCodeAt(0));
};
const bufToB64url = b =>
  btoa(String.fromCharCode(...new Uint8Array(b)))
    .replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");

/* Platform authenticators need a secure context. Checked up front so the UI
   can offer the OTP fallback rather than throwing an opaque DOMException at
   the moment the user expects Face ID. */
function passkeySupported() {
  return Boolean(window.PublicKeyCredential && window.isSecureContext);
}

async function loadCredentials() {
  try {
    const res = await fetch(api("/api/passkey/credentials"));
    pay.credentials = res.ok ? await res.json() : [];
  } catch { pay.credentials = []; }
  return pay.credentials;
}

/* Register this device as a payment passkey. */
async function registerPasskey(label) {
  if (!passkeySupported()) {
    throw new Error("This browser or context cannot use passkeys. Open the app over https:// or on localhost.");
  }
  const opts = await payFetch("/api/passkey/register/options", {});
  const credential = await navigator.credentials.create({
    publicKey: {
      ...opts,
      challenge: b64urlToBuf(opts.challenge),
      user: { ...opts.user, id: b64urlToBuf(opts.user.id) },
      excludeCredentials: (opts.excludeCredentials || []).map(c => ({ ...c, id: b64urlToBuf(c.id) })),
    },
  });
  if (!credential) throw new Error("No passkey was created.");
  const summary = await payFetch("/api/passkey/register", {
    credential_id: bufToB64url(credential.rawId),
    client_data_json: bufToB64url(credential.response.clientDataJSON),
    attestation_object: bufToB64url(credential.response.attestationObject),
    transports: credential.response.getTransports?.() || [],
    label: label || "This device",
  });
  await loadCredentials();
  return summary;
}

/* Prove the cardholder is present, for one specific payment.
   `intentId` is null when provisioning a mandate rather than paying. */
async function assertPasskey(intentId, purpose = "payment") {
  const opts = await payFetch("/api/passkey/challenge", {
    intent_id: intentId, purpose,
  });
  const assertion = await navigator.credentials.get({
    publicKey: {
      ...opts,
      challenge: b64urlToBuf(opts.challenge),
      allowCredentials: (opts.allowCredentials || []).map(c => ({ ...c, id: b64urlToBuf(c.id) })),
    },
  });
  if (!assertion) throw new Error("Verification was cancelled.");
  return await payFetch("/api/passkey/verify", {
    credential_id: bufToB64url(assertion.rawId),
    client_data_json: bufToB64url(assertion.response.clientDataJSON),
    authenticator_data: bufToB64url(assertion.response.authenticatorData),
    signature: bufToB64url(assertion.response.signature),
    intent_id: intentId,
    purpose,
  });
}

/* Translate the WebAuthn DOMException names into something a person can act
   on. The raw messages are written for developers and say nothing useful to
   someone who just cancelled a Face ID prompt. */
function passkeyErrorMessage(err) {
  switch (err?.name) {
    case "NotAllowedError":
      return "Verification was cancelled or timed out. Try again when you're ready.";
    case "InvalidStateError":
      return "This device already has a passkey for Room Hack.";
    case "SecurityError":
      return "Passkeys need a secure connection. Open the app on localhost or over https://.";
    case "NotSupportedError":
      return "This device has no biometric authenticator available.";
    default:
      return err?.message || "Could not verify with this device.";
  }
}

/* --- Agent mandate --------------------------------------------------------- */

async function loadAgentToken() {
  try {
    const res = await fetch(api("/api/agent-token"));
    const tokens = res.ok ? await res.json() : [];
    // The live one, if any. Revoked tokens stay in the list for the audit
    // trail but must never be picked up as the active mandate.
    pay.token = tokens.find(t => t.status === "active") || null;
    if (!pay.token) pay.mandateCredential = "";
    return tokens;
  } catch { return []; }
}

/* Grant the agent a scoped, revocable spending mandate. */
async function provisionAgentToken({ fundingMethodId, perTxnCents, totalCents, ttlHours }) {
  let assertionId = "";
  if (pay.credentials.length) {
    const verified = await assertPasskey(null, "provisioning");
    assertionId = verified.assertion_id;
  }
  const result = await payFetch("/api/agent-token/provision", {
    funding_method_id: fundingMethodId,
    per_transaction_cap_cents: perTxnCents,
    cumulative_cap_cents: totalCents,
    ttl_hours: ttlHours,
    assertion_id: assertionId,
  });
  pay.token = result.token;
  // Returned exactly once. Held in memory for this session only.
  pay.mandateCredential = result.mandate_credential;
  return result;
}

/* The kill switch. Ends the agent's authority; the card keeps working. */
async function revokeAgentToken(reason) {
  if (!pay.token) return null;
  const revoked = await payFetch("/api/agent-token/revoke", {
    token_id: pay.token.token_id,
    reason: reason || "revoked by user",
  });
  pay.token = null;
  pay.mandateCredential = "";
  return revoked;
}
function payOpen() { $("pay-overlay").hidden = false; }
function payClose() {
  $("pay-overlay").hidden = true;
  pay.intent = null; pay.challenge = null; pay.idem = null;
  pay.assertion = null;
}
function payStep(active) {
  // 1 review -> 2 verify -> 3 authorize. The verify step is dropped entirely
  // when the server did not ask for one, rather than shown as skipped.
  const steps = [["review", "Review"], ["verify", "Verify"], ["done", "Authorize"]];
  const shown = pay.intent?.requires_step_up ? steps : steps.filter(x => x[0] !== "verify");
  const at = shown.findIndex(x => x[0] === active);
  return `<div class="pay-steps">${shown.map((sx, i) =>
    `<span class="${i < at ? "done" : i === at ? "on" : ""}"><b>${i < at ? "✓" : i + 1}</b>${sx[1]}</span>` +
    (i < shown.length - 1 ? "<i></i>" : "")).join("")}</div>`;
}

async function payFetch(path, body) {
  const res = await fetch(api(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || res.statusText);
  return data;
}

/* Step 0: the agent prices the basket. Charges nothing. */
async function startPayment() {
  const lines = state.cart?.lines || [];
  if (!lines.length) return;
  setBusy(true);
  const status = setStatus("Pricing your order…");
  try {
    if (!pay.methods.length) {
      const res = await fetch(api("/api/payment/methods"));
      pay.methods = await res.json();
    }
    // Load the passkey and mandate state before pricing, so the preview can
    // say which guardrails applied rather than discovering them at authorize
    // time - a limit the user only learns about when it blocks them is a
    // worse experience than no limit at all.
    await Promise.all([loadCredentials(), loadAgentToken()]);
    pay.intent = await payFetch("/api/payment/intent", {
      item_ids: lines.map(l => l.item_id),
      session_id: state.sessionId,
      payment_method_ids: pay.assignments,
      mandate_credential: pay.mandateCredential,
    });
    // The intent prices the purchase server-side, so it - not the cart - is
    // the last word on what the user is actually being charged in.
    setCurrency(pay.intent?.currency);
    // A fresh key per preview: retries of THIS authorization are safe, while a
    // genuinely new purchase gets a new key and is not swallowed as a replay.
    pay.idem = `idem_${Date.now()}_${Math.random().toString(16).slice(2, 10)}`;
    renderPreview();
    payOpen();
  } catch (err) {
    alert("Could not price this order: " + err.message);
  } finally {
    status.clear(); setBusy(false);
  }
}

/* Grant the agent a mandate. Every field here is a limit the user chooses,
   and the defaults are deliberately modest - a default nobody thinks about
   should be the small one. */
async function openMandateDialog(onDone) {
  // A mandate is only as strong as the proof behind it, so enrol a passkey
  // first if this device has none.
  if (!pay.credentials.length && passkeySupported()) {
    return renderPasskeySetup(() => openMandateDialog(onDone));
  }
  let defaults = { per_transaction_cap_cents: 200000, cumulative_cap_cents: 500000, ttl_hours: 24 };
  try {
    const res = await fetch(api("/api/agent-token/defaults"));
    if (res.ok) defaults = await res.json();
  } catch { /* fall back to the values above */ }

  const cardOptions = pay.methods.map(m =>
    `<option value="${esc(m.id)}"${m.is_default ? " selected" : ""}>${esc(m.network.toUpperCase())} ···· ${esc(m.last4)} — ${esc(m.label)}</option>`).join("");

  $("pay-title").textContent = "Agent spending limits";
  $("pay-sub").textContent = "You can change or revoke these at any time";
  $("pay-body").innerHTML = `
    <p class="muted">Room Hack's agent will get a <strong>Visa network token</strong>,
    not your card number. It can only spend inside the limits you set here, and
    revoking it later leaves your card working normally.</p>
    <label class="fld"><span>Card to draw on</span>
      <select id="md-card">${cardOptions}</select></label>
    <label class="fld"><span>Most it can spend in one purchase</span>
      <input id="md-txn" type="number" min="1" step="1" value="${(defaults.per_transaction_cap_cents / 100).toFixed(0)}"></label>
    <label class="fld"><span>Most it can spend in total</span>
      <input id="md-total" type="number" min="1" step="1" value="${(defaults.cumulative_cap_cents / 100).toFixed(0)}"></label>
    <label class="fld"><span>Mandate expires after</span>
      <select id="md-ttl">
        <option value="1">1 hour</option>
        <option value="24" selected>24 hours</option>
        <option value="168">7 days</option>
      </select></label>
    <div class="risk ok"><span class="risk-ico">✓</span><span>Locked to
    <strong>${esc(defaults.category_label || "Furniture & Home Decor")}</strong>
    merchants. The agent cannot spend this token anywhere else.</span></div>
    <div id="md-err"></div>`;
  $("pay-foot").innerHTML = `
    <button class="primary" id="md-go">${pay.credentials.length ? "Verify and grant mandate" : "Grant mandate"}</button>
    <button class="linkish" id="md-cancel">Not now</button>
    <div class="pay-secure">🔒 Granting a mandate charges nothing.</div>`;

  $("md-cancel").onclick = onDone;
  $("md-go").onclick = async () => {
    const perTxn = Math.round(parseFloat($("md-txn").value) * 100);
    const total = Math.round(parseFloat($("md-total").value) * 100);
    const err = m => { $("md-err").innerHTML = `<div class="skip">${esc(m)}</div>`; };
    if (!(perTxn > 0) || !(total > 0)) return err("Both limits must be greater than zero.");
    // Caught here as well as server-side so the user is told immediately
    // rather than after a round trip.
    if (perTxn > total) return err("The per-purchase limit cannot exceed the total limit.");
    $("md-go").disabled = true;
    $("md-err").innerHTML = "";
    try {
      await provisionAgentToken({
        fundingMethodId: $("md-card").value,
        perTxnCents: perTxn, totalCents: total,
        ttlHours: parseInt($("md-ttl").value, 10),
      });
      await onDone();
    } catch (e) {
      err(passkeyErrorMessage(e));
      $("md-go").disabled = false;
    }
  };
}

/* The agent's spending mandate, rendered as a live meter rather than a
   paragraph. A cap the user cannot see the remaining headroom on is one they
   have to take on trust, which is the opposite of the point. */
function renderMandatePanel() {
  const t = pay.token;
  if (!t) {
    return `<div class="mandate mandate-none">
      <div class="mandate-head"><strong>No agent mandate</strong></div>
      <p class="muted">You're paying with your card directly. Grant the agent a
      scoped Visa token to let it assemble orders under limits you set — and
      revoke it at any time without cancelling the card.</p>
      <button class="secondary" id="grant-mandate">Set up agent spending limits</button>
    </div>`;
  }
  const s = t.scope;
  const pct = s.cumulative_cap_cents
    ? Math.min(100, Math.round((s.spent_cents / s.cumulative_cap_cents) * 100)) : 0;
  const hours = Math.max(0, Math.round((s.expires_at * 1000 - Date.now()) / 3600000));
  return `<div class="mandate">
    <div class="mandate-head">
      <strong>Agent mandate</strong>
      <span class="card-pill token">VISA TOKEN ···· ${esc(t.token_last4)}</span>
      <span class="card-pill">${esc(t.presentation_type)}</span>
    </div>
    <div class="mandate-grid">
      <div><span class="muted">Per purchase</span><strong>${money(s.per_transaction_cap_cents)}</strong></div>
      <div><span class="muted">Category</span><strong>${esc(s.category_label)}</strong></div>
      <div><span class="muted">Expires</span><strong>in ${hours}h</strong></div>
    </div>
    <div class="meter"><span style="width:${pct}%"></span></div>
    <div class="mandate-foot">
      <span class="muted">${money(s.spent_cents)} of ${money(s.cumulative_cap_cents)} used · ${money(s.remaining_cents)} left</span>
      <button class="linkish" id="revoke-mandate">Revoke</button>
    </div>
    <p class="mandate-note">Your 16-digit card number is never shared with any
    merchant. Revoking stops this agent only — your card keeps working.</p>
  </div>`;
}

/* Wire the mandate panel's two buttons. Called after every render that
   includes the panel, since innerHTML replaces the nodes each time. */
function wireMandatePanel(onChange) {
  const grant = $("grant-mandate");
  if (grant) grant.onclick = () => openMandateDialog(onChange);
  const revoke = $("revoke-mandate");
  if (revoke) revoke.onclick = async () => {
    if (!confirm("Revoke the agent's spending mandate?\n\nYour card is not affected — only this agent's ability to spend on it.")) return;
    try { await revokeAgentToken("revoked from checkout"); await onChange(); }
    catch (err) { alert("Could not revoke: " + err.message); }
  };
}

/* Step 1: the transaction preview. Everything the user is agreeing to, before
   they agree to it - which merchant, which item, which card, what it costs,
   when it arrives, and why they are being asked to verify. */
function renderPreview() {
  const it = pay.intent;
  const cardOpts = m => pay.methods.map(pm =>
    `<option value="${esc(pm.id)}" ${pm.id === m ? "selected" : ""}>${esc(pm.network.toUpperCase())} ···· ${esc(pm.last4)} — ${esc(pm.label)}</option>`).join("");

  const charges = it.charges.map(c => `
    <div class="charge">
      <div class="charge-head"><strong>${esc(c.merchant)}</strong><span class="amt">${money(c.total_cents)}</span></div>
      ${c.lines.map(l => `
        <div class="charge-line">
          <img src="${esc(assetUrl(l.image_url))}" alt="" loading="lazy" onerror="this.style.visibility='hidden'">
          <div><span>${esc(l.name)}</span><br><span class="muted">${esc(titleize(l.role))}</span></div>
          <span>${money(l.price_cents * (l.qty || 1))}</span>
        </div>`).join("")}
      <div class="row"><span class="muted">Shipping</span><span class="muted">${c.shipping_cents ? money(c.shipping_cents) : "Free"}</span></div>
      <div class="row"><span class="muted">Tax</span><span class="muted">${money(c.tax_cents)}</span></div>
      <div class="charge-meta">
        <span>Pay with</span>
        <select data-merchant="${esc(c.merchant)}" class="pay-card-select">${cardOpts(c.payment_method_id)}</select>
        ${c.token_last4 ? `<span class="card-pill token">token ···· ${esc(c.token_last4)}</span>` : ""}
        ${c.category_label ? `<span class="card-pill mcc" title="Merchant category — what the mandate's category lock is checked against${c.mcc ? ` (MCC ${esc(c.mcc)})` : ""}">${esc(c.category_label)}</span>` : ""}
        <span>· arrives in ~${c.eta_days} days</span>
      </div>
    </div>`).join("");

  const risk = it.risk.map(r => `
    <div class="risk ${r.triggers_step_up ? "alert" : r.code === "routine" ? "ok" : ""}">
      <span class="risk-ico">${r.triggers_step_up ? "!" : r.code === "routine" ? "✓" : "i"}</span>
      <span>${esc(r.detail)}</span>
    </div>`).join("");

  $("pay-title").textContent = "Confirm your purchase";
  $("pay-sub").textContent = `${it.merchant_count ?? it.charges.length} merchant${it.charges.length > 1 ? "s" : ""} · nothing charged yet`;
  const blocked = it.mandate_blocked;
  $("pay-body").innerHTML = `
    ${payStep("review")}
    <div class="banner">${esc(it.disclaimer)}</div>
    ${renderMandatePanel()}
    ${blocked ? `<div class="mandate-blocked">
      <strong>Outside the agent's mandate</strong>
      <p>This order breaks a limit you set, so the agent cannot charge it.
      Raise the limit, remove the item, or pay for it yourself.</p>
    </div>` : ""}
    ${charges}
    ${risk}
    <div class="row"><span class="muted">Subtotal</span><span>${money(it.subtotal_cents)}</span></div>
    <div class="row"><span class="muted">Shipping</span><span>${it.shipping_cents ? money(it.shipping_cents) : "Free"}</span></div>
    <div class="row"><span class="muted">Tax</span><span>${money(it.tax_cents)}</span></div>
    <div class="row total"><strong>Total</strong><strong>${money(it.total_cents)}</strong></div>
    ${it.over_budget ? `<div class="skip">This is over the ${money(it.budget_cents)} budget you set.</div>` : ""}`;

  const usePasskey = it.step_up_method === "passkey" && pay.credentials.length && passkeySupported();
  const nextLabel = blocked
    ? "Blocked by your spending limits"
    : it.requires_step_up
      ? (usePasskey ? "Verify with Face ID / Touch ID" : "Verify identity to continue")
      : "Continue to authorize";
  $("pay-foot").innerHTML = `
    <button class="primary" id="pay-next"${blocked ? " disabled" : ""}>${esc(nextLabel)}</button>
    <div class="pay-secure">🔒 You are still on Room Hack — no redirect. Cancel any time before you authorize.</div>`;

  // Changing a card re-prices server-side: a different card can change which
  // limits apply, so the risk assessment must be recomputed, not patched here.
  $("pay-body").querySelectorAll(".pay-card-select").forEach(sel => {
    sel.onchange = async () => {
      pay.assignments[sel.dataset.merchant] = sel.value;
      await repriceIntent();
    };
  });
  wireMandatePanel(repriceIntent);
  $("pay-next").onclick = () => {
    if (blocked) return;
    if (!it.requires_step_up) return renderAuthorize();
    // Passkey when the device has one; the OTP path survives only as a
    // fallback for a device with no authenticator.
    return usePasskey ? beginPasskeyVerify() : beginVerify();
  };
}

async function repriceIntent() {
  try {
    pay.intent = await payFetch("/api/payment/intent", {
      item_ids: (state.cart?.lines || []).map(l => l.item_id),
      session_id: state.sessionId,
      payment_method_ids: pay.assignments,
      mandate_credential: pay.mandateCredential,
    });
    pay.challenge = null;
    // A new preview invalidates any identity check: the assertion was bound
    // to the old total and the server will refuse it against the new one.
    pay.assertion = null;
    pay.idem = `idem_${Date.now()}_${Math.random().toString(16).slice(2, 10)}`;
    renderPreview();
  } catch (err) {
    $("pay-body").innerHTML = `<div class="skip">${esc(err.message)}</div>`;
  }
}

/* Step 2 (preferred): Visa Payment Passkey. The device's own biometric
   unlocks a hardware-held key and signs a challenge bound to this exact
   order and total. Nothing about the user's face or fingerprint leaves the
   device - the server only ever sees a signature. */
function renderPasskeyPrompt(error, busy) {
  const it = pay.intent;
  $("pay-title").textContent = "Verify it's you";
  $("pay-sub").textContent = "Face ID, Touch ID or your device PIN";
  $("pay-body").innerHTML = `
    ${payStep("verify")}
    <div class="passkey-hero">
      <div class="passkey-ico${busy ? " pulsing" : ""}" aria-hidden="true">☰</div>
      <strong>Confirm ${money(it.total_cents)} with your device</strong>
      <p class="muted">Your passkey is stored in this device's secure chip. Your
      biometrics never leave the device and are never sent to Room Hack, the
      merchants, or Visa — only a signature is.</p>
    </div>
    <div class="risk ok"><span class="risk-ico">🔒</span><span>This signature is
    locked to <strong>${money(it.total_cents)}</strong> and to this order. It
    cannot be reused for a different amount.</span></div>
    ${error ? `<div class="skip">${esc(error)}</div>` : ""}`;
  $("pay-foot").innerHTML = `
    <button class="primary" id="passkey-go"${busy ? " disabled" : ""}>
      ${busy ? '<span class="spin"></span> Waiting for your device…' : "Verify with Face ID / Touch ID"}
    </button>
    ${pay.credentials.length ? `<button class="linkish" id="use-otp">Use a one-time code instead</button>` : ""}
    <div class="pay-secure">🔒 Still nothing charged.</div>`;
  const go = $("passkey-go");
  if (go && !busy) go.onclick = runPasskeyVerify;
  const otp = $("use-otp");
  if (otp) otp.onclick = beginVerify;
}

async function beginPasskeyVerify() {
  renderPasskeyPrompt();
  // Fire immediately: the user already pressed a button meaning "verify me",
  // and making them press a second one to reach the same prompt is friction
  // with no security value.
  await runPasskeyVerify();
}

async function runPasskeyVerify() {
  renderPasskeyPrompt(null, true);
  try {
    const verified = await assertPasskey(pay.intent.id, "payment");
    pay.assertion = verified.assertion_id;
    // Re-read the intent: verifying cleared the step-up server-side, and the
    // authorize screen should reflect the server's view, not assume it.
    try {
      const res = await fetch(api(`/api/payment/intent/${encodeURIComponent(pay.intent.id)}`));
      if (res.ok) pay.intent = await res.json();
    } catch { /* the authorize call re-validates anyway */ }
    renderAuthorize();
  } catch (err) {
    renderPasskeyPrompt(passkeyErrorMessage(err));
  }
}

/* Enrol this device as a payment passkey. Offered when none is registered. */
async function renderPasskeySetup(onDone) {
  $("pay-title").textContent = "Set up your payment passkey";
  $("pay-sub").textContent = "One-time setup on this device";
  $("pay-body").innerHTML = `
    <div class="passkey-hero">
      <div class="passkey-ico" aria-hidden="true">☰</div>
      <strong>Replace codes with your face or fingerprint</strong>
      <p class="muted">A passkey lives in this device's secure chip. It can't be
      phished, texted to someone else, or reused on a fake site — and it proves
      the card is yours without your card number ever being shared.</p>
    </div>`;
  $("pay-foot").innerHTML = `
    <button class="primary" id="pk-create">Create passkey</button>
    <div class="pay-secure">🔒 Nothing is charged during setup.</div>`;
  $("pk-create").onclick = async () => {
    $("pk-create").disabled = true;
    try { await registerPasskey("This device"); await onDone(); }
    catch (err) {
      $("pay-body").innerHTML += `<div class="skip">${esc(passkeyErrorMessage(err))}</div>`;
      $("pk-create").disabled = false;
    }
  };
}

/* Step 2 (fallback): step-up in the shape of a 3-D Secure OTP prompt. Reached
   only when the device has no authenticator, or the user asks for it. */
async function beginVerify() {
  try {
    pay.challenge = await payFetch("/api/payment/verify/start", { intent_id: pay.intent.id });
    renderVerify();
  } catch (err) {
    $("pay-body").innerHTML = `<div class="skip">${esc(err.message)}</div>`;
  }
}

function renderVerify(error) {
  const ch = pay.challenge;
  $("pay-title").textContent = "Verify it's you";
  $("pay-sub").textContent = "Extra check before this can be charged";
  $("pay-body").innerHTML = `
    ${payStep("verify")}
    <p class="muted">We sent a 6-digit code to <strong>${esc(ch.sent_to)}</strong>. Enter it to authorize a purchase this size.</p>
    <input id="otp" class="otp-input" inputmode="numeric" maxlength="6" placeholder="······" aria-label="6-digit verification code">
    ${error ? `<div class="skip">${esc(error)}</div>` : ""}
    <div class="demo-code">Demo only — a real check never shows you the code. Yours is <code>${esc(ch.demo_code)}</code></div>`;
  $("pay-foot").innerHTML = `
    <button class="primary" id="otp-submit">Verify</button>
    <div class="pay-secure">🔒 Still nothing charged.</div>`;
  const submit = async () => {
    const code = $("otp").value.trim();
    if (code.length !== 6) return renderVerify("Enter all 6 digits.");
    $("otp-submit").disabled = true;
    try {
      pay.intent = await payFetch("/api/payment/verify", {
        intent_id: pay.intent.id, challenge_id: ch.challenge_id, code,
      });
      renderAuthorize();
    } catch (err) {
      renderVerify(err.message);
    }
  };
  $("otp-submit").onclick = submit;
  $("otp").addEventListener("keydown", e => { if (e.key === "Enter") submit(); });
  $("otp").focus();
}

/* Step 3: the authorization itself. Hold-to-confirm rather than a click, so
   money cannot move on a stray tap, and the exact total is echoed back to the
   server - if it drifted since this screen rendered, the charge is refused. */
function renderAuthorize(error) {
  const it = pay.intent;
  $("pay-title").textContent = "Authorize payment";
  $("pay-sub").textContent = "This is the step that charges your card";
  $("pay-body").innerHTML = `
    ${payStep("done")}
    ${pay.assertion
      ? `<div class="risk ok"><span class="risk-ico">✓</span><span>Verified by
         your device biometric. This approval is locked to
         <strong>${money(it.total_cents)}</strong> and expires in a few minutes.</span></div>`
      : it.requires_step_up
        ? `<div class="risk ok"><span class="risk-ico">✓</span><span>Identity verified.</span></div>`
        : ""}
    ${it.charges.map(c => {
      const m = pay.methods.find(x => x.id === c.payment_method_id);
      // Name the token when one is presented: the merchant is charged against
      // the token, and the card sits behind it. Showing only the card would
      // misrepresent what actually travels.
      const instrument = c.token_last4
        ? `<span class="card-pill token">TOKEN ···· ${esc(c.token_last4)}</span>`
        : `<span class="card-pill">${esc((m?.network || "visa").toUpperCase())} ···· ${esc(m?.last4 || "")}</span>`;
      return `<div class="row"><span>${esc(c.merchant)} ${instrument}</span><strong>${money(c.total_cents)}</strong></div>`;
    }).join("")}
    <div class="row total"><strong>You are authorizing</strong><strong>${money(it.total_cents)}</strong></div>
    ${error ? `<div class="skip">${esc(error)}</div>` : ""}
    <p class="muted" style="margin-top:12px">Room Hack's agent assembled this order. Holding the button below is your instruction to charge it — the agent cannot do this for you.</p>
    ${it.agent_token_id ? `<p class="muted mandate-note">Charged through Visa agent
    token <code>${esc(it.agent_token_id)}</code>. Your card number is not shared
    with ${it.charges.length === 1 ? "this merchant" : "any of these merchants"}.</p>` : ""}`;
  $("pay-foot").innerHTML = `
    <button class="primary hold-btn" id="authorize"><span class="hold-fill" id="hold-fill"></span><span>Hold to pay ${money(it.total_cents)}</span></button>
    <div class="pay-secure">🔒 <span class="brand">VISA</span> secure checkout · simulated</div>`;
  wireHold($("authorize"), doAuthorize);
}

/* An 850ms deliberate press. Short enough not to annoy, long enough that it
   cannot happen by accident. Releasing early cancels with nothing charged. */
function wireHold(btn, onComplete) {
  let raf = null, t0 = 0;
  const HOLD_MS = 850;
  // Looked up per frame rather than captured: re-rendering the footer (an
  // error, a retry) replaces this node, and a stale reference would leave the
  // progress bar frozen while the hold still counted down invisibly.
  const stop = () => {
    if (raf) cancelAnimationFrame(raf);
    raf = null;
    const f = $("hold-fill"); if (f) f.style.width = "0%";
  };
  const tick = () => {
    const pct = Math.min(1, (performance.now() - t0) / HOLD_MS);
    const f = $("hold-fill"); if (f) f.style.width = (pct * 100) + "%";
    if (pct >= 1) { stop(); onComplete(); return; }
    raf = requestAnimationFrame(tick);
  };
  const begin = e => { e.preventDefault(); if (pay.busy) return; t0 = performance.now(); raf = requestAnimationFrame(tick); };
  btn.addEventListener("pointerdown", begin);
  btn.addEventListener("pointerup", stop);
  btn.addEventListener("pointerleave", stop);
  btn.addEventListener("pointercancel", stop);
  // Keyboard equivalent: a hold gesture nobody can perform is not a safeguard,
  // it is a lockout. Enter/Space authorizes directly.
  btn.addEventListener("keydown", e => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onComplete(); }
  });
}

async function doAuthorize() {
  if (pay.busy) return;
  pay.busy = true;
  $("pay-body").innerHTML = `${payStep("done")}<p class="muted"><span class="spin"></span> Authorizing with your bank…</p>`;
  $("pay-foot").innerHTML = `<div class="pay-secure">Do not close this window.</div>`;
  try {
    const receipt = await payFetch("/api/payment/authorize", {
      intent_id: pay.intent.id,
      idempotency_key: pay.idem,
      confirmed_total_cents: pay.intent.total_cents,
      assertion_id: pay.assertion || "",
    });
    // Spent. The server refuses it a second time regardless, but holding a
    // dead assertion in the client only invites confusing errors.
    pay.assertion = null;
    renderReceipt(receipt);
  } catch (err) {
    pay.busy = false;
    renderAuthorize(err.message);
  } finally {
    pay.busy = false;
  }
}

/* Step 4: the receipt, including anything that did NOT go through. A partial
   failure is the case a checkout most needs to be honest about. */
function renderReceipt(r) {
  const declined = r.charges.filter(c => c.status === "declined");
  $("pay-title").textContent = declined.length ? "Partly completed" : "Order confirmed";
  $("pay-sub").textContent = `Order ${r.order_id}`;
  $("pay-body").innerHTML = `
    <div class="banner">${esc(r.disclaimer)}</div>
    ${r.charges.map(c => {
      const m = pay.methods.find(x => x.id === c.payment_method_id);
      const ok = c.status === "approved";
      return `<div class="charge">
        <div class="charge-head"><strong>${esc(c.merchant)}</strong><span class="amt">${money(c.total_cents)}</span></div>
        <div class="charge-meta">
          <span class="card-pill ${ok ? "approved" : "declined"}">${ok ? "Approved" : "Declined"}</span>
          ${c.token_last4
            ? `<span class="card-pill token">TOKEN ···· ${esc(c.token_last4)}</span>`
            : `<span class="card-pill">${esc((m?.network || "visa").toUpperCase())} ···· ${esc(m?.last4 || "")}</span>`}
          ${c.cryptogram_id ? `<span class="card-pill mcc" title="Single-use cryptogram for this merchant leg">${esc(c.cryptogram_id)}</span>` : ""}
          ${ok ? `<span>${esc(c.auth_code)}</span><span>· arrives in ~${c.eta_days} days</span>`
               : `<span>${esc(c.decline_reason)}</span>`}
        </div></div>`;
    }).join("")}
    <div class="row"><span class="muted">Charged</span><strong>${money(r.approved_cents)}</strong></div>
    ${r.declined_cents ? `<div class="row"><span class="muted">Not charged</span><span class="skip">${money(r.declined_cents)}</span></div>` : ""}
    <div class="audit"><strong>Authorization record</strong>${r.audit.map(a => `<div>· ${esc(a)}</div>`).join("")}</div>`;
  $("pay-foot").innerHTML = `<button class="secondary" id="pay-done">Done</button>`;
  $("pay-done").onclick = () => { payClose(); renderOrderSummary(r); };
}

/* Mirror the receipt into the page behind the sheet, so it survives closing. */
function renderOrderSummary(r) {
  $("checkout").innerHTML = `
    <div class="banner">${esc(r.disclaimer)}</div>
    <div class="row"><span class="muted">Order</span><span>${esc(r.order_id)}</span></div>
    ${r.charges.map(c => `
      <div class="merchant">
        <div class="row"><strong>${esc(c.merchant)}</strong><strong>${money(c.total_cents)}</strong></div>
        <div class="row"><span class="muted">${c.status === "approved" ? esc(c.auth_code) : "Declined — not charged"}</span>
        <span class="muted">${c.status === "approved" ? `~${c.eta_days} days` : ""}</span></div>
      </div>`).join("")}
    <div class="row total"><strong>Charged</strong><strong>${money(r.approved_cents)}</strong></div>`;
  $("panel-checkout").hidden = false;
}

async function cancelPayment() {
  const id = pay.intent?.id;
  payClose();
  // Tell the server too, so the intent cannot be authorized later by a stale
  // client holding the same id.
  if (id) { try { await payFetch("/api/payment/cancel", { intent_id: id }); } catch {} }
}

/* --- Health --------------------------------------------------------------- */
async function checkHealth() {
  try {
    const res = await fetch(api("/api/health"));
    const h = await res.json();
    const mode = h.offline_mode ? "offline mock" : h.providers;
    $("health").innerHTML = `<span class="dot ${h.status === "ok" ? "ok" : "warn"}"></span>${esc(mode)} · ${h.renderer?.can_compose ? "photo render" : "schematic"}`;
  } catch {
    $("health").innerHTML = `<span class="dot err"></span>API offline — start the backend on ${esc(API)}`;
  }
}

function updateActionAvailability() {
  const hasDesign = !!(state.sessionId && state.cart?.lines?.length);
  // Visualizing composes the furniture into the user's own room photo, so
  // without one there is nothing to paint into and the server returns
  // no_photo. Say that up front instead of letting the click fail: the
  // requirement is not guessable from a button labelled "Visualize".
  const canRender = hasDesign && !!state.roomPhoto;
  const render = $("do-render");
  render.disabled = !canRender || state.busy;
  render.title = !hasDesign
    ? "Generate a layout first"
    : !state.roomPhoto
      ? "Add a room photo in your brief — the visualization paints your design into your own room"
      : "";
  const hint = $("render-hint");
  if (hint) hint.hidden = !hasDesign || !!state.roomPhoto;
}


/* --- The sequence ---------------------------------------------------------
   The workspace is a four-stage sequence - brief, identify, design, checkout
   - not a set of tabs. Two rules govern it:

   1. A stage unlocks when it has real content behind it. `syncStages` is the
      single place that decides this, called from the render functions as they
      fill their panels, so a panel becoming reachable is never spread across
      the file.
   2. Once unlocked, a stage stays reachable in both directions. Swapping a
      piece legitimately sends you back to the layout, so a one-way flow would
      trap the user in the stage they were trying to leave.

   The brief is always reachable - it is where the design starts, and editing
   it is how you correct a room the analysis got wrong. */

const STAGES = ["brief", "identify", "design", "checkout"];

/* What each stage offers as its forward action. `next` is the stage the button
   advances to; `action` runs instead when the button does work rather than
   just navigating. A stage whose `next` is unreachable falls back to running
   its action, which is what makes "Generate my layout" work from the brief
   before anything downstream exists. */
const STAGE_PLAN = {
  brief:    { label: "Generate my design →", action: () => generateLayout() },
  identify: { label: "See the design →",     next: "design" },
  design:   { label: "Review cart →",        next: "checkout" },
  checkout: { label: "Checkout",             action: () => startPayment() },
};

/* A one-line reminder in the rail header that the assistant knows where you
   are. The chat is the same conversation throughout; this just tells the user
   what asking will affect right now. */
const STAGE_CONTEXT = {
  brief:    "Describe your room and I'll design it",
  identify: "Ask me about anything I spotted in your photo",
  design:   "Ask me to correct a reading, or trade a piece for another",
  checkout: "Ask about pricing or delivery",
};

function stageEl(name) { return $("view-" + name); }
function stepEl(name)  { return $("step-" + name); }

function setStage(name) {
  if (!STAGES.includes(name)) return;
  if (stepEl(name).disabled) return;   // locked stages are not navigable
  state.stage = name;

  STAGES.forEach(s => { stageEl(s).hidden = s !== name; });
  paintStepper();
  paintStageFoot();
  $("rail-context").textContent = STAGE_CONTEXT[name];
  // A stage change is a context change, so start it at the top rather than
  // inheriting the previous stage's scroll position.
  $("stage-scroll").scrollTop = 0;
}

/* Dots, labels, and the progress fill. The fill spans from the first bubble to
   the current one: with five bubbles across an 80% rail, each step is a fifth
   of that, so stage i sits at i/4 of the way along. */
function paintStepper() {
  const current = STAGES.indexOf(state.stage);
  STAGES.forEach((s, i) => {
    const el = stepEl(s);
    el.classList.toggle("current", i === current);
    // Three states, not two. "done" means BEHIND you - a stage you actually
    // passed through. A stage AHEAD can be unlocked (its content exists: one
    // generate fills the cart, which unlocks checkout) without having been
    // visited, and painting those as done showed a fully-completed progress
    // bar the moment a design landed. Those get "ahead": navigable, hover
    // affordance, but uncoloured.
    el.classList.toggle("done", i < current && !el.disabled);
    el.classList.toggle("ahead", i > current && !el.disabled);
    el.setAttribute("aria-current", i === current ? "step" : "false");
  });
  // Rail geometry follows the stage count rather than being hardcoded: with N
  // equal columns the rail is inset half a cell (50/N) at each end and spans
  // what is left. Published to CSS so the two cannot drift apart.
  const n = STAGES.length;
  const span = 100 - 100 / n;
  const pct = current <= 0 ? 0 : (current / (n - 1)) * span;
  const steps = $("steps");
  steps.style.setProperty("--inset", (50 / n) + "%");
  steps.style.setProperty("--span", span + "%");
  steps.style.setProperty("--progress", pct + "%");
}

/* The footer carries one forward action per stage, plus Back. Visualize is
   surfaced only on the layout stage, where the render actually appears. */
function paintStageFoot() {
  const name = state.stage;
  const i = STAGES.indexOf(name);
  const plan = STAGE_PLAN[name];

  $("step-of").textContent = `Step ${i + 1} of ${STAGES.length}`;

  // Back goes to the nearest unlocked stage behind this one.
  const prev = STAGES.slice(0, i).reverse().find(s => !stepEl(s).disabled);
  const back = $("stage-back");
  back.hidden = !prev;
  back.onclick = prev ? () => setStage(prev) : null;

  // Visualize belongs to the design stage and needs a photo plus a design.
  const render = $("do-render");
  render.hidden = name !== "design";

  const next = $("stage-next");
  next.textContent = plan.label;
  // Advance when the next stage is reachable; otherwise fall back to the
  // stage's own action, which is usually what produces that stage.
  const canAdvance = plan.next && !stepEl(plan.next).disabled;
  next.onclick = canAdvance ? () => setStage(plan.next) : plan.action || null;
  next.disabled = state.busy || (!canAdvance && !plan.action);
  // On a navigation-only stage with nothing ahead yet, the button would be a
  // dead control - say why instead of leaving it inert and unexplained.
  next.title = next.disabled && !state.busy ? "Generate a layout first" : "";
  updateActionAvailability();
}

/* Stages that carry the user to them when they first open. Checkout is
   absent on purpose - see unlockStage. */
const AUTO_ADVANCE = new Set(["identify", "design"]);

/** Unlock a stage once its panel has content. */
function unlockStage(name) {
  const step = stepEl(name);
  if (!step || !step.disabled) return;
  step.disabled = false;
  const empty = $("empty-" + name);   // the brief has none: it is never empty
  if (empty) empty.hidden = true;
  paintStepper();
  paintStageFoot();
  // Identify and Design each pull the user forward the first time they open.
  //
  // WHY BOTH, AND IN THAT ORDER. Identify opens first - the room reading and
  // its catalog matches land while the solver is still working - so the user
  // reads what we found in their photo during the wait instead of watching a
  // spinner. Design opens when the layout lands, and carrying them there is
  // the point of pressing Generate: the result should arrive, not sit behind
  // a bubble they have to notice.
  //
  // WHY ONLY ONCE EACH. A stage re-renders on every swap and every follow-up
  // turn. Advancing on each of those would drag the user off whatever they
  // were reading, so the move is spent the first time and never repeats.
  //
  // CHECKOUT IS DELIBERATELY EXCLUDED. It unlocks during the same generate
  // (the cart fills with the design), but nobody asked to pay - landing a
  // user on a payment screen they did not navigate to is the one advance
  // that would be presumptuous rather than helpful.
  if (AUTO_ADVANCE.has(name) && !state.autoAdvanced[name]) {
    state.autoAdvanced[name] = true;
    setStage(name);
  }
}

/* Panels reveal their own stage as they gain content. Called from the render
   functions rather than wrapped around them, so there is one obvious place
   where a stage becomes reachable. */
function syncStages() {
  // Identify opens on the room reading, which lands well before the layout -
  // and its own content (what we found, what we sell like it) is useful with
  // no design at all.
  const identifyReady = ["panel-room", "panel-shop"].some(id => !$(id).hidden);
  // Design opens on anything the solver produced.
  const designReady = ["panel-plan", "panel-render", "panel-alts"]
    .some(id => !$(id).hidden);
  if (identifyReady) unlockStage("identify");
  if (designReady) unlockStage("design");
  if (!$("panel-cart").hidden) unlockStage("checkout");
  // An empty state only makes sense while its stage has nothing in it.
  if (identifyReady) $("empty-identify").hidden = true;
  if (designReady) $("empty-design").hidden = true;
}

/* Kicks off the design from the brief - the forward action of stage 1. */
function generateLayout() {
  const style = $("chat-style").value;
  const budget = $("chat-budget").value.trim();
  const text = `Design my room in a ${style} style${budget ? ` with a budget of $${budget}` : ""}.`;
  addBubble("You", esc(text), "user");
  sendTurn(text, { includePhoto: !state.sessionId });
}

/* --- Brief ---------------------------------------------------------------- */

function briefSummary() {
  const style = $("chat-style").value;
  const budget = $("chat-budget").value.trim();
  const w = $("room-width").value.trim(), d = $("room-depth").value.trim();
  const bits = [style];
  if (budget) bits.push(currencySymbol() + Number(budget).toLocaleString());
  if (w && d) bits.push(`${w}×${d}cm`);
  if (state.roomPhoto) bits.push("photo added");
  const text = bits.join(" · ");
  $("brief-summary").textContent = text;
  // Surfaced under the Brief step, so the settings stay readable from every
  // later stage without navigating back to check them.
  const label = document.querySelector("#step-brief .label");
  if (label) label.title = text;
}

["chat-style", "chat-budget", "room-width", "room-depth"].forEach(id =>
  $(id).addEventListener("input", briefSummary));

/* --- Wiring --------------------------------------------------------------- */

$("chat-send").onclick = () => {
  const input = $("chat-message");
  const text = input.value.trim();
  if (!text) return;
  addBubble("You", esc(text), "user");
  input.value = "";
  input.style.height = "auto";
  // The photo rides along on the first turn that has one.
  sendTurn(text, { includePhoto: !state.sessionId });
};

$("chat-message").addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("chat-send").click(); }
});
// Grow the composer with the message, to a ceiling - a one-line box for a
// paragraph of room description is the wrong shape.
$("chat-message").addEventListener("input", e => {
  e.target.style.height = "auto";
  e.target.style.height = Math.min(e.target.scrollHeight, 140) + "px";
});

/* Suggestion chips: a blank composer is the hardest part of a chat UI, so the
   empty transcript offers openers that are real, sendable turns. */
document.querySelectorAll("[data-suggest]").forEach(b => b.onclick = () => {
  $("chat-message").value = b.dataset.suggest;
  $("chat-send").click();
});

$("room-photo-input").addEventListener("change", e => {
  const file = e.target.files?.[0];
  if (!file) return;
  state.roomPhoto = file;
  $("photo-preview").src = URL.createObjectURL(file);
  $("photo-preview").hidden = false;
  $("photo-prompt").hidden = true;
  $("upload-name").textContent = file.name;
  // A new photo starts a new design rather than refining the previous room.
  // The old cart belongs to that previous session, so it is cleared too -
  // leaving it on screen would offer a Visualize button for a design the new
  // session does not have.
  state.sessionId = null;
  state.cart = null;
  state.bundles = [];
  $("panel-bundles").hidden = true;
  // Matches belong to the previous photo, so they go with it.
  state.shopResults = [];
  // A new photo is a new run, so the sequence gets to carry the user forward
  // again. Without this the second design would land silently behind a bubble
  // the first run had already spent its advance on.
  state.autoAdvanced = {};
  briefSummary();
  updateActionAvailability();
  // Identify deliberately does NOT run here. Picking a file is not a request
  // to analyse it: the user has not said what they want yet, and matching
  // costs a vision call plus one embedding per object found. It runs when the
  // design does - off `room_analysis`, which arrives with the detections
  // already made - so a photo the user re-picks or abandons costs nothing.
});

/* --- Accuracy wiring ------------------------------------------------------ */

$("confirm-accept").onclick   = () => submitDimensions({ measured: false });
$("confirm-measured").onclick = () => submitDimensions({ measured: true });

$("add-door").onclick   = () => addOpening("door");
$("add-window").onclick = () => addOpening("window");

// Openings are rebuilt on every change, so both edits and removals delegate.
$("openings-list").addEventListener("input", e => {
  const row = e.target.closest(".opening");
  const field = e.target.dataset.f;
  if (!row || !field) return;
  const o = state.openings[Number(row.dataset.i)];
  if (!o) return;
  o[field] = field === "wall" ? e.target.value : Number(e.target.value) || 0;
  // A door's swing tracks its width unless the room says otherwise.
  if (field === "width_cm" && o.kind === "door") o.swing_cm = o.width_cm;
});

$("openings-list").addEventListener("click", e => {
  const drop = e.target.closest("[data-drop]");
  if (!drop) return;
  state.openings.splice(Number(drop.dataset.drop), 1);
  renderOpenings();
});

$("apply-openings").onclick = () => {
  if (!state.openings.length) return;
  const doors = state.openings.filter(o => o.kind === "door").length;
  const windows = state.openings.length - doors;
  const parts = [];
  if (doors) parts.push(`${doors} door${doors > 1 ? "s" : ""}`);
  if (windows) parts.push(`${windows} window${windows > 1 ? "s" : ""}`);
  const text = `I've marked ${parts.join(" and ")}. Re-solve the layout around them.`;
  addBubble("You", esc(text), "user");
  sendTurn(text, { openings: true });
};

// Alternatives are rendered dynamically, so the click is delegated.
$("alts").addEventListener("click", e => {
  const button = e.target.closest(".alt");
  if (button) doSwap(button.dataset.role, button.dataset.item);
});

STAGES.forEach(s => stepEl(s).onclick = () => setStage(s));

// Visualize renders into the layout stage, where the image appears.
$("do-render").onclick = () => { setStage("design"); doRender(); };
// Re-render from the Swap stage. Stays on this stage rather than bouncing to
// Layout: the user is mid-comparison, and moving them would lose their place.
$("rerender-swap").onclick = () => doRender({ stay: true });
$("pay-close").onclick = cancelPayment;

// Esc closes the payment sheet, but only before anything is charged - a
// receipt should not vanish on a stray keypress.
document.addEventListener("keydown", e => {
  if (e.key === "Escape" && !$("pay-overlay").hidden && !pay.busy) cancelPayment();
});

checkHealth();
briefSummary();
renderOpenings();
setStage("brief");
