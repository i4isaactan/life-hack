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

const money = cents => "$" + (cents / 100).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
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
  const gen = $("generate-layout");
  if (gen) gen.disabled = on;
  $("chat-send").textContent = on ? "…" : "Send";
  document.body.classList.toggle("is-busy", on);
}

/* --- The one call that drives everything ---------------------------------- */
async function sendTurn(message, { includePhoto = false } = {}) {
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

  const width = $("room-width").value.trim();
  const depth = $("room-depth").value.trim();
  if (width && depth) {
    form.append("room_width_cm", width);
    form.append("room_depth_cm", depth);
  }

  const bubble = startAssistantBubble();
  const status = setStatus("Thinking…");

  try {
    const done = await streamSSE("/api/chat", form, {
      text_delta: d => bubble.append(d.text || ""),
      room_analysis: d => { state.room = d.room; renderRoomFacts(d.room); },
      layout_update: d => { state.layout = d.layout; renderFloorPlan(d.layout); },
      // `alternatives` arrives before `cart_update`, and the swap filter needs
      // the cart to know which roles were actually placed, so re-render the
      // options once the cart lands.
      cart_update: d => { state.cart = d.cart; renderCart(d); renderAlternatives(state.options); },
      alternatives: d => { state.options = d.options || []; renderAlternatives(state.options); },
      bundles: d => renderBundles(d.bundles),
      clarification_needed: d => renderClarifications(d),
      intent: () => {},
      error: d => addBubble("Error", `<span class="err">${esc(d.message)}</span>`),
    });
    if (done?.session_id) state.sessionId = done.session_id;
    if (bubble.empty) bubble.remove();
  } catch (err) {
    bubble.remove();
    addBubble("Error", `<span class="err">${esc(err.message)}</span>`);
  } finally {
    status.clear();
    setBusy(false);
    updateActionAvailability();
  }
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
  $("panel-room").hidden = false;
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

/* --- Floor plan ----------------------------------------------------------- */
/* Placements are top-left origin in centimetres; the SVG viewBox is the room,
   so no manual scaling is needed. z=0 (rugs) must paint underneath. */
function renderFloorPlan(layout) {
  if (!layout) return;
  const W = layout.room_width_cm, D = layout.room_depth_cm;
  const placements = [...(layout.placements || [])].sort((a, b) => (a.z ?? 1) - (b.z ?? 1));

  const rects = placements.map(p => {
    const provisional = p.confidence && p.confidence !== "high";
    const label = `${p.name} — ${Math.round(p.w_cm)}×${Math.round(p.d_cm)}cm${provisional ? ` (±${Math.round(p.tolerance_cm)}cm, ${p.confidence} confidence)` : ""}${p.rationale ? `\n${p.rationale}` : ""}`;
    // A rug is metres wide and a lamp is a 38cm dot, so a label sized for one
    // is unreadable on the other. Scale it to the piece and pin it to the top
    // of the rect, which keeps a large rug's name clear of whatever sits on it.
    const size = Math.max(9, Math.min(17, Math.min(p.w_cm / 5.5, p.d_cm / 2.2)));
    const fits = p.w_cm >= 55 && p.d_cm >= 30;
    const cx = p.x_cm + p.w_cm / 2;
    return `<g>
      <rect x="${p.x_cm}" y="${p.y_cm}" width="${p.w_cm}" height="${p.d_cm}" rx="4"
            fill="${esc(p.swatch || "#8B7355")}" fill-opacity="${p.z === 0 ? 0.5 : 0.92}"
            stroke="#26333f" stroke-width="1.5"
            stroke-dasharray="${provisional ? "7 5" : "0"}"><title>${esc(label)}</title></rect>
      ${fits
        ? `<text x="${cx}" y="${p.y_cm + size * 1.35}" font-size="${size}" text-anchor="middle"
                 fill="#26333f" style="pointer-events:none;font-weight:700">${esc(p.name)}</text>
           <text x="${cx}" y="${p.y_cm + size * 2.5}" font-size="${size * 0.8}" text-anchor="middle"
                 fill="#26333f" fill-opacity=".7" style="pointer-events:none">${Math.round(p.w_cm)}×${Math.round(p.d_cm)}cm</text>`
        // Too small to write inside, so the label goes below the footprint,
        // clamped to the room so a piece in a corner is not cut off by the
        // viewBox, and anchored to whichever side keeps it in frame.
        : (() => {
            const anchor = cx < W * 0.15 ? "start" : cx > W * 0.85 ? "end" : "middle";
            const lx = Math.max(2, Math.min(W - 2, cx));
            return `<text x="${lx}" y="${p.y_cm + p.d_cm + 15}" font-size="13" text-anchor="${anchor}"
                          fill="#26333f" style="pointer-events:none;font-weight:700">${esc(p.name)}</text>`;
          })()}
    </g>`;
  }).join("");

  $("floorplan").innerHTML =
    `<svg viewBox="-10 -10 ${W + 20} ${D + 20}" role="img" aria-label="Floor plan">
       <rect x="0" y="0" width="${W}" height="${D}" fill="#fdfcfa" stroke="#26333f" stroke-width="3"/>
       ${rects}
     </svg>
     <p class="muted note-line">${placements.length} piece${placements.length === 1 ? "" : "s"} placed in ${Math.round(W)}×${Math.round(D)}cm. Dashed outlines are provisional positions.</p>`;

  const skipped = layout.skipped || [];
  const withheld = layout.withheld || [];
  $("plan-notes").innerHTML = [
    ...skipped.map(s => `<div class="skip">Skipped ${esc(s.name)}: ${esc(s.reason)}</div>`),
    ...withheld.map(w => `<div class="skip">Withheld ${esc(w.name)}: ${esc(w.reason)}</div>`),
  ].join("");
  $("panel-plan").hidden = false;
  syncViews();
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
  syncViews();
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
  syncViews();
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
      swap_started: d => addBubble("Swap", `${esc(d.from.name)} → <strong>${esc(d.to.name)}</strong>`),
      layout_update: d => { state.layout = d.layout; renderFloorPlan(d.layout); },
      cart_update: d => { state.cart = d.cart; renderCart(d); renderAlternatives(state.options); },
      alternatives: d => { state.options = d.options || []; renderAlternatives(state.options); },
      render_failed: d => addBubble("Render", `<span class="err">${esc(d.reason)}</span>`),
      error: d => addBubble("Error", `<span class="err">${esc(d.message)}</span>`),
    });
  } catch (err) {
    addBubble("Error", `<span class="err">${esc(err.message)}</span>`);
  } finally {
    status.clear();
    setBusy(false);
  }
}

/* --- Render --------------------------------------------------------------- */
/* Two shapes come back depending on the backend: `room_render` (one composed
   image of the whole room) or a stream of per-item `render_update` frames. */
async function doRender() {
  if (state.busy || !state.sessionId) return;
  setBusy(true);
  const status = setStatus("Rendering…");
  $("renders").innerHTML = "";
  $("panel-render").hidden = false;

  try {
    await streamSSE("/api/render", { session_id: state.sessionId, item_ids: [], per_item: false }, {
      render_started: d => {
        status.clear();
        setStatus(`Rendering ${d.total} ${d.total === 1 ? "image" : "images"} (${d.method})…`);
      },
      room_render: d => {
        $("renders").innerHTML = `
          <img class="render-img" src="${esc(assetUrl(d.image_url))}" alt="Your room, visualized">
          <p class="muted note-line">${esc(d.disclaimer || "")}</p>
          ${d.omitted?.length ? `<div class="skip">Not shown: ${esc(d.omitted.join(", "))}</div>` : ""}
          ${d.replaced?.length ? `<p class="muted note-line">Removed from the photo: ${esc(d.replaced.join(", "))}</p>` : ""}`;
      },
      render_update: d => {
        const item = d.render || d;
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
  }
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
};
function payOpen() { $("pay-overlay").hidden = false; }
function payClose() {
  $("pay-overlay").hidden = true;
  pay.intent = null; pay.challenge = null; pay.idem = null;
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
    pay.intent = await payFetch("/api/payment/intent", {
      item_ids: lines.map(l => l.item_id),
      session_id: state.sessionId,
      payment_method_ids: pay.assignments,
    });
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
  $("pay-body").innerHTML = `
    ${payStep("review")}
    <div class="banner">${esc(it.disclaimer)}</div>
    ${charges}
    ${risk}
    <div class="row"><span class="muted">Subtotal</span><span>${money(it.subtotal_cents)}</span></div>
    <div class="row"><span class="muted">Shipping</span><span>${it.shipping_cents ? money(it.shipping_cents) : "Free"}</span></div>
    <div class="row"><span class="muted">Tax</span><span>${money(it.tax_cents)}</span></div>
    <div class="row total"><strong>Total</strong><strong>${money(it.total_cents)}</strong></div>
    ${it.over_budget ? `<div class="skip">This is over the ${money(it.budget_cents)} budget you set.</div>` : ""}`;

  $("pay-foot").innerHTML = `
    <button class="primary" id="pay-next">${it.requires_step_up ? "Verify identity to continue" : "Continue to authorize"}</button>
    <div class="pay-secure">🔒 You are still on Room Hack — no redirect. Cancel any time before you authorize.</div>`;

  // Changing a card re-prices server-side: a different card can change which
  // limits apply, so the risk assessment must be recomputed, not patched here.
  $("pay-body").querySelectorAll(".pay-card-select").forEach(sel => {
    sel.onchange = async () => {
      pay.assignments[sel.dataset.merchant] = sel.value;
      await repriceIntent();
    };
  });
  $("pay-next").onclick = () => (it.requires_step_up ? beginVerify() : renderAuthorize());
}

async function repriceIntent() {
  try {
    pay.intent = await payFetch("/api/payment/intent", {
      item_ids: (state.cart?.lines || []).map(l => l.item_id),
      session_id: state.sessionId,
      payment_method_ids: pay.assignments,
    });
    pay.challenge = null;
    pay.idem = `idem_${Date.now()}_${Math.random().toString(16).slice(2, 10)}`;
    renderPreview();
  } catch (err) {
    $("pay-body").innerHTML = `<div class="skip">${esc(err.message)}</div>`;
  }
}

/* Step 2: step-up identity verification, in the shape of a 3-D Secure prompt.
   Only reached when the server's risk assessment asked for it. */
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
    ${it.requires_step_up ? `<div class="risk ok"><span class="risk-ico">✓</span><span>Identity verified.</span></div>` : ""}
    ${it.charges.map(c => {
      const m = pay.methods.find(x => x.id === c.payment_method_id);
      return `<div class="row"><span>${esc(c.merchant)} <span class="card-pill">${esc((m?.network || "visa").toUpperCase())} ···· ${esc(m?.last4 || "")}</span></span><strong>${money(c.total_cents)}</strong></div>`;
    }).join("")}
    <div class="row total"><strong>You are authorizing</strong><strong>${money(it.total_cents)}</strong></div>
    ${error ? `<div class="skip">${esc(error)}</div>` : ""}
    <p class="muted" style="margin-top:12px">Room Hack's agent assembled this order. Holding the button below is your instruction to charge it — the agent cannot do this for you.</p>`;
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
    });
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
          <span class="card-pill">${esc((m?.network || "visa").toUpperCase())} ···· ${esc(m?.last4 || "")}</span>
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
  $("do-checkout").disabled = !hasDesign || state.busy;
}


/* --- Workspace shell ------------------------------------------------------
   The old planner stacked every panel down one page, so a fresh session showed
   four empty cards and a finished design needed scrolling to read. The canvas
   is tabbed instead: one view at a time, and a tab only becomes reachable when
   there is something behind it. */

const VIEWS = ["plan", "render", "cart", "alts"];

function setView(name) {
  if (!VIEWS.includes(name)) return;
  state.view = name;
  VIEWS.forEach(v => {
    $("view-" + v).hidden = v !== name;
    $("tab-" + v).classList.toggle("active", v === name);
    $("tab-" + v).setAttribute("aria-selected", String(v === name));
  });
}

/** Enable a tab once its panel has content, and reveal the canvas. */
function unlockView(name) {
  const tab = $("tab-" + name);
  if (!tab || !tab.disabled) return;
  tab.disabled = false;
  $("canvas-empty").hidden = true;
  $("canvas-tabs").hidden = false;
  // First unlocked view wins focus, so a design lands on the floor plan
  // rather than leaving the user on an empty tab.
  if (!state.view) setView(name);
}

/* Panels reveal their own tab as they gain content. Called from the render
   functions rather than wrapped around them, so there is one obvious place
   where a panel becomes reachable. */
function syncViews() {
  if (!$("panel-plan").hidden) unlockView("plan");
  if (!$("panel-cart").hidden) unlockView("cart");
  if (!$("panel-alts").hidden) unlockView("alts");
  if (!$("panel-render").hidden) unlockView("render");
}

/* --- Brief ---------------------------------------------------------------- */

function briefSummary() {
  const style = $("chat-style").value;
  const budget = $("chat-budget").value.trim();
  const w = $("room-width").value.trim(), d = $("room-depth").value.trim();
  const bits = [style];
  if (budget) bits.push("$" + Number(budget).toLocaleString());
  if (w && d) bits.push(`${w}×${d}cm`);
  if (state.roomPhoto) bits.push("photo added");
  $("brief-summary").textContent = bits.join(" · ");
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

$("generate-layout").onclick = () => {
  const style = $("chat-style").value;
  const budget = $("chat-budget").value.trim();
  const text = `Design my room in a ${style} style${budget ? ` with a budget of $${budget}` : ""}.`;
  addBubble("You", esc(text), "user");
  sendTurn(text, { includePhoto: !state.sessionId });
};

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
  briefSummary();
  updateActionAvailability();
});

// Alternatives are rendered dynamically, so the click is delegated.
$("alts").addEventListener("click", e => {
  const button = e.target.closest(".alt");
  if (button) doSwap(button.dataset.role, button.dataset.item);
});

VIEWS.forEach(v => $("tab-" + v).onclick = () => setView(v));

$("do-render").onclick = () => { setView("render"); unlockView("render"); doRender(); };
$("do-checkout").onclick = startPayment;
$("pay-close").onclick = cancelPayment;

// Esc closes the payment sheet, but only before anything is charged - a
// receipt should not vanish on a stray keypress.
document.addEventListener("keydown", e => {
  if (e.key === "Escape" && !$("pay-overlay").hidden && !pay.busy) cancelPayment();
});

checkHealth();
updateActionAvailability();
briefSummary();
