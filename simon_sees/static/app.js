// Simon Sees phone client: camera frames up, Simon's voice down. The camera IS the interface: the picture
// fills the screen and a thin HUD floats on it; each player gets a ring drawn from their SAM box, coloured by
// the verdict (see docs/design/DIRECTION.md).
const $ = (id) => document.getElementById(id);
const FPS = 3;          // frames per second sent to the referee
const SIDE = 640;       // long side of each frame, px (SAM finds people 3-5 m away; each crop goes to d1)
const QUALITY = 0.72;   // JPEG quality

const state = { lineup: [], names: {}, skip: new Set(), lining: false, ws: null, stream: null, facing: "environment", sending: false, info: {}, playing: false,
  view: null, board: [], hist: {}, gameId: null, t0: null, tEnd: null, alivePrev: null, mood: "info", s: null, lineupView: null };
const params = new URLSearchParams(location.search);
const CAPTIONS = params.has("captions");  // ?captions=1: subtitles of what Simon says + the referee's decisions in Ref's notes
const MUTE = params.has("mute");          // ?mute=1: no phone speech (the demo recorder adds Simon's voice afterwards)
window.__says = [];                        // every spoken line with its time, for tests and demo audio
const pad2 = (n) => String(Math.max(0, n | 0)).padStart(2, "0");

// ---------------------------------------------------------------- websocket
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  state.ws = ws;
  ws.onopen = () => { state.statusMsg = null; guide(); };
  ws.onclose = () => { setStatus("Disconnected, reconnecting…"); setTimeout(connect, 1500); };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "info") onInfo(msg);
    else if (msg.type === "state") onState(msg);
    else if (msg.type === "say") speak(msg);
    else if (msg.type === "lineup") onLineup(msg);
    else if (msg.type === "error") { setStatus(msg.message); state.lining = false; $("lineup").textContent = state.lineup.length ? "Line up again" : "Line up"; if (state.playing) showLobby(); updateStart(); }
  };
}
const send = (obj) => state.ws && state.ws.readyState === 1 && state.ws.send(JSON.stringify(obj));

function onInfo(info) {
  state.info = info;
  $("replay-banner").hidden = !info.replay;
  $("live-pill").hidden = !!info.replay;
  // in a replay the camera isn't used: the scene shows the recorded frames instead of offering a camera button
  $("cam-on").hidden = !!info.replay;
  $("cam-flip").hidden = !!info.replay;
  if (info.replay && !state.stream) {
    $("cam-hint").textContent = "";
    if (!state.view) setView({ url: "/api/view/l0.jpg" });
  }
  const m = info.models || {};
  const stack = [m.segmenter, m.embedder, m.decider].filter(Boolean).map(short).join(" → ");
  // the machinery goes in Ref's notes; the visible status line only says what to do next
  state.stackLine = info.loading ? "loading SAM → DINO → referee…" : info.model_error ? `model error: ${info.model_error}` : `${stack || info.model} ready`;
  renderNotes();
  state.statusMsg = null;
  guide();
  if (info.loading) setTimeout(() => send({ type: "info" }), 2000);
  updateStart();
}
// Ref's notes: model names, where they run, latencies. Closed by default; one sheet in the lobby, one in play.
function renderNotes() {
  const i = state.info || {};
  const rows = [["Vision", state.stackLine || "…"]];
  if (i.backend === "modal") rows.push(["Runs on", "Modal GPU"]);
  rows.push(["Voice", i.tts === "browser" ? "phone" : i.tts || "…"]);
  if (i.replay) rows.push(["Mode", "replay demo: recorded frames, not a live camera"]);
  if (state.lineupMs != null) rows.push(["Line-up", `${state.lineup.length} found in ${(state.lineupMs / 1000).toFixed(1)} s`]);
  $("lobby-notes").innerHTML = rows.map(([k, v]) => `<p><b>${esc(k)}</b> <span>${esc(v)}</span></p>`).join("");
}
function toggleNotes(btn, panel) {
  const open = btn.getAttribute("aria-expanded") !== "true";
  btn.setAttribute("aria-expanded", String(open));
  panel.hidden = !open;
  panel.closest(".screen").classList.toggle("notes-open", open);
}

// ---------------------------------------------------------------- the scene: camera or judged frame, framed on the players
// The picture is scaled so everyone playing fits the middle band of the screen (between the HUD's top and
// bottom). A landscape replay frame on a portrait phone can't fill the screen without cutting players off, so it
// is shown sharp in the band and continues as a blurred copy behind (the backdrop), feathered at the seams.
function setView(v) {
  if (!v || !v.url) return;
  const still = $("still");
  const changed = !state.view || state.view.url !== v.url;
  state.view = { ...(state.view || {}), ...v };
  if (state.stream) { layoutScene(); return; }  // live: the camera is the picture, boxes map onto it
  if (changed) {
    const img = new Image();
    img.onload = () => {
      still.src = img.src; still.hidden = false; $("backdrop").src = img.src; $("backdrop").classList.add("on");
      state.view.w = state.view.w || img.naturalWidth; state.view.h = state.view.h || img.naturalHeight;
      state.media = { w: img.naturalWidth, h: img.naturalHeight };
      $("framer").classList.remove("empty");
      layoutScene();
    };
    img.src = v.url;
  } else layoutScene();
}
function mediaSize() {
  if (state.stream) { const v = $("preview"); return v.videoWidth ? { w: v.videoWidth, h: v.videoHeight } : null; }
  return state.media || null;
}
function focusBox(ms) {  // the players' boxes, in media pixels (union + margin); null = no players known
  const vw = (state.view && state.view.w) || ms.w, vh = (state.view && state.view.h) || ms.h;
  const boxes = ringItems().filter((r) => r.box && r.kind !== "gone").map((r) => r.box);
  if (!boxes.length) return null;
  const k = ms.w / vw, kh = ms.h / vh;
  let x0 = Math.min(...boxes.map((b) => b[0])) * k, x1 = Math.max(...boxes.map((b) => b[2])) * k;
  let y0 = Math.min(...boxes.map((b) => b[1])) * kh, y1 = Math.max(...boxes.map((b) => b[3])) * kh;
  const mx = (x1 - x0) * 0.1 + 12, my = (y1 - y0) * 0.06 + 8;
  return [x0 - mx, y0 - my, x1 + mx, y1 + my];
}
function layoutScene() {
  const ms = mediaSize();
  const W = innerWidth, H = innerHeight;
  const fr = $("framer");
  if (!ms) { fr.style.cssText = ""; drawRings(); return; }
  const cover = Math.max(W / ms.w, H / ms.h);
  // the band where players should stand: under the numeral, above the call and the referee panel
  const scr = $("play").hidden ? $("lobby") : $("play");
  const hudTop = scr.querySelector(".hud-top").getBoundingClientRect().bottom || H * 0.2;
  const under = ($("play").hidden ? scr.querySelector(".hud-bottom") : $("stage")).getBoundingClientRect().top || H * 0.6;
  const goneRows = 26 * goneLayout(ringItems().filter((r) => r.kind === "gone" || !r.box)).rows;
  const top = Math.min(hudTop + 38 + goneRows, H * 0.4), bot = Math.max(under - 24, top + 120);
  // live camera: show the whole field of view, so players see all the room they have and their rings move
  // inside it (framing on the line-up spot read as "stand in this box"); replays frame the judged players
  const F = state.stream ? [0, 0, ms.w, ms.h] : (focusBox(ms) || [ms.w * 0.2, 0, ms.w * 0.8, ms.h]);
  const fw = F[2] - F[0], fh = F[3] - F[1];
  let s = Math.min((W - 24) / fw, (bot - top) / fh);
  // fill the width when the players still fit the band; a crowded HUD may shrink the picture a little (its sides
  // then fade into the blurred backdrop) rather than run the players under the call
  s = Math.max(s, (W / ms.w) * 0.8);
  if (s * ms.w < W && (W / ms.w) * fh <= (bot - top) * 1.04) s = W / ms.w;
  if (s > cover * 0.985) s = Math.max(s, cover);  // close enough to full-bleed: go full-bleed
  const dw = ms.w * s, dh = ms.h * s;
  let ox = W / 2 - ((F[0] + F[2]) / 2) * s;
  ox = dw >= W ? Math.min(0, Math.max(W - dw, ox)) : (W - dw) / 2;
  let oy = (top + bot) / 2 - ((F[1] + F[3]) / 2) * s;
  if (dh >= H) oy = Math.min(0, Math.max(H - dh, oy));
  else oy = Math.min(Math.max(oy, Math.min(0, H - dh)), Math.max(0, H - dh));
  state.xf = { s, ox, oy, mw: ms.w, mh: ms.h };
  fr.style.width = `${dw}px`; fr.style.height = `${dh}px`;
  fr.style.transform = `translate(${ox}px, ${oy}px)`;
  fr.classList.toggle("feather", dh < H - 1);
  fr.classList.toggle("sides", dw < W - 1);
  $("backdrop").classList.toggle("on", (dh < H - 1 || dw < W - 1) && !state.stream);
  drawRings();
}
addEventListener("resize", () => { layoutScene(); fitCall(); });
$("preview").addEventListener("loadedmetadata", layoutScene);

// the "already out" chips under the numeral wrap like words; estimate their rows to keep the players clear of them
function goneLayout(items) {
  let x = 24, rows = items.length ? 1 : 0;
  for (const it of items) {
    const w = 7.2 * (`${it.number} · ${it.name || it.display || ""} Out · round ${it.out_round}`.length) + 24;
    if (x > 24 && x + w > innerWidth - 24) { rows++; x = 24; }
    x += w + 6;
  }
  return { rows };
}
// which rings to draw: the line-up (lobby) or the board (play), each with a verdict kind
function ringItems() {
  if (!state.playing && !(state.s && state.s.phase === "over" && !$("play").hidden)) {
    return playing().map((p) => ({ id: p.id, number: p.number, name: (state.names[p.id] || "").trim(), box: p.box, kind: p.seen === false ? "lost" : "found" }));
  }
  const s = state.s || {};
  const r = state.lastRound && state.lastRound.round === s.round ? state.lastRound : null;  // verdicts for this round
  const verdictUp = r && ["judging", "rest", "final", "over"].includes(s.phase);
  const winners = new Set(s.winners || []);
  return (state.board || []).map((p) => {
    const d = r && (r.decisions || []).find((x) => x.pid === p.id);
    let kind = "idle";
    if (s.phase === "over") kind = winners.has(p.display_name) ? "win" : p.alive ? "idle" : "gone";
    else if (!p.alive) kind = p.out_round === s.round ? "out" : "gone";
    else if (p.seen === false) kind = "lost";
    else if (verdictUp && d) kind = r.simon === false ? "trick" : "safe";
    return { id: p.id, number: p.number, name: p.named === false || /^Number \d+$/.test(p.call_name || "") ? "" : p.display_name, display: p.display_name, box: p.box, kind, out_round: p.out_round, alive: p.alive };
  });
}
const SVGNS = "http://www.w3.org/2000/svg";
function drawRings() {
  const layer = $("ring-layer");
  const inPlay = !$("play").hidden;
  const list = inPlay ? $("board") : $("tags");
  (inPlay ? $("tags") : $("board")).innerHTML = "";
  layer.innerHTML = "";
  list.innerHTML = "";
  const xf = state.xf;
  const items = ringItems();
  const vw = (state.view && state.view.w) || (xf && xf.mw), vh = (state.view && state.view.h) || (xf && xf.mh);
  $("scene").classList.toggle("judging", inPlay && state.s && state.s.phase === "judging");
  const placed = [];
  const gone = [];
  for (const it of items) {
    const li = document.createElement("li");
    li.className = "tag";
    const label = it.name ? `<span class="n">${it.number}</span> · ${esc(it.name)}` : inPlay ? esc(it.display || `Player ${it.number}`) : `<span class="n">${it.number}</span> · Player ${it.number}`;
    let stateChip = "";
    if (it.kind === "out" || it.kind === "gone") stateChip = `<span class="chip state out">Out · round ${it.out_round}</span>`;
    else if (it.kind === "lost") stateChip = `<span class="chip state lost">Lost you</span>`;
    li.innerHTML = `<span class="chip${it.kind === "win" ? " win-name" : ""}">${label}</span>${it.kind === "win" ? `<span class="sr"> wins</span>` : ""}`;
    // the verdict stamp (OUT, LOST YOU) sits on the body, centred in the ring; it belongs to the player's card
    let stamp = null;
    if (stateChip) { stamp = document.createElement("span"); stamp.className = "stamp"; stamp.innerHTML = stateChip; li.append(stamp); }
    if (inPlay) {
      li.dataset.testid = "player-card";
      li.dataset.state = it.alive === false ? "out" : it.kind === "lost" ? "lost" : "alive";
    }
    if (it.kind === "gone") {  // out in an earlier round: no ring any more, a quiet line under the numeral
      li.classList.add("gone");
      li.innerHTML = `<span class="chip gone"><span class="n">${it.number}</span> · ${esc(it.name || it.display || "")} <span class="out-tag">Out · round ${it.out_round}</span></span>`;
      gone.push(li); list.append(li); continue;
    }
    if (!xf || !it.box || !vw) { li.classList.add("nobox"); gone.push(li); list.append(li); continue; }
    const k = xf.mw / vw, kh = xf.mh / vh;
    const x0 = xf.ox + it.box[0] * k * xf.s, x1 = xf.ox + it.box[2] * k * xf.s;
    const y0 = xf.oy + it.box[1] * kh * xf.s, y1 = xf.oy + it.box[3] * kh * xf.s;
    const w = x1 - x0, h = y1 - y0, padx = 6, pady = 6;
    const g = document.createElementNS(SVGNS, "g");
    g.setAttribute("class", `r-${it.kind}`);
    const rx = Math.min(w / 2 + padx, 30);
    const rect = (cls) => { const e = document.createElementNS(SVGNS, "rect"); e.setAttribute("class", cls);
      e.setAttribute("x", x0 - padx); e.setAttribute("y", y0 - pady); e.setAttribute("width", w + 2 * padx); e.setAttribute("height", h + 2 * pady);
      e.setAttribute("rx", rx); e.setAttribute("ry", rx); e.setAttribute("pathLength", 400); return e; };
    const foot = document.createElementNS(SVGNS, "ellipse");
    foot.setAttribute("class", "ring-foot"); foot.setAttribute("cx", (x0 + x1) / 2); foot.setAttribute("cy", y1 + 2);
    foot.setAttribute("rx", w * 0.62 + 6); foot.setAttribute("ry", 5);
    g.append(rect("ring-glow"), rect("ring"), foot);
    layer.append(g);
    // the chip sits above the head, centred; the stamp on the chest
    list.append(li);
    const ty = y0 - pady - 8;
    placed.push({ li, cx: (x0 + x1) / 2, w: li.offsetWidth || 80, y: ty, stamp, sy: y0 + h * 0.4, bx: (x0 + x1) / 2 });
  }
  // neighbours' chips that would overlap are spread apart sideways (a few px each), then stepped up if still touching
  placed.sort((a, b) => a.cx - b.cx);
  for (let pass = 0; pass < 6; pass++) {
    for (let i = 1; i < placed.length; i++) {
      const a = placed[i - 1], b = placed[i];
      if (Math.abs(a.y - b.y) > 24) continue;
      const gap = (b.cx - b.w / 2) - (a.cx + a.w / 2) - 6;
      if (gap < 0) { a.cx += gap / 2; b.cx -= gap / 2; }
    }
  }
  const W = innerWidth;
  for (let i = 0; i < placed.length; i++) {
    const p = placed[i];
    p.cx = Math.min(W - 10 - p.w / 2, Math.max(10 + p.w / 2, p.cx));
    const prev = placed[i - 1];
    if (prev && p.cx - p.w / 2 < prev.cx + prev.w / 2 + 4 && Math.abs(prev.y - p.y) < 24) p.y = prev.y - 26;
    p.li.style.left = `${p.cx}px`; p.li.style.top = `${p.y}px`; p.li.style.transform = "translate(-50%, -100%)";
    if (p.stamp) { p.stamp.style.left = `calc(50% + ${p.bx - p.cx}px)`; p.stamp.style.top = `calc(100% + ${p.sy - p.y}px)`; }
  }
  // under the numeral: who is already out, then anyone without a box yet, so every player is always on screen
  const sub = (inPlay ? $("play") : $("lobby")).querySelector(".hud-top").getBoundingClientRect();
  let gx = 24, gy = sub.bottom + 8;
  for (const li of gone) {
    li.style.transform = "none"; li.style.top = `${gy}px`;
    if (gx > 24 && gx + li.offsetWidth > innerWidth - 24) { gx = 24; gy += 26; li.style.top = `${gy}px`; }
    li.style.left = `${gx}px`; gx += li.offsetWidth + 6;
  }

}

// ---------------------------------------------------------------- game state
function onState(s) {
  if (s.phase === "stopped" || s.phase === "lobby") { showLobby(); return; }
  if (!state.playing && s.phase !== "over") showPlay();
  if (s.phase === "over" && $("play").hidden) showPlay();
  if (s.game_id && s.game_id !== state.gameId) { state.gameId = s.game_id; state.hist = {}; state.t0 = Date.now(); state.tEnd = null; state.alivePrev = null; }
  state.s = s;
  state.simon = s.simon;
  state.board = s.players || state.board || [];
  if (s.view && s.view.url) setView(s.view);
  if (s.last_round && s.last_round.round_id) {
    state.lastRound = s.last_round;
    for (const d of s.last_round.decisions || []) {  // the referee's confidence per player, round by round
      const h = (state.hist[d.pid] = state.hist[d.pid] || []);
      if (!h.length || h[h.length - 1].round !== s.last_round.round) h.push({ round: s.last_round.round, p: d.p, simon: s.last_round.simon, out: d.out, kind: s.last_round.kind });
    }
  }
  if (s.phase === "over") {
    const won = s.winners && s.winners.length;
    setMood("win", won ? (s.winners.length > 1 ? "Winners" : "Winner") : "Full time");
    setCall(`<span class="big">${esc(won ? s.winners.join(", ") : "Simon wins")}</span>`, "call win");
    const r = $("result");
    r.textContent = won ? `Winner: ${s.winners.join(", ")}` : "Winner: Simon";
    r.hidden = false;
    $("wrong").hidden = true;
    if (!state.noteShownFor || state.noteShownFor !== s.game_id) { state.noteShownFor = s.game_id; showEndNote(); }
    if (!state.tEnd) state.tEnd = Date.now();
  }
  renderCalls();
  if (CAPTIONS) renderCaptions(s);
  // the numeral: players still in, with a red "−1" when someone goes out
  const alive = (s.players || []).filter((p) => p.alive).length;
  $("alive-count").textContent = pad2(alive);
  if (state.alivePrev != null && alive < state.alivePrev) flash($("alive-flash"), `−${state.alivePrev - alive}`, true);
  state.alivePrev = alive;
  $("play").dataset.phase = s.phase;
  $("meta").innerHTML = `Simon Says <span class="sep">/</span> ${s.phase === "over" ? "Full time" : s.round ? `Round ${s.round}` : "Kick-off"}<span class="sr"> (round ${s.round}/${s.max_rounds})</span>`;
  $("stat-round").innerHTML = `${s.round || 0}<small>/ ${s.max_rounds}</small>`;
  const bd = (s.last_round && s.last_round.breakdown) || s.latency_breakdown || {};
  const callMs = bd.total ?? s.last_latency_ms;
  $("stat-call").innerHTML = callMs != null ? `${(callMs / 1000).toFixed(callMs < 10000 ? 2 : 1)}<small>s</small>` : "–";
  renderStatus();
  renderSpark();
  const model = short(s.model || "");
  $("play-stack").innerHTML = `<b>Referee</b> <span>${esc(model || "…")}${s.last_latency_ms != null ? ` · ${Math.round(s.last_latency_ms)} ms` : ""}</span>` +
    (state.stackLine ? `<br><b>Vision</b> ${esc(state.stackLine.replace(/ ready$/, ""))}` : "") +
    (state.info.replay ? `<br><b>Mode</b> replay demo: recorded frames, not a live camera` : "");
  layoutScene();
  fitCall();
  if (s.phase === "over") { state.playing = false; setTimeout(() => { $("stop").setAttribute("aria-label", "Back"); }, 500); }
}
// the status word over the referee panel: what the referee is doing, or the verdict
function renderStatus() {
  const s = state.s || {};
  const el = $("phase");
  let word = { intro: "Listen", locating: "Finding you", calling: "Listen", watching: "Watching", judging: "Judging…", rest: "Breather", final: "Full time", over: "Game over" }[s.phase] ?? "";
  let cls = "status-word";
  const r = state.lastRound && state.lastRound.round === s.round ? state.lastRound : null;
  if (r && ["judging", "rest"].includes(s.phase)) {
    const outs = (r.decisions || []).filter((d) => d.out).length;
    const lost = (r.decisions || []).some((d) => d.seen === false);
    if (outs) { word = outs > 1 ? `${outs} out ↓` : "Out ↓"; cls += " out"; }
    else if (lost && state.mood === "warn") { word = "Lost you"; cls += " quiet"; }
    else if (r.simon === false) { word = "Safe · no Simon"; cls += " trick"; }
    else word = "Safe ↑";
  } else if (["calling", "intro", "locating", "final", "over"].includes(s.phase)) cls += " quiet";
  el.className = cls;
  el.textContent = word;
}
// sparkline: the referee's P(did it) per round. The most at-risk player in bright lime, everyone else faint.
function renderSpark() {
  const svg = $("spark");
  const s = state.s || {};
  const W = 342, H = 56, n = Math.max(2, s.max_rounds || 8);
  const X = (round) => 4 + ((round - 1) / (n - 1)) * (W - 8);
  const Y = (p) => 6 + (1 - p) * (H - 14);
  let out = `<line class="base" x1="0" x2="${W}" y1="${Y(0.5)}" y2="${Y(0.5)}"/>`;
  for (let i = 1; i <= n; i++) out += `<circle class="tick${i <= (s.round || 0) ? " on" : ""}" cx="${X(i)}" cy="${H - 1}" r="1.2"/>`;
  // who to follow: the alive player the referee was least sure about last round (lowest margin from "safe")
  const r = state.lastRound;
  let focus = null;
  if (r && r.decisions) {
    const margin = (d) => (d.p == null ? 9 : r.simon === false ? 1 - d.p : d.p);
    const ds = [...r.decisions].filter((d) => d.p != null).sort((a, b) => (b.out - a.out) || margin(a) - margin(b));
    focus = ds[0] || null;
  }
  const path = (pts) => {
    if (pts.length === 1) return `M${pts[0][0] - 3},${pts[0][1]} L${pts[0][0] + 3},${pts[0][1]}`;
    let d = `M${pts[0][0]},${pts[0][1]}`;
    for (let i = 1; i < pts.length; i++) {
      const [x0, y0] = pts[i - 1], [x1, y1] = pts[i];
      const cx = (x0 + x1) / 2;
      d += ` C${cx},${y0} ${cx},${y1} ${x1},${y1}`;
    }
    return d;
  };
  const series = (pid) => (state.hist[pid] || []).filter((h) => h.p != null && h.kind !== "freeze").map((h) => [X(h.round), Y(h.p), h]);
  for (const pid of Object.keys(state.hist)) {
    if (focus && String(focus.pid) === pid) continue;
    const pts = series(pid);
    if (pts.length) out += `<path class="others" d="${path(pts)}"/>`;
  }
  if (focus) {
    const pts = series(focus.pid);
    if (pts.length) {
      out += `<path class="line" d="${path(pts)}"/>`;
      const [lx, ly, lh] = pts[pts.length - 1];
      out += `<line class="head" x1="${lx}" x2="${lx}" y1="0" y2="${H - 6}"/><circle class="dot${lh.out ? " out" : ""}" cx="${lx}" cy="${ly}" r="3"/>`;
    }
    $("focus-name").textContent = s.phase === "over" ? "" : `${focus.player} · P(${r.kind === "freeze" ? "moved" : "did it"})`;
  } else {
    // nothing judged yet: a quiet flat trace, like a monitor waiting for a signal
    let d = "M0," + Y(0.5);
    for (let x = 6; x <= W; x += 6) d += ` L${x},${Y(0.5) + Math.sin(x / 9) * 2.2 * Math.sin(x / 61)}`;
    out += `<path class="waiting" d="${d}"/>`;
    $("focus-name").textContent = "P(did it) · waiting";
  }
  svg.innerHTML = out;
}
function tickClock() {
  if (state.t0 && !$("play").hidden) {
    const sec = Math.round(((state.tEnd || Date.now()) - state.t0) / 1000);
    $("stat-time").textContent = `${pad2(sec / 60)}:${pad2(sec % 60)}`;
  }
}
setInterval(tickClock, 1000);
function flash(el, text, minus) { el.textContent = text; el.classList.toggle("minus", !!minus); replay(el, "go"); }

function renderCaptions(s) {  // the referee's notebook: what d1 decided, how sure it was, how long it took
  const box = $("captions");
  const r = s.last_round || {};
  const model = short(s.model || "");
  box.hidden = false;
  if (!r.decisions) {
    box.innerHTML = `<div class="head"><b>Decisions</b><span class="dim">waiting for the first decision…</span></div>`;
    return;
  }
  const what = r.kind === "freeze" ? "P(moved)" : "P(did it)";
  const rows = r.decisions.map((d) => {
    const pct = d.p == null ? 0 : Math.round(d.p * 100);
    const verdict = d.out ? `<span class="out">OUT</span>` : d.seen === false ? `<span class="dim">not seen</span>` : d.uncertain ? `<span class="dim">unsure → safe</span>` : `<span class="ok">safe</span>`;
    return `<tr><td>${d.number != null ? `<b class="no">${d.number}</b> ` : ""}${esc(d.player)}</td><td class="bar"><span class="meter"><i style="width:${pct}%"></i></span></td>` +
      `<td class="p">${d.p == null ? "–" : d.p.toFixed(2)}</td><td class="v">${verdict}</td></tr>`;
  });
  const bd = r.breakdown || {};
  const ms = (k) => (bd[k] != null ? Math.round(bd[k]) : "–");
  box.innerHTML = `<div class="head"><b>Round ${r.round} · ${r.simon ? "Simon said" : "trick, no Simon"}</b><span>referee ${r.latency_ms != null ? Math.round(r.latency_ms) : "–"} ms</span></div>` +
    `<div class="head"><span>SAM ${ms("segment")} · DINO ${ms("embed")} · match ${ms("match")} ms${bd.round_trip != null ? ` · round trip ${Math.round(bd.round_trip)} ms` : ""}</span></div>` +
    `<table><thead><tr><th>${esc(model)}</th><th class="bar"></th><th class="p">${what}</th><th class="v"></th></tr></thead><tbody>${rows.join("")}</tbody></table>`;
}

// ---------------------------------------------------------------- playtest feedback ("Wrong call", end-of-game note)
// The players are the linesmen: a tap says which player the referee misjudged and what it should have been.
// Everything is appended to the session's feedback.jsonl, linked to the round by round_id.
state.flags = {};  // "round_id:pid" -> the verdict the players say was right
const flagKey = (rid, pid) => `${rid}:${pid}`;
async function postFeedback(body) {
  try {
    const r = await fetch("/api/feedback", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    return r.ok;
  } catch (_) { return false; }
}
async function flagCall(rid, d, where) {
  const k = flagKey(rid, d.pid);
  const retract = k in state.flags;
  const correct = d.out ? "safe" : "out";
  if (retract) delete state.flags[k]; else state.flags[k] = correct;
  renderCalls(); syncWrong();
  const ok = await postFeedback({ kind: "wrong_call", round_id: rid, pid: d.pid, correct, retracted: retract, where });
  if (!ok) { if (retract) state.flags[k] = correct; else delete state.flags[k]; renderCalls(); syncWrong(); setStatus("Couldn't save that correction."); }
}
// Ref's notes: the last call, one row per player, each with its own Wrong call control
function renderCalls() {
  const box = $("calls");
  const r = state.lastRound;
  if (!r || !r.decisions || !r.decisions.length) { box.hidden = true; return; }
  box.hidden = false;
  const photos = Object.fromEntries((state.board || []).map((p) => [p.id, p.photo]));
  box.innerHTML = `<p class="calls-head"><b>Last call</b> <span>${esc(r.call || "")}</span></p>`;
  const ol = document.createElement("ol");
  ol.className = "calls-list";
  for (const d of r.decisions) {
    const li = document.createElement("li");
    li.className = "call-row";
    li.dataset.testid = "call-row";
    const k = flagKey(r.round_id, d.pid);
    const flagged = k in state.flags;
    const verdict = d.seen === false ? "not seen" : d.out ? "out" : "safe";
    const img = photos[d.pid] ? `<img class="thumb" src="${esc(photos[d.pid])}" alt="">` : `<span class="thumb"></span>`;
    li.innerHTML = `${img}<b class="no">${d.number ?? ""}</b><span class="who">${esc(d.player)}</span>` +
      `<span class="given${d.out ? " is-out" : ""}">${flagged ? `${esc(verdict)} → ${esc(state.flags[k])}` : esc(verdict)}</span>`;
    if (d.seen !== false) {
      const b = document.createElement("button");
      b.type = "button"; b.className = "btn chip-btn";
      b.textContent = flagged ? "Undo" : "Wrong call";
      b.dataset.testid = "wrong-call";
      b.setAttribute("aria-label", flagged ? `undo wrong call for ${d.player}` : `wrong call for ${d.player}`);
      b.onclick = () => flagCall(r.round_id, d, "notes");
      li.append(b);
    }
    ol.append(li);
  }
  box.append(ol);
}
// on an out the call is about one player, so one tap flags them; when everyone was safe the tap opens Ref's
// notes to say who should have gone out
function showWrong(pid) {
  const w = $("wrong");
  const r = state.lastRound;
  if (!r || !r.decisions) { w.hidden = true; return; }
  w.dataset.pid = pid == null ? "" : String(pid);
  w.dataset.rid = r.round_id;
  w.hidden = false;
  syncWrong();
}
function syncWrong() {
  const w = $("wrong");
  if (w.hidden || !w.dataset.pid) { w.textContent = "Wrong call"; return; }
  w.textContent = flagKey(w.dataset.rid, w.dataset.pid) in state.flags ? "Noted ✓" : "Wrong call";
}
$("wrong").onclick = () => {
  const w = $("wrong");
  const r = state.lastRound;
  if (!r) return;
  if (!w.dataset.pid) {  // everyone safe: pick the player in the sheet
    if ($("notes-toggle").getAttribute("aria-expanded") !== "true") toggleNotes($("notes-toggle"), $("ref-notes"));
    return;
  }
  const d = r.decisions.find((x) => String(x.pid) === w.dataset.pid);
  if (d) flagCall(r.round_id, d, "verdict");
};
function showEndNote() {
  $("endnote").hidden = false;
  $("spark").setAttribute("hidden", "");
  $("note-text").value = "";
  $("note-status").textContent = "";
  $("note-send").disabled = false;
}
$("endnote").onsubmit = async (e) => {
  e.preventDefault();
  const text = $("note-text").value.trim();
  if (!text) { $("note-status").textContent = "Write something first."; return; }
  $("note-send").disabled = true;
  const ok = await postFeedback({ kind: "note", text, game_id: state.gameId });
  $("note-status").textContent = ok ? "Thanks, noted." : "Couldn't save it. Try again.";
  if (ok) $("note-text").value = ""; else $("note-send").disabled = false;
};

// ---------------------------------------------------------------- the call: kicker + one line, set over the bottom scrim
const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
function setMood(mood, kicker, cls) {
  const play = $("play");
  play.dataset.mood = mood;
  state.mood = mood;
  document.body.dataset.mood = mood;
  document.documentElement.dataset.mood = mood;
  $("kicker").innerHTML = kicker || "";
  $("kicker").className = `kicker${cls ? ` ${cls}` : ""}`;
  $("wrong").hidden = true;
  if (mood !== "win") $("result").hidden = true;
  renderStatus();
  drawRings();
}
function replay(el, cls) { el.classList.remove(cls); void el.offsetWidth; el.classList.add(cls); }
function setCall(html, cls) { const c = $("call"); c.innerHTML = html; c.className = cls; fitCall(); }
function fitCall() {  // largest size at which no word breaks and the call (big line + small line) fits 148 px
  const c = $("call");
  if (!c.offsetParent) return;
  const big = c.querySelector(".big");
  if (!big) return;
  const why = c.querySelector(".why");
  const maxH = 138, cw = c.clientWidth;
  let lo = 26, hi = c.classList.contains("win") ? 120 : 76;
  // block heights (line boxes), not scrollHeight: a .94 line-height lets glyph boxes poke out of the lines
  const fits = () => big.offsetHeight + (why ? why.offsetHeight + 10 : 0) <= maxH && big.scrollWidth <= cw + 1 &&
    [...big.querySelectorAll(".w")].every((w) => w.offsetWidth <= cw + 1);
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    big.style.fontSize = mid + "px";
    if (fits()) lo = mid; else hi = mid;
  }
  big.style.fontSize = lo + "px";
}
document.fonts && document.fonts.ready.then(() => { fitCall(); drawRings(); });

const words = (t) => esc(t).split(/\s+/).map((w) => `<span class="w">${w}</span>`).join(" ");
const dropStop = (t) => words(t.replace(/[.!]$/, "")) + (/[.!]$/.test(t) ? `<span class="sr">.</span>` : "");
function stageFor(msg) {
  const t = msg.text;
  if (CAPTIONS) { const sb = $("subs"); sb.textContent = t; sb.hidden = false; }
  if (msg.mood === "call") {
    const m = t.match(/^(simon says)[,:]?\s+(.*)$/i);
    if (m) { setMood("call", "Simon says"); return setCall(`<span class="sr">${esc(m[1])} </span><span class="big">${dropStop(m[2])}</span>`, "call"); }
    setMood("call", "No Simon", "no-simon"); return setCall(`<span class="big">${dropStop(t)}</span>`, "call simon-no");
  }
  if (msg.mood === "out") {
    const i = t.indexOf(",");
    const who = i > 0 ? t.slice(0, i) : t;
    let why = i > 0 ? t.slice(i + 1).trim() : "";
    const trick = /simon didn'?t say/i.test(why);
    const tail = why.match(/\s*you'?re out\.?$/i);
    if (tail) why = why.slice(0, tail.index);
    if (trick) why = "Simon didn't say it";
    setMood("out", trick ? `Out <span class="tagline">No Simon</span>` : "Out");
    const pl = (state.board || []).find((p) => p.call_name === who);
    showWrong(pl ? pl.id : null);
    return setCall(`<span class="big">${words(who)}</span><span class="sr">, ${esc(t.slice(i + 1).trim())}</span>${why ? `<span class="why" aria-hidden="true">${esc(why.charAt(0).toUpperCase() + why.slice(1))}</span>` : ""}`, "call out");
  }
  if (msg.mood === "safe") {
    if (state.simon === false) { setMood("trick", "Simon didn't say"); showWrong(null); return setCall(headline(t, "Nobody fell for it"), "call"); }
    setMood("safe", "Play on"); showWrong(null); return setCall(headline(t), "call");
  }
  if (msg.mood === "win") { setMood("win", "Full time"); return setCall(headline(t), "call"); }
  if (msg.mood === "warn") { setMood("warn", /lost you/i.test(t) ? "Lost you" : "Heads up"); return setCall(headline(t), "call"); }
  if (msg.mood === "rest") { setMood("rest", "Breather"); return setCall(headline(t), "call"); }
  setMood("info", "Listen up"); setCall(headline(t), "call");
}
// A spoken line is often two or three sentences; the glance needs one. The first sentence is set big and the
// rest small underneath.
function headline(t, override) {
  if (override) return `<span class="sr">${esc(t)}</span><span class="big" aria-hidden="true">${words(override)}</span><span class="why" aria-hidden="true">${esc(t)}</span>`;
  const m = t.match(/^(.+?[.!?:])\s+(.+)$/);
  if (!m || t.length <= 26) return `<span class="big">${dropStop(t)}</span>`;
  return `<span class="big">${words(m[1].replace(/[.:]$/, ""))}</span><span class="sr">. </span><span class="why">${esc(m[2])}</span>`;
}

// ---------------------------------------------------------------- voice
const queue = [];
let talking = false;
function speak(msg) {
  window.__says.push({ t: performance.timeOrigin + performance.now(), text: msg.text, mood: msg.mood, audio: msg.audio, seconds: msg.seconds });
  stageFor(msg);
  queue.push(msg);
  if (!talking) next();
}
function next() {
  const msg = queue.shift();
  if (!msg) { talking = false; return; }
  talking = true;
  if (msg.audio) {
    const a = new Audio(msg.audio);
    a.onended = a.onerror = next;
    a.play().catch(() => browserSay(msg.text, next));
  } else browserSay(msg.text, next, msg.seconds);
}
function browserSay(text, done, seconds) {
  if (MUTE || !("speechSynthesis" in window)) { setTimeout(done, (seconds || 1.5) * 1000); return; }
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 1.0; u.pitch = 0.9; u.lang = "en-US";
  u.onend = u.onerror = done;
  speechSynthesis.speak(u);
}
function unlockAudio() {  // iOS: audio must start from a tap
  if (MUTE) return;
  try { speechSynthesis.speak(new SpeechSynthesisUtterance("")); } catch (_) {}
  try { const a = new Audio("data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="); a.play().catch(() => {}); } catch (_) {}
}

// ---------------------------------------------------------------- camera
async function cameraOn() {
  if (!navigator.mediaDevices?.getUserMedia) { setStatus("This page needs HTTPS (or localhost) for the camera. See docs/phone-setup.md."); return; }
  if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: state.facing, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
  } catch (e) { setStatus(`Camera blocked: ${e.message}`); return; }
  $("preview").srcObject = state.stream;
  $("still").hidden = true;
  $("framer").classList.remove("empty");
  $("cam-hint").textContent = "";
  $("cam-on").hidden = true;
  state.statusMsg = null; guide();
  $("cam-flip").disabled = false;
  if (!state.sending) { state.sending = true; pump(); }
  updateStart();
}
function pump() {  // send ~FPS downscaled JPEGs; skip a beat if the network is behind
  const v = $("preview"), c = $("grab");
  const tick = () => {
    if (!state.stream) { state.sending = false; return; }
    const ws = state.ws;
    if (v.videoWidth && ws && ws.readyState === 1 && ws.bufferedAmount < 200000 && !state.info.replay) {
      const k = SIDE / Math.max(v.videoWidth, v.videoHeight);
      c.width = Math.round(v.videoWidth * k); c.height = Math.round(v.videoHeight * k);
      c.getContext("2d").drawImage(v, 0, 0, c.width, c.height);
      c.toBlob((b) => b && b.arrayBuffer().then((buf) => ws.readyState === 1 && ws.send(buf)), "image/jpeg", QUALITY);
    }
    setTimeout(tick, 1000 / FPS);
  };
  tick();
}

// ---------------------------------------------------------------- line-up: a ring forms around each person found
const short = (id) => String(id).replace(/^FALLBACK /, "FALLBACK ").replace(/(LiquidAI|facebook|PekingU|Qwen)\//g, "").replace(/-pretrain-lvd1689m/, "").replace(/\(Tinker tinker:\/\/[^)]*\)/, "(Tinker)");
function onLineup(msg) {
  if (msg.status === "follow") {  // the rings follow the players between Line up and Start (no new photos, names kept)
    if (state.lining || state.playing) return;
    const by = new Map((msg.players || []).map((p) => [p.id, p]));
    state.lineup.forEach((p) => { const u = by.get(p.id); if (u) { p.box = u.box || null; p.seen = u.seen; } });
    if (msg.view) setView(msg.view); else layoutScene();
    return;
  }
  if (msg.status === "looking") {
    state.lining = true;
    $("lineup").textContent = "Looking…";
    state.statusMsg = null;
    updateStart(); guide();
    return;
  }
  state.lining = false;
  const before = playing().length;
  state.lineup = msg.players || [];
  state.names = {}; state.skip = new Set();
  $("lineup").textContent = state.lineup.length ? "Line up again" : "Line up";
  state.lineupMs = msg.latency_ms && msg.latency_ms.total != null ? msg.latency_ms.total : null;
  $("lineup-status").textContent = state.lineup.length ? `${state.lineup.length} found` : "Nobody found";
  if (msg.view) { state.view = null; setView(msg.view); }
  if (state.lineup.length > before) flash($("found-flash"), `+${state.lineup.length - before}`);
  state.statusMsg = null;
  renderNotes();
  renderPlayers();
}
function renderPlayers() {
  const ul = $("player-list");
  ul.innerHTML = "";
  state.lineup.forEach((p) => {
    if (state.skip.has(p.id)) return;
    const li = document.createElement("li");
    li.className = "rost";
    li.dataset.testid = "lineup-player";
    const img = document.createElement("img");
    img.className = "photo"; img.alt = `Player ${p.number}`; img.decoding = "async";
    if (p.photo) img.src = p.photo;
    const no = document.createElement("b");
    no.className = "no"; no.textContent = p.number; no.setAttribute("aria-hidden", "true");
    const input = document.createElement("input");
    input.className = "name"; input.maxLength = 24; input.autocomplete = "off";
    input.placeholder = `Player ${p.number}`; input.value = state.names[p.id] || "";
    input.setAttribute("aria-label", `Name for player ${p.number}`);
    input.oninput = () => { state.names[p.id] = input.value.trim(); drawRings(); };
    li.append(img, no, input);
    if (state.info.stt === "faster-whisper") {
      const mic = document.createElement("button");
      mic.className = "mic"; mic.type = "button"; mic.textContent = "🎙";
      mic.title = "Say your name"; mic.setAttribute("aria-label", `say a name for player ${p.number}`);
      mic.onclick = () => recordName(mic, input, p);
      li.append(mic);
    }
    const x = document.createElement("button");
    x.className = "x"; x.title = "not playing"; x.type = "button";
    x.innerHTML = `<svg viewBox="0 0 10 10" aria-hidden="true"><path d="M1 1l8 8M9 1L1 9"/></svg>`;
    x.setAttribute("aria-label", `remove player ${p.number}`);
    x.onclick = () => { state.skip.add(p.id); renderPlayers(); };
    li.append(x);
    ul.append(li);
  });
  $("found-count").textContent = pad2(playing().length);
  layoutScene();
  updateStart();
  guide();
}
async function recordName(btn, input, p) {  // optional open STT on the server (faster-whisper)
  let rec, chunks = [];
  try {
    const s = await navigator.mediaDevices.getUserMedia({ audio: true });
    rec = new MediaRecorder(s);
    rec.ondataavailable = (e) => chunks.push(e.data);
    rec.onstop = async () => {
      s.getTracks().forEach((t) => t.stop());
      btn.textContent = "…";
      const fd = new FormData();
      fd.append("audio", new Blob(chunks, { type: rec.mimeType || "audio/webm" }), "say.webm");
      try {
        const r = await fetch("/api/stt", { method: "POST", body: fd });
        const j = await r.json();
        if (j.name) { input.value = j.name; state.names[p.id] = j.name; setStatus(`Heard: “${j.text}”`); drawRings(); }
      } catch (e) { setStatus("Couldn't hear that. Type it instead."); }
      btn.textContent = "🎙";
    };
    rec.start();
    btn.textContent = "■";
    setStatus(`Player ${p.number}: say your name… (2 s)`);
    setTimeout(() => rec.state === "recording" && rec.stop(), 2000);
  } catch (e) { setStatus("Mic blocked. Type it instead."); }
}
const playing = () => state.lineup.filter((p) => !state.skip.has(p.id));

// ---------------------------------------------------------------- screens
function updateStart() {
  const models = (state.stream || state.info.replay) && !state.info.loading && !state.info.model_error;
  $("lineup").disabled = !models || state.lining;
  $("start").disabled = !(models && !state.lining && playing().length > 0);
  // one primary at a time: Line up is the lime pill until someone is on the team sheet, then Start takes over
  const ready = playing().length > 0;
  $("lineup").classList.toggle("lead", !ready);
  $("start").classList.toggle("lead", ready);
}
// the status line: one plain sentence about what to do next (errors and replies replace it)
function guide() {
  if (state.statusMsg) { $("status").textContent = state.statusMsg; return; }
  const i = state.info || {};
  let t = "Connecting…";
  if (i.loading) t = "Warming up the referee…";
  else if (i.model_error) t = "The referee's models failed to load. See Ref's notes.";
  else if (i.model || i.models) {
    if (state.lining) t = "Hold still while Simon finds everyone.";
    else if (playing().length) t = "Ready. Tap a name to change it, then Start.";
    else if (!state.stream && !i.replay) t = "Turn the camera on, then get everyone in shot.";
    else t = "Ready. Get everyone in shot, then tap Line up.";
  }
  $("status").textContent = t;
}
function showPlay() {
  state.playing = true; $("lobby").hidden = true; $("play").hidden = false; $("stop").setAttribute("aria-label", "Stop"); $("result").hidden = true;
  $("endnote").hidden = true; $("spark").removeAttribute("hidden"); $("wrong").hidden = true; state.lastRound = null; renderCalls();
  $("subs").hidden = true;
  if (!$("play").dataset.mood || $("play").dataset.mood === "win") setMood("info", "Listen up");
  renderSpark();
  layoutScene();
  fitCall();
}
function showLobby() {
  state.playing = false; $("play").hidden = true; $("lobby").hidden = false; state.s = null;
  document.body.dataset.mood = document.documentElement.dataset.mood = "lobby"; $("play").dataset.mood = "";
  if (state.lineupView) setView(state.lineupView);
  layoutScene();
}
function setStatus(t) { state.statusMsg = t; $("status").textContent = t; }
let wakeLock = null;
async function keepAwake() { try { wakeLock = await navigator.wakeLock?.request("screen"); } catch (_) {} }

$("cam-on").onclick = cameraOn;
$("cam-flip").onclick = () => { state.facing = state.facing === "environment" ? "user" : "environment"; cameraOn(); };
$("lineup").onclick = () => { unlockAudio(); send({ type: "lineup" }); };
$("start").onclick = () => {
  unlockAudio(); keepAwake(); state.lineupView = state.view; showPlay(); setMood("info", "Listen up"); setCall(`<span class="big">Get ready</span>`, "call");
  send({ type: "start", names: state.names, skip: [...state.skip] });
};
$("stop").onclick = () => { if (state.playing) send({ type: "stop" }); showLobby(); };
$("dim").onclick = () => { $("shade").hidden = false; };
$("notes-toggle").onclick = () => toggleNotes($("notes-toggle"), $("ref-notes"));
$("lobby-notes-toggle").onclick = () => toggleNotes($("lobby-notes-toggle"), $("lobby-notes"));
$("shade").onclick = () => { $("shade").hidden = true; };
document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible" && state.playing) keepAwake(); });
$("framer").classList.add("empty");
document.body.classList.toggle("cc", CAPTIONS);  // captions mode: the spoken line is the subtitle, so the call drops its small line

connect();
