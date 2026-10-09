// Renders every UI state of the Simon Sees phone client at 390x844 by faking the WebSocket.
// usage: node scripts/render_ui_states.mjs simon_sees/static docs/design/after after- [onlyState]
// Needs e2e/node_modules (npm ci) and a Playwright firefox. Unset HTTP(S)_PROXY if a proxy swallows 127.0.0.1.
import { createRequire } from 'node:module';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
const require = createRequire(new URL('../e2e/package.json', import.meta.url));
const { firefox } = require('playwright-core');
const [staticDir, outDir, prefix = '', only] = process.argv.slice(2);
fs.mkdirSync(outDir, { recursive: true });
const types = { '.html': 'text/html', '.css': 'text/css', '.js': 'text/javascript', '.woff2': 'font/woff2', '.svg': 'image/svg+xml' };
// the scene: /samples/* and /api/view/* serve the replay frames (samples/people), like the server's /api/view
const samples = new URL('../samples/people/', import.meta.url).pathname;
const srv = http.createServer((req, res) => {
  let p = req.url.split('?')[0];
  if (p.startsWith('/api/view/')) p = '/samples/000_lineup.jpg';
  p = p === '/' ? path.join(staticDir, 'index.html') : p.startsWith('/samples/') ? path.join(samples, p.slice(9)) : path.join(staticDir, p.replace(/^\/static\//, ''));
  if (!fs.existsSync(p)) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { 'content-type': types[path.extname(p)] || 'application/octet-stream' });
  res.end(fs.readFileSync(p));
}).listen(0);
const port = srv.address().port;
const init = () => {
  window.__sent = [];
  class FakeWS { constructor() { this.readyState = 1; this.bufferedAmount = 0; window.__ws = this; setTimeout(() => this.onopen && this.onopen(), 0); }
    send(d) { window.__sent.push(d); } close() {} }
  window.WebSocket = FakeWS;
  window.__push = (m) => window.__ws.onmessage({ data: JSON.stringify(m) });
  Object.defineProperty(window, 'speechSynthesis', { configurable: true, get: () => ({ speak: (u) => setTimeout(() => u.onend && u.onend(), 50), cancel() {} }) });
  window.SpeechSynthesisUtterance = function (t) { this.text = t; };
};
// line-up photos: crops of samples/people/000_lineup.jpg (made by the mock pipeline; #2 deliberately a different
// aspect ratio, like real SAM crops), so every render shows the photo cards the way the server sends them
const PHOTOS = JSON.parse(fs.readFileSync(new URL('./fixtures/ui-photos.json', import.meta.url)));
const photo = (i) => PHOTOS[i % PHOTOS.length];
// real person boxes for each replay frame (RT-DETR on the 640 px frames: scripts/fixtures/sample-boxes.json),
// mapped to player ids: in 004 the players swapped places, in 006 the middle player left the frame
const BOXES = JSON.parse(fs.readFileSync(new URL('./fixtures/sample-boxes.json', import.meta.url)));
const ORDER = { '004_swapped-places_stand.jpg': [1, 2, 0], '006_centre-player-left-frame.jpg': [0, null, 1] };
const boxFor = (frame, i) => { const o = ORDER[frame]; const k = o ? o[i] : i; return k == null ? null : (BOXES[frame] || [])[k] ?? null; };
let FRAME = '000_lineup.jpg';
const P = (names, outs = {}, lost = [], frame = FRAME) => names.map((n, i) => ({ id: i, number: i + 1, display_name: n, call_name: n, alive: !(n in outs), out_round: outs[n] ?? null,
  seen: !lost.includes(n), photo: photo(i), position: ['left', 'middle', 'right', 'far_left'][i], named: !/^(Number|Player) \d+$/.test(n),
  box: boxFor(frame, i) || (lost.includes(n) ? boxFor('000_lineup.jpg', i) : null) }));
const view = (frame) => ({ seq: 1, w: 640, h: 478, url: `/samples/${frame}` });
const info = { type: 'info', replay: true, model: 'mock', tts: 'browser', stt: null, loading: false };
const names = ['Kofi', 'Ama', 'Number 3'];
const st = (o, frame = '001_lineup.jpg') => ({ type: 'state', game_id: 'x', phase: 'calling', round: 2, max_rounds: 8, model: 'LiquidAI/d1-omni-600M', players: P(names, {}, [], frame),
  winners: [], last_latency_ms: 412.3, last_round: {}, simon: true, view: view(frame), ...o });
// earlier rounds, so the referee's sparkline has a history (each state replaces last_round; the page keeps the series)
const hist = (upto) => [[1, true, [0.86, 0.71, 0.9]], [2, false, [0.12, 0.3, 0.08]], [3, true, [0.91, 0.22, 0.83]], [4, true, [0.78, 0.6, 0.88]]]
  .filter(([r]) => r < upto).map(([r, simon, ps]) => st({ phase: 'rest', round: r, last_round: { round_id: `x-r0${r}`, round: r, call: '', kind: 'pose', simon, latency_ms: 400, breakdown: {},
    decisions: ps.map((p, i) => ({ player: names[i], number: i + 1, pid: i, p, out: false, seen: true })) } }));
const lr = (simon, decs, call = '') => ({ round_id: 'x-r02', round: 2, call, kind: 'pose', simon, latency_ms: 412.3, breakdown: bd,
  decisions: decs.map((d) => ({ number: names.indexOf(d.player) + 1, pid: names.indexOf(d.player), seen: true, ...d })) });
const say = (text, mood) => ({ type: 'say', text, mood, audio: null, seconds: 1 });
// the line-up: the server's answer to "Line up" (one photo per person, numbered left to right)
const lobbyPlayers = async (page) => {
  await page.evaluate(([ph, bx]) => window.__push({ type: 'lineup', players: ph.slice(0, 3).map((p, i) => ({ id: i, number: i + 1, photo: p, box: bx[i] })),
    latency_ms: { total: 2140 }, view: { w: 640, h: 478, url: '/samples/000_lineup.jpg' } }), [PHOTOS, BOXES['000_lineup.jpg']]);
  await page.evaluate(() => { const i = document.querySelector('[aria-label="Name for player 1"]'); i.value = 'Kofi'; i.dispatchEvent(new Event('input')); });
};
const lr3 = (simon, decs, call = '') => ({ ...lr(simon, decs, call), round_id: 'x-r03', round: 3 });
const lr4 = (simon, decs, call = '') => ({ ...lr(simon, decs, call), round_id: 'x-r04', round: 4 });
const typeNote = async (page) => { await page.evaluate(() => { const t = document.getElementById('note-text'); t.value = 'The squat call on Ama was harsh'; }); };
const openNotes = async (page) => { await page.evaluate(() => document.getElementById('notes-toggle')?.click()); };
const five = ['Kofi', 'Ama', 'Number 3', 'Esi', 'Number 5'];
const bd = { segment: 1830, embed: 240, match: 2, round_trip: 2690 };
const states = [
  ['01-lobby', '', [], null],
  ['02-lobby-players', '', [], lobbyPlayers],
  ['03-listen-up', '', [st({ phase: 'intro', round: 0, simon: null, last_latency_ms: null }, '000_lineup.jpg'), say('Welcome to Simon Sees. Players: Kofi, Ama, Number 3. Only do it if Simon says. Let\'s look at you.', 'info')]],
  ['04-call-simon', '', [...hist(2), st({ phase: 'calling' }), say('Simon says touch your toes.', 'call')]],
  ['05-watching', '', [...hist(2), st({ phase: 'calling' }), say('Simon says touch your toes.', 'call'), st({ phase: 'watching' })]],
  ['06-judging', '', [...hist(2), st({ phase: 'calling' }), say('Simon says touch your toes.', 'call'), st({ phase: 'judging' })]],
  ['07-safe', '', [...hist(2), st({ phase: 'judging', last_round: lr(true, [{ player: 'Kofi', p: 0.91, out: false }, { player: 'Ama', p: 0.72, out: false }, { player: 'Number 3', p: 0.88, out: false }], 'Simon says touch your toes.') }), say('Sharp. You all survive.', 'safe')]],
  ['08-out', '', [...hist(3), st({ phase: 'calling', round: 3 }, '002_onefoot_hands-head_stand.jpg'), say('Simon says stand on one foot.', 'call'), st({ phase: 'judging', round: 3, players: P(names, { 'Ama': 3 }, [], '002_onefoot_hands-head_stand.jpg'), last_round: lr3(true, [{ player: 'Kofi', p: 0.91, out: false }, { player: 'Ama', p: 0.08, out: true }, { player: 'Number 3', p: 0.31, out: false, uncertain: true }]) }, '002_onefoot_hands-head_stand.jpg'), say('Ama, you put your foot down. You\'re out.', 'out')]],
  ['09-trick-call', '', [...hist(4), st({ phase: 'calling', simon: false, round: 4, players: P(names, { 'Ama': 3 }, [], '002_onefoot_hands-head_stand.jpg') }, '002_onefoot_hands-head_stand.jpg'), say('Hands on your head!', 'call')]],
  ['10-trick-out', '', [...hist(4), st({ phase: 'judging', simon: false, round: 4, players: P(names, { 'Ama': 3, 'Number 3': 4 }, [], '003_arms-up_stand_arms-up.jpg'), last_round: lr4(false, [{ player: 'Kofi', p: 0.1, out: false }, { player: 'Number 3', p: 0.93, out: true }]) }, '003_arms-up_stand_arms-up.jpg'), say('Number 3, Simon didn\'t say! You\'re out.', 'out')]],
  ['11-trick-safe', '', [...hist(4), st({ phase: 'judging', simon: false, round: 4, players: P(names, { 'Ama': 3 }, [], '004_swapped-places_stand.jpg'), last_round: lr4(false, [{ player: 'Kofi', p: 0.06, out: false }, { player: 'Number 3', p: 0.21, out: false }]) }, '004_swapped-places_stand.jpg'), say('Ha. I didn\'t say Simon says, and none of you fell for it.', 'safe')]],
  ['12-warn', '', [...hist(2), st({ phase: 'judging', players: P(names, {}, ['Ama'], '006_centre-player-left-frame.jpg'), last_round: lr(true, [{ player: 'Kofi', p: 0.9, out: false }, { player: 'Ama', p: null, out: false, seen: false }, { player: 'Number 3', p: 0.84, out: false }]) }, '006_centre-player-left-frame.jpg'), say('Heads up, Ama: I lost you. Get back in the picture.', 'warn')]],
  ['13-rest', '', [...hist(3), st({ phase: 'rest' }), say('Relax.', 'rest')]],
  ['14-winner', '', [...hist(5), st({ phase: 'final', round: 5, players: P(names, { 'Ama': 3, 'Number 3': 4 }, [], '006_centre-player-left-frame.jpg') }, '006_centre-player-left-frame.jpg'), say('Kofi wins! Now go touch some grass, all of you.', 'win'),
    st({ phase: 'over', round: 5, winners: ['Kofi'], players: P(names, { 'Ama': 3, 'Number 3': 4 }, [], '006_centre-player-left-frame.jpg') }, '006_centre-player-left-frame.jpg')]],
  ['15-captions-waiting', '?captions=1', [st({ phase: 'intro', round: 0, simon: null }, '000_lineup.jpg'), say('Welcome to Simon Sees. Players: Kofi, Ama, Number 3. Only do it if Simon says. Let\'s look at you.', 'info')]],
  ['16-captions-call', '?captions=1', [...hist(3), st({ phase: 'calling', round: 3 }), say('Simon says stand on one foot.', 'call')]],
  ['17-captions-out', '?captions=1', [...hist(3), st({ phase: 'judging', round: 3, players: P(names, { 'Ama': 3 }, [], '002_onefoot_hands-head_stand.jpg'), last_round: lr3(true, [{ player: 'Kofi', p: 0.91, out: false }, { player: 'Ama', p: 0.08, out: true }, { player: 'Number 3', p: 0.88, out: false }]) }, '002_onefoot_hands-head_stand.jpg'), say('Ama, you put your foot down. You\'re out.', 'out')]],
  ['18-captions-winner', '?captions=1', [...hist(5), st({ phase: 'final', round: 5, players: P(names, { 'Ama': 3, 'Number 3': 4 }, [], '006_centre-player-left-frame.jpg'), last_round: lr(false, [{ player: 'Kofi', p: 0.03, out: false }, { player: 'Number 3', p: 0.97, out: true }]) }, '006_centre-player-left-frame.jpg'), say('Kofi wins! Now go touch some grass, all of you.', 'win'),
    st({ phase: 'over', round: 5, winners: ['Kofi'], players: P(names, { 'Ama': 3, 'Number 3': 4 }, [], '006_centre-player-left-frame.jpg'), last_round: lr(false, [{ player: 'Kofi', p: 0.03, out: false }, { player: 'Number 3', p: 0.97, out: true }]) }, '006_centre-player-left-frame.jpg')]],
  ['19-captions-notes-open', '?captions=1', [...hist(3), st({ phase: 'judging', round: 3, players: P(names, { 'Ama': 3 }, [], '002_onefoot_hands-head_stand.jpg'), last_round: lr3(true, [{ player: 'Kofi', p: 0.91, out: false }, { player: 'Ama', p: 0.08, out: true }, { player: 'Number 3', p: 0.62, out: false, uncertain: true }]) }, '002_onefoot_hands-head_stand.jpg'), say('Ama, you put your foot down. You\'re out.', 'out')], openNotes],
  // playtest feedback: "Wrong call" on the verdict, per player in Ref's notes, and the end-of-game note
  ['21-out-wrong-call', '', [...hist(3), st({ phase: 'judging', round: 3, players: P(names, { 'Ama': 3 }, [], '002_onefoot_hands-head_stand.jpg'), last_round: lr3(true, [{ player: 'Kofi', p: 0.91, out: false }, { player: 'Ama', p: 0.08, out: true }, { player: 'Number 3', p: 0.88, out: false }], 'Simon says stand on one foot.') }, '002_onefoot_hands-head_stand.jpg'), say('Ama, you put your foot down. You\'re out.', 'out')]],
  ['22-safe-wrong-call', '', [...hist(2), st({ phase: 'judging', last_round: lr(true, [{ player: 'Kofi', p: 0.91, out: false }, { player: 'Ama', p: 0.72, out: false }, { player: 'Number 3', p: 0.88, out: false }], 'Simon says touch your toes.') }), say('Sharp. You all survive.', 'safe')]],
  ['23-notes-calls', '', [...hist(3), st({ phase: 'judging', round: 3, players: P(names, { 'Ama': 3 }, [], '002_onefoot_hands-head_stand.jpg'), last_round: lr3(true, [{ player: 'Kofi', p: 0.91, out: false }, { player: 'Ama', p: 0.08, out: true }, { player: 'Number 3', p: 0.88, out: false }], 'Simon says stand on one foot.') }, '002_onefoot_hands-head_stand.jpg'), say('Ama, you put your foot down. You\'re out.', 'out')], openNotes],
  ['24-winner-note', '', [...hist(5), st({ phase: 'final', round: 5, players: P(names, { 'Ama': 3, 'Number 3': 4 }, [], '006_centre-player-left-frame.jpg') }, '006_centre-player-left-frame.jpg'), say('Kofi wins! Now go touch some grass, all of you.', 'win'),
    st({ phase: 'over', round: 5, winners: ['Kofi'], players: P(names, { 'Ama': 3, 'Number 3': 4 }, [], '006_centre-player-left-frame.jpg') }, '006_centre-player-left-frame.jpg')], typeNote],
  ['20-five-players', '', [st({ phase: 'calling', round: 4, players: P(five, { 'Ama': 3 }, ['Esi']) }), say('Simon says stand on one foot.', 'call')]],
];
const b = await firefox.launch();
for (const [name, q, msgs, act] of states) {
  if (only && !name.includes(only)) continue;
  const ctx = await b.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 1, reducedMotion: 'reduce' });
  const page = await ctx.newPage();
  page.on('pageerror', (e) => console.log('pageerror', name, e.message)); page.on('console', (m) => m.type() === 'error' && console.log('console', name, m.text()));
  await page.addInitScript(init);
  await page.goto(`http://127.0.0.1:${port}/${q}`);
  await page.waitForFunction(() => window.__ws);
  await page.waitForTimeout(200);
  await page.evaluate((m) => window.__push(m), info);
  if (act && !msgs.length) await act(page);
  if (msgs.length) {
    await page.evaluate(() => { const s = document.getElementById('start'); s.disabled = false; s.click(); });
    for (const m of msgs) { await page.evaluate((m) => window.__push(m), m); await page.waitForTimeout(60); }
    if (act) await act(page);
  }
  await page.waitForTimeout(900);
  // layout check: nothing wider than the phone, the play screen fits without scrolling, the call fits its box
  const bad = await page.evaluate(() => {
    const out = [];
    if (document.documentElement.scrollWidth > innerWidth) out.push(`page ${document.documentElement.scrollWidth}px wide`);
    const play = document.getElementById('play');
    if (!play.hidden && play.scrollHeight > play.clientHeight + 1) out.push(`play overflows by ${play.scrollHeight - play.clientHeight}px`);
    const c = document.getElementById('call');
    // block heights, not scrollHeight: the call's .94 line-height lets glyph boxes poke out of their lines
    const h = [...c.children].filter((e) => getComputedStyle(e).position !== 'absolute' && e.offsetHeight).reduce((t, e) => t + e.offsetHeight + parseFloat(getComputedStyle(e).marginTop), 0);
    if (!play.hidden && (c.scrollWidth > c.clientWidth + 1 || h > c.clientHeight + 1)) out.push(`call overflows (${h} > ${c.clientHeight})`);
    for (const t of document.querySelectorAll('.chip, .label, .status-word, .stat, .btn, .kicker')) {  // nothing clipped or off-screen
      const r = t.getBoundingClientRect();
      if (r.width && t.offsetParent && (r.right > innerWidth + 0.5 || r.left < -0.5)) out.push(`off-screen: ${t.textContent.trim().slice(0, 24)}`);
    }
    return out;
  });
  if (bad.length) console.log('LAYOUT', name, bad.join('; '));
  await page.screenshot({ path: path.join(outDir, `${prefix}${name}.png`) });
  await ctx.close();
}
await b.close(); srv.close(); console.log('done');
