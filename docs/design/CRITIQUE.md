# Render → critique → refine

All renders are 390×844 CSS px, Firefox, `prefers-reduced-motion: reduce` (so they show resting states), made by
driving the real `index.html` + `app.js` with scripted server messages. Folders:

- `before/v0-original-*.png`: the original UI (dark background, rounded cards, monospace log).
- `before/v1-first-pass-*.png`: the first pass of the referee's-kit direction.
- `before/before-*.png`: the referee's kit as shipped before the clarity pass (the version called "cluttered
  and sloppy"), every state, with real photo cards.
- `after/after-*.png`: after the clarity pass, every state. `after-overview.png` is all of them on one sheet;
  `before-after-compare.png` puts eight key states side by side.

States: 01 lobby · 02 line-up with photo cards · 03 listen up · 04 Simon says call · 05 watching · 06 judging ·
07 safe (play on) · 08 out · 09 trick call · 10 out on a trick (Simon didn't say) · 11 trick survived ·
12 heads up (player lost from view) · 13 breather · 14 winner · 15–18 captions mode (waiting, call, out,
winner) · 19 captions mode with Ref's notes open · 20 five players (one out, one lost).
Render with `node scripts/render_ui_states.mjs simon_sees/static docs/design/after after-`; the script also
prints a `LAYOUT` line for any state where the page is wider than the phone, the play screen scrolls, or the
call overflows its box (none do). Photos come from `scripts/fixtures/ui-photos.json` (crops of
`samples/people/000_lineup.jpg`; #2 is deliberately a wider crop, as real SAM 3 crops vary).

## Clarity pass: critique of `before/before-*.png`

Honest list of what made it read as cluttered and sloppy on a 390×844 phone:

| Problem | Where | Fix |
|---|---|---|
| Debug text in the glance zone: the scorebar carried the model id and latency (`ROUND 2/8 · D1-OMNI-600M · 412 MS`) on every screen; the lobby footer read `MOCK READY · VOICE: PHONE · DRY RUN (REPLAYED FRAMES)`, repeating the banner. | every play state, lobby | Scorebar is just `ROUND 2/8` + phase. Models, latencies, voice, backend and line-up timing live in **Ref's notes**, a drawer closed by default. |
| Captions mode dumped a two-header confidence table (`D1 DECISION · ROUND 2`, `SAM 1830 · DINO 240 · MATCH 2 MS · ROUND TRIP 2690`, P(DID IT) meters) between the call and the board, shrinking the call and the red-card photo. | 15–18 | The table moved into Ref's notes; captions mode looks like play until you open the drawer (19). |
| The replay label was a yellow-on-black strip shouting in caps on every screen. | all | One small quiet line, same words. |
| The red card stacked four things that all said "out": RED CARD kicker, photo, name + reason, and a big rotated OUT stamp floating on its own. | 08, 17 | The red screen is the red card: no kicker; the OUT stamp sits on the corner of the player's photo; then the name; then why. |
| Duplicate status: kicker "LISTEN UP" + phase "LISTEN UP"; kicker "FULL TIME · WINNER" + phase "GAME OVER". | 03, 14 | The phase line only says what the kicker doesn't (watching, judging, game over). Winner kicker is just WINNER. |
| Player photos were 34×44 px mugshots in ruled rows (28×36 with 5+ players), unreadable from 3 m; with 5 players the two-column board truncated names to "E:" next to a dashed LOST YOU chip. | board, 20 | One card style everywhere: an even grid of 3:4 photos (square crops past four players), same top-weighted crop, the shirt number on the photo's corner and the name underneath. Out = greyed photo + red "Out, round N" band; lost = dashed frame + "Lost you" band. |
| The lobby was a form: wordmark + lede, two numbered sections with paragraph hints (one naming SAM 3 → DINO), camera buttons, line-up button + status, roster rows, then Start + status. Three button styles competed. | 01, 02 | Wordmark, camera, one plain status sentence that says what to do next, the team sheet as photo cards, and one sticky action bar. Line up is solid until someone is on the sheet, then it turns to outline and Start becomes the solid primary. In a replay the camera buttons are replaced by "Replaying recorded frames". |
| Two footer buttons (Dim screen, Stop) on every play screen. | play | One secondary action (Stop / Back). Dim screen moved into Ref's notes. |
| Ad-hoc spacing (6, 7, 9, 10, 12, 14, 18, 22, 26 px) and seven-plus type sizes (12–30 px labels plus display), rules of 2 and 4 px. | all | 8 px scale only (8/16/24/32/48), four type sizes (14 / 20 / 32 / display), one 3 px rule. |
| Long lines wrapped raggedly ("then Line / up."). | lobby | `text-wrap: balance` on the status, kicker and secondary lines; shorter copy. |

### Remaining, knowingly

- The call is auto-fitted, so "display" is one size per screen rather than one size globally: that is what
  lets RELAX. be enormous and the welcome line still fit.
- Dim screen is now one tap deeper (inside Ref's notes). It is rarely used mid-game; Stop is not.
- Captions mode (`?captions=1`) no longer shows decisions without a tap. The demo recording opens the drawer
  once so the real per-player decisions are on camera.

## History: the first referee's-kit pass

### Critique of the original UI against the article's tells

| Tell | Original |
|---|---|
| Generic component patterns | Dark green "app" background, rounded cards, pill chips, rounded green buttons: a stock template. |
| Weak hierarchy | The call, the player grid and the log all compete; the out line is the same size as a normal call, only red. |
| Repetitive cards | Every player is an identical rounded card; setup steps are cards inside the page. |
| Template layout | Centred stack of cards; "Dim screen / Stop" pills; meta text in grey under everything. |
| Debug box | Captions are a monospace terminal box. |
| Outdoors | Dark screen and mid-grey text are the first thing sunlight erases. |

### Critique of the first pass (v1) and what changed

| Problem in v1 | Tell | Fix in final |
|---|---|---|
| The welcome line and the trick-survived line became walls of display type (`v1-first-pass-03`, `-11`). | Weak hierarchy | Multi-sentence lines: first sentence big, the rest in Barlow Condensed underneath. The trick-survived line gets the headline "NOBODY FELL FOR IT" over the spoken line. |
| OUT stamp sat on top of the reason text (`v1-first-pass-08`). | Collision | Stamp moved into the flow under the call; the auto-fit shrinks the name to make room. Smaller stamp in captions mode. |
| Captions mode overflowed the viewport: Dim/Stop pushed off-screen (`v1-first-pass-17`). | Inconsistent spacing | Play screen is a fixed-height column; the call area absorbs the difference and re-fits. |
| Scorebar meta truncated ("LAST DECI…"). | Noise in the glance zone | Shortened to "ROUND 2/8 · d1-omni-600M · 412 MS". |
| "P(DID IT)" header and "UNSURE → SAFE" wrapped in the notebook; notebook header could run off the edge. | Inconsistent spacing | No-wrap headers, wider verdict column, ellipsis on the long half of the header. |
| Winner said the name twice (huge name + a "WINNER: RED SHIRT" slab). | Repetition | Kicker reads "FULL TIME · WINNER"; the `#result` line stays for screen readers and tests but is visually hidden. |
| "JUDGING" had no presence; with an outline it overflowed the edge. | Weak state signal | Judging = boxed chip + sweeping bar; watching = blinking dot. Fixed scorebar height so nothing jumps. |
| Remove "×" in the lobby was a tiny glyph. | Touch target | 44×44 px target. |
| Trailing full stops on calls ("TOES.") added noise at 100+ px. | Polish | Kept for screen readers, hidden visually on calls. |

### Remaining then

- The emoji trophy on the winner screen is the platform's colour emoji (the e2e suite asserts 🏆 in the call),
  so it differs by OS.
- Red card text is 5.1:1; it is only used for display-size type, where that is comfortably legible, but it is
  the weakest pair in the system.
- Not tested on a physical phone in sunlight; contrast numbers are computed, not measured outdoors.
