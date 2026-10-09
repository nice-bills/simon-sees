# Simon Sees: design direction

## Current: "the camera is the interface" (HUD pass, 8 Oct 2026)

**Reference:** the viral football-juggling counter (Hewad Mubariz's app, recreated by Ansh Nanda,
https://x.com/anshnanda/status/2107593775946051910). Side by side with ours: `hud-vs-reference.png`.
Every state rendered at 390×844 in Firefox: `hud/01-lobby.png` … `hud/24-winner-note.png`
(`node scripts/render_ui_states.mjs simon_sees/static docs/design/hud`).

What we took from it, and why it fits a referee better than the paper match card did:

- **The picture is the screen.** The camera (in a replay, the exact frame the referee judged) runs edge to
  edge. Players don't need to read a card to know who is out: they see a red ring break around a person.
- **A thin HUD floats on it.** Top left: a letter-spaced caps label (`SIMON SAYS / ROUND 3`), one huge
  condensed numeral (players still in), a tiny flash beside it (red `−1` when someone goes out, lime `+3` when
  the line-up finds people) and a caps sublabel (`PLAYERS IN`). Top right: the honest mode pill (`● REPLAY` /
  `● LIVE`), then two round translucent buttons (Ref's notes, close).
- **The tracked object gets a glowing ring.** Each player gets a rounded ring drawn from their SAM box (the
  server now sends `players[*].box` and `view` = the frame those boxes are in), a soft ground ellipse, and a dark
  chip above the head (`2 · AMA`). Colour is the verdict: white while watching (a lime scan pulse while
  judging), **lime** safe, **red** out (the ring breaks into dashes and fades, an `OUT · ROUND 3` stamp sits on
  the body), **amber** survived a "Simon didn't say" round, **dashed white + `LOST YOU`** when the referee
  can't see them. Players knocked out in earlier rounds leave the frame and become a quiet line of chips under
  the numeral. The winner's ring glows lime on the frozen frame with their chip filled lime.
- **The bottom is a waveform panel, not a card.** `REFEREE / Confidence` on the left, a status word on the
  right (`LISTEN`, `WATCHING`, `JUDGING…`, `SAFE ↑`, `OUT ↓`, `SAFE · NO SIMON`, `GAME OVER`) with the followed
  player under it; a sparkline of the referee's P(did it) per round for the most at-risk player (lime, with a
  playhead dot, red if they went out) over everyone else's faint lines, round ticks along the bottom; then a
  hairline and a three-up stat row: `ROUND 3/8`, `CALL TIME 0.41s` (the measured vision + referee time of the
  last judged look), `TIME 02:14` (game clock).
- **The call** is set over the bottom scrim in Barlow Condensed 600, auto-fitted (26–76 px; the winner's name
  up to 120 px) so no word ever breaks, with `text-wrap: balance`. A lime caps kicker above it says who spoke
  (`SIMON SAYS`); when Simon did *not* say it the kicker turns amber (`NO SIMON`) and the command is set in
  outline only, so the trick is visible at a glance without being given away by the voice.

### System

- **Type:** Barlow Condensed 500 (numerals, stats) and 600 (the call); Inter 500/600 for every label, chip and
  sentence. Labels are 10.5–11 px caps at +0.12–0.18em; numerals use tabular, lining figures. Anton is gone (it
  was the paper direction's poster face). All four files are self-hosted WOFF2 subsets in
  `simon_sees/static/fonts/` (SIL OFL 1.1; `OFL-Barlow.txt`, `OFL-Inter.txt`); the page needs no CDN.
- **Colour:** white type, one accent `#d4ff3a` (lime). Red `#ff4b4b` and amber `#ffb627` appear only as
  verdicts. No panels or cards over the picture: a top scrim, a tall bottom scrim and a vignette carry contrast
  (white labels sit on ≥ 60 % black where they meet bright sky or grass), plus a soft text shadow.
- **Framing:** a landscape replay frame can't fill a portrait phone without cutting players off, so the page
  scales the picture to fit everyone between the HUD's top and the call (`layoutScene`), shows it sharp there and
  lets it continue as a blurred, darkened copy behind (feathered seams). A portrait live camera that fits goes
  full-bleed. The rings are computed in the same transform, so they stay on the people at any size.
- **Spacing:** 24 px side margins, hairlines 1 px at 22 % white, 36 px round buttons, 52 px pill buttons.
- **Line-up:** a ring forms around each person as they're found (lime), numbered left to right, with
  tap-to-rename name chips (photo, number, name field, ×) in two columns over the bottom scrim. `LINE UP` is the
  lime pill until someone is found, then `START` takes over and Line up becomes `LINE UP AGAIN`.
- **Ref's notes** is a slide-up sheet (dark glass, 18 px top radius, grab handle), closed by default: the
  referee and vision models with latencies, the honest replay label, Dim screen, the last call with a
  `WRONG CALL` control per player, and with `?captions=1` the referee's decision table (P(did it) per player,
  SAM/DINO/d1 timings).
- **Feedback:** `WRONG CALL` sits at the right end of the kicker row on a verdict; at game over the sparkline
  makes way for a one-line note field (`ANYTHING SIMON GOT WRONG?` + `SEND NOTE`).
- **Captions mode** (`?captions=1`): the spoken line runs as a subtitle over the picture just above the call
  (the call then drops its small second line so nothing is said twice).
- **Accessibility:** the call is `aria-live="assertive"` (an out reads "Ama, you put your foot down. You're
  out."); status words and the line-up count are `role="status"`; the mode pill reads "Replay demo: recorded
  frames, not a live camera"; every icon button has a label; focus rings are lime.
  `prefers-reduced-motion: reduce` removes the ring break, scan pulse, flashes and sheet slide.

### Hooks

Kept: `replay-banner` (now the mode pill), `status`, `lineup`, `lineup-status`, `player-list`,
`lineup-player`, `player-card` (each ring's chip, `data-state` alive/out/lost), `board`, `call`, `phase`,
`result`, `meta`, `captions`, `calls`, `call-row`, `wrong-call`, `wrong-call-verdict`, `endnote`,
`note-status`, `ref-notes`, `lobby-notes`, the `Start` / `Line up` / `Ref's notes` / `Send note` buttons,
`Name for player N`, `remove player N` and the `Player N` photo alt. e2e changes: the replay check reads
"Replay demo"; the elimination check looks for the `Out · round N` stamp; the 🏆 check is gone (the winner is
`result` + the lime ring).

Server additions (additive only): `Player.box` (in every state broadcast), `state.view = {seq, w, h, url}`,
`lineup.view`, and `GET /api/view/<key>.jpg` (the judged frame, the line-up frame, or a replay's first frame).

---

## Earlier: "the referee's kit" (paper match card), superseded by the HUD pass above

Process (from Tobi Adonadon, "How to Stop Making your Apps Look Like AI Slop", 3 Oct 2026): concept → three
directions → design → render → critique → refine. Everything here was rendered at 390×844 CSS px in Firefox.

## The job of the screen

The phone sits on a stump 3–5 m from the players, outdoors, often in direct sun. Simon's voice runs the game.
The screen is **glanced at, not stared at**: from across a garden a player should be able to tell, in under a
second, *what was called* and *whether someone is out*. Everything else is secondary.

## Concept: the referee's kit

d1 is literally the referee of this game, so the interface borrows the one visual language every player
already reads from across a field: **the referee's cards and the match card.**

- **Paper match card + black ink** for everything routine: the lobby is a team sheet with shirt numbers; the
  call is printed huge in a condensed poster face; the board is the team sheet, ruled.
- **Full-bleed cards for verdicts.** The whole screen *becomes* the card, so the colour alone carries the
  news before anyone can read a word:
  - **Red card** = out (with an OUT stamp that lands on it)
  - **Yellow card** = "Simon didn't say!" (the trick round, when nobody fell for it)
  - **Green** = play on (everyone survived)
  - **Black** = heads up (the referee lost sight of someone)
- **Whistle** = the Start button. "Full time" = the end. The `?captions=1` view is the **referee's
  notebook**: d1's decision per player as a ruled table with a probability meter, latency in the header.

## The three directions explored

| | Mock | Idea | Verdict |
|---|---|---|---|
| A | `direction-a-referee-call.png`, `direction-a-referee-out.png` | Referee's kit: paper + ink, red/yellow cards, Anton + Barlow Condensed | **Chosen** |
| B | `direction-b-lanes-call.png`, `direction-b-lanes-out.png` | Sports-day track: turf green, painted white lanes, one lane per player, "DQ" for out, Big Shoulders Stencil | Rejected |
| C | `direction-c-chalk-call.png`, `direction-c-chalk-out.png` | Chalk on asphalt: charcoal ground, hopscotch grid of players, marker lettering | Rejected |

Side by side: `directions-compare.png`.

### Why A

1. **The metaphor is the mechanic, not decoration.** A red card *means* "you're out" to almost everyone on
   earth; a yellow card means "caught, warning". The verdict is legible as a colour from further away than any
   type can be. B's "DQ" and lanes need explaining (the players aren't racing) and C's hopscotch is cute but
   says nothing about who's out.
2. **Sunlight.** Dark ink on light matte paper is the most robust combination on a phone in direct sun (glare
   washes out dark screens first). Measured contrast: ink on paper 16.3:1, ink on yellow 13.1:1, paper on
   green 6.2:1, paper on red 5.1:1 (red is only ever used under display-size type). C's charcoal ground is the
   worst case outdoors; B's white-on-green is acceptable but every routine screen would be a saturated field.
3. **Letterforms at distance.** Anton is a tall, plain condensed grotesque: big x-height, no stencil breaks, no
   hand-drawn wobble. B's stencil cuts "TOES" into fragments at a glance, C's marker face and rotated lines cost
   legibility for charm.
4. **Room for the words.** A condensed face lets "STAND ON ONE FOOT" fill a 354 px column at ~100 px, which is
   roughly a 12–16 mm cap height on a real phone.

Taken from B: numbered rows per player (shirt numbers) for the team sheet. Taken from C: nothing visual, but the
reminder that it is a *playground* game, which is why the copy stays plain and a little cheeky.

## System

- **Type:** Anton (display: wordmark, call, OUT stamp, shirt numbers, round), Barlow Condensed 600/800
  (labels, names, buttons, notes), Barlow 500 (the lobby status sentence only). Four sizes: 14 px (small
  labels, the replay line, Ref's notes), 20 px (body, names, buttons, kicker), 32 px (wordmark, whistle, stamp)
  and display (the call, auto-fitted). Self-hosted WOFF2 subsets in `simon_sees/static/fonts/`
  (SIL OFL 1.1, licences alongside). No CDN; the page works offline.
- **Colour:** paper `#f3efe4`, ink `#121212`, soft ink `#4d493f`, red `#c8102e`, yellow `#ffd400`,
  green `#0a6633`. Four card colours, each with one meaning. No gradients anywhere.
- **Spacing:** 8 px scale only (8 / 16 / 24 / 32 / 48); 16 px side margin; one rule weight (3 px).
- **One primary thing per screen, at most one secondary action.** Lobby: the team sheet and one action bar
  (Line up is primary until someone is on the sheet, then Start). Play: the call; Stop is the only button.
- **Player cards:** one style everywhere (lobby and board): an even grid of 3:4 photos (square past four
  players), top-weighted crop, the shirt number on the photo's corner, the name underneath. State lives on the
  photo: out = greyscale + red band "Out, round N"; lost from view = dashed frame + "Lost you".
- **Ref's notes:** a drawer, closed by default, for the machinery: model names, where they run, latencies,
  voice, line-up timing, and (with `?captions=1`) d1's per-player decision table. In play it rises over the
  board instead of pushing the layout. Dim screen lives here too.
- **Replay honesty:** in a replay one small quiet line at the top says "Replay demo · recorded frames, not a
  live camera".
- **The call is auto-fitted:** a binary search picks the largest size (24–176 px) at which no word breaks and
  the call fits its space. Multi-sentence lines set the first sentence big and the rest at 20 px underneath.
- **Kicker** (the black tag above the call) names the moment: LISTEN UP, SIMON SAYS, PLAY ON, SIMON DIDN'T
  SAY!, HEADS UP, BREATHER, WINNER. A trick call has **no** kicker, exactly as Simon didn't say it; a plain
  out has none either, because the red screen is the red card.
- **Motion (purposeful only):** the kicker and call rise 24 px into place on each new line; on an elimination
  the OUT stamp slams onto the photo and the screen gives a short thud; a blinking dot while d1 is watching; a
  sweeping bar while it judges. `prefers-reduced-motion: reduce` turns all of it off.

## Hooks kept

All element IDs, `data-testid`s and button names the server, the e2e suite and the demo recorder use are
unchanged (`replay-banner`, `status`, `lineup`, `lineup-status`, `player-list`, `lineup-player`, `player-card`,
`call`, `board`, `phase`, `result`, `captions`, `meta`, `Start`, `Line up`, `remove player N`,
`Name for player N`, `Player N` photo alt). New: `ref-notes` / `lobby-notes` (the drawers) and their
`Ref's notes` toggle buttons. The stack line moved from `status` into `lobby-notes`, and the captions table
sits inside `ref-notes`, so the e2e suite opens the drawer before asserting on either. Because labels are now set in
capitals with CSS and the e2e runner compares rendered text, the e2e text assertions use `ignoreCase`.
`?mute=1` (new) keeps the phone silent; the demo recorder uses it so headless Firefox never raises its
"speech-dispatcher missing" bar, and `scripts/make_demo.py` adds Simon's voice afterwards.
