// Deterministic end-to-end flow of the phone client: lobby -> line up (everyone's photo becomes their card)
// -> name players -> start -> rounds -> eliminations -> winner. Locators and assertions only; no model calls
// (the server runs with mocked vision + the seeded mock decider, see e2e.config.ts).
// The page sets labels in capitals with CSS (text-transform) and the runner reads rendered text, so text
// assertions compare case-insensitively.
import { test } from '@e2e-dev/web';
import { expect } from 'e2e';

test('lobby shows the replay pill and the vision stack is ready', async ({ app, screen }) => {
  await app.open('/');
  // the honest mode pill, top right: "● REPLAY" (its accessible text says "Replay demo: recorded frames, not a live camera")
  await expect(screen.getByTestId('replay-banner')).toBeVisible();
  await expect(screen.getByTestId('replay-banner')).toContainText('Replay demo', { ignoreCase: true });
  await expect(screen.getByTestId('status')).toContainText('Ready', { ignoreCase: true });
  // the vision stack lives in Ref's notes, a slide-up sheet that is closed until asked for
  await screen.getByRole('button', "Ref's notes").click();
  await expect(screen.getByTestId('lobby-notes')).toContainText('mock → mock → mock ready', { ignoreCase: true });
  // nobody has lined up yet: Start stays disabled, and there is no clothing field anywhere
  await expect(screen.getByRole('button', 'Start')).toBeDisabled();
  await expect(screen.getByRole('button', 'Line up')).toBeEnabled();
  await expect(screen.getByText(/what you're wearing/i)).toHaveCount(0);
});

test('line up: each person gets a ring and a name chip with their photo, numbered left to right, and can be named or removed', async ({ app, screen }) => {
  await app.open('/');
  await screen.getByRole('button', 'Line up').click();
  await expect(screen.getByTestId('lineup-player')).toHaveCount(3, { timeout: 15_000 });
  await expect(screen.getByTestId('lineup-status')).toContainText('3 found', { ignoreCase: true });
  await expect(screen.getByRole('img', 'Player 1')).toBeVisible();
  await expect(screen.getByRole('img', 'Player 3')).toBeVisible();
  await screen.getByLabel('Name for player 1').fill('Kofi');
  await screen.getByRole('button', 'remove player 3').click();
  await expect(screen.getByTestId('lineup-player')).toHaveCount(2);
  await expect(screen.getByRole('button', 'Start')).toBeEnabled();
});

test('a full replay game: rounds, eliminations and a winner, called by name or number', async ({ app, screen }) => {
  await app.open('/?captions=1');
  await screen.getByRole('button', 'Line up').click();
  await expect(screen.getByTestId('lineup-player')).toHaveCount(3, { timeout: 15_000 });
  await screen.getByLabel('Name for player 1').fill('Kofi');
  await screen.getByLabel('Name for player 2').fill('Ama');
  await screen.getByRole('button', 'Start').click();

  // the play screen takes over: one ring + chip per player, drawn over the judged frame
  await expect(screen.getByTestId('player-card')).toHaveCount(3);
  await expect(screen.getByTestId('board')).toContainText('Kofi', { ignoreCase: true });
  await expect(screen.getByTestId('board')).toContainText('Player 3', { ignoreCase: true });
  await expect(screen.getByTestId('call')).toContainText('Welcome to Simon Sees', { ignoreCase: true, timeout: 15_000 });

  // a round is called and the referee's decisions appear in Ref's notes, with the SAM -> DINO -> match timings
  await expect(screen.getByTestId('meta')).toContainText('round 1/8', { ignoreCase: true, timeout: 20_000 });
  await screen.getByRole('button', "Ref's notes").click();
  await expect(screen.getByTestId('captions')).toContainText('P(', { timeout: 20_000 });
  await expect(screen.getByTestId('captions')).toContainText('SAM', { timeout: 20_000 });
  await screen.getByRole('button', "Ref's notes").click();  // closed again: the board is back in view

  // someone gets eliminated, and their ring gets an "OUT · ROUND n" stamp
  await expect(screen.getByText(/out · round \d+/i).first()).toBeVisible({ timeout: 60_000 });

  // the game ends with a winner
  await expect(screen.getByTestId('result')).toContainText('Winner:', { timeout: 90_000 });
  await expect(screen.getByTestId('phase')).toContainText('Game over', { ignoreCase: true });

  // playtest feedback: the end-of-game note box, and "Wrong call" on a player in Ref's notes
  await expect(screen.getByTestId('endnote')).toBeVisible();
  await screen.getByLabel('Anything Simon got wrong?').fill('The squat call on Ama was harsh');
  await screen.getByRole('button', 'Send note').click();
  await expect(screen.getByTestId('note-status')).toContainText('Thanks', { ignoreCase: true });
  await screen.getByRole('button', "Ref's notes").click();
  await expect(screen.getByTestId('calls')).toContainText('Last call', { ignoreCase: true });
  await screen.getByTestId('wrong-call').first().click();
  await expect(screen.getByTestId('calls')).toContainText('Undo', { ignoreCase: true });
});
