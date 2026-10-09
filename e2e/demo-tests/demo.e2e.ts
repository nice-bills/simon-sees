// The demo recording: line up (SAM finds everyone, each photo becomes a card), name two of the three
// players, play a replayed game with the real models, and keep the spoken lines' timestamps so
// scripts/make_demo.py can lay Simon's voice over the video.
import { test } from '@e2e-dev/web';
import { expect } from 'e2e';
import { writeFileSync } from 'node:fs';

test('demo: replay game on a phone viewport', async ({ app, screen, browser }) => {
  const videoStart = Date.now();
  // mute=1: the phone stays silent (no speech-dispatcher bar in headless Firefox); make_demo.py adds the voice
  await app.open('/?captions=1&mute=1');
  await expect(screen.getByTestId('status')).toContainText('ready', { ignoreCase: true, timeout: 400_000 });
  const lineupClick = Date.now();
  await screen.getByRole('button', 'Line up').click();
  await expect(screen.getByTestId('lineup-player')).toHaveCount(3, { timeout: 300_000 });
  // the cards sit under the sticky Start bar on a phone: scroll the team sheet into view so the video shows them
  await browser.evaluate(() => document.getElementById('player-list')?.scrollIntoView({ block: 'center' }));
  await new Promise((r) => setTimeout(r, 2500));  // let the photo cards be seen
  await screen.getByLabel('Name for player 1').fill('Kofi');
  await screen.getByLabel('Name for player 2').fill('Ama');
  await new Promise((r) => setTimeout(r, 2000));
  const startClick = Date.now();
  await screen.getByRole('button', 'Start').click();
  // once d1 has judged a round, open Ref's notes for a few seconds so the video shows the real decisions
  // (models, P(did it) per player, SAM/DINO/d1 timings); the drawer is closed by default
  // (polled from here: a long-running browser.evaluate times out)
  while (!(await browser.evaluate(() => !!document.querySelector('#captions table')))) await new Promise((r) => setTimeout(r, 500));
  await screen.getByRole('button', "Ref's notes").click();
  await new Promise((r) => setTimeout(r, 9000));
  await screen.getByRole('button', "Ref's notes").click();
  await expect(screen.getByTestId('result')).toContainText('Winner', { ignoreCase: true, timeout: 1_000_000 });
  await new Promise((r) => setTimeout(r, 2500));  // hold the final frame
  const says = await browser.evaluate(() => (window as unknown as { __says: unknown[] }).__says);
  writeFileSync('.e2e/demo-says.json', JSON.stringify({ videoStart, lineupClick, startClick, end: Date.now(), says }, null, 2));
});
