// Optional agent steps (natural-language act/assert). They need a model: the config wires up Vercel AI
// Gateway when AI_GATEWAY_API_KEY is set. Without a key this test is skipped at collection (requesting the
// agent fixture with no model would be a configuration error) and the deterministic suite still runs.
import { test } from '@e2e-dev/web';
import { expect } from 'e2e';

const withModel = process.env.AI_GATEWAY_API_KEY ? test : test.skip;

withModel('an agent lines everyone up, names the first player and starts a game', async ({ app, agent, screen }) => {
  await app.open('/');
  await agent.act('tap Line up, wait for the name chips to appear, name the first player {name}, then start the game', {
    params: { name: 'Kofi' },
  });
  await agent.assert('a game is in progress and every player is marked over the picture with a ring and a name chip');
  await expect(screen.getByTestId('player-card')).toHaveCount(3);
});
