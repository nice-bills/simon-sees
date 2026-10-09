// e2e (TesterArmy) config for the Simon Sees phone client.
// The runner starts the real Python server in dry-run mode: replayed sample frames (samples/people)
// instead of a camera, and MOCKED vision (vertical-band "segmenter" + pixel-layout "embedder") plus the
// seeded mock decider instead of SAM 3 / DINOv3 / d1, so the game is deterministic and needs no GPU,
// no weights and no camera permission. The real models are exercised by the demo recorder (e2e.demo.config.ts). Two targets: a desktop window and a phone-sized viewport.
import type { E2EConfig } from 'e2e';
import { web } from '@e2e-dev/web';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const python = process.env.SIMON_PYTHON ?? 'python3';
// chromium in CI; the sandbox this was written in is linux-arm64, where Playwright ships no chromium, so it ran firefox
const browser = (process.env.E2E_BROWSER ?? 'chromium') as 'chromium' | 'firefox' | 'webkit';

const app = {
  url: 'http://127.0.0.1:0',
  command: {
    executable: python,
    args: [
      '-m', 'simon_sees', 'serve',
      '--host', '127.0.0.1', '--port', '{port}',
      '--decider', 'mock', '--segmenter', 'mock', '--embedder', 'mock', '--seed', '3', '--rounds', '8',
      '--replay', path.join(repo, 'samples', 'people'),
      '--tts', 'browser', '--no-stt', '--fast', '--speech-scale', '0.15',
      '--log', path.join(repo, 'e2e', '.e2e', 'decisions.jsonl'),
    ],
    // the playtest recorder is on by default: keep the e2e sessions inside .e2e/
    env: { PYTHONPATH: repo, SIMON_LOGLEVEL: 'WARNING', SIMON_SESSIONS: path.join(repo, 'e2e', '.e2e', 'sessions') },
    log: '.e2e/logs/server.log',
    startupTimeout: 60_000,
  },
};

const agentModel = process.env.AI_GATEWAY_API_KEY ? (await import('ai')).gateway('openai/gpt-6-luna-fast') : undefined;

export default {
  ...(agentModel ? { agents: { default: { model: agentModel } } } : {}),
  targets: [
    { name: 'desktop', engine: web({ browser }), app },
    { name: 'phone', engine: web({ browser, viewport: { width: 390, height: 844 } }), app },
  ],
  assertionTimeout: 10_000,
  workers: 1,
} satisfies E2EConfig;
