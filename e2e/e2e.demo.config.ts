// Records the README demo: the phone client at 390x844, a replayed game over samples/people with the REAL
// models: SAM 3 (facebook/sam3, text prompt "person"; 560 px on CPU) -> DINOv3, or the DINOv2 FALLBACK while
// the DINOv3 gate is pending -> the referee: LiquidAI/d1-omni-600M by default (CPU so it runs anywhere;
// SIMON_DEMO_MODEL=LiquidAI/d1-3B on a GPU/Mac), or SIMON_DEMO_DECIDER=tinker for the fine-tuned Qwen3.5-4B
// referee on Tinker (docs/tinker.md). SIMON_DEMO_SEGMENTER=fallback uses the ungated RT-DETR person detector instead of SAM 3.
// Run: npm run demo  -> video in .e2e/artifacts/**/video, spoken-line timings in .e2e/demo-says.json
// Then: python ../scripts/make_demo.py  (adds Simon's voice with Kokoro + an honesty caption, writes demo/)
import type { E2EConfig } from 'e2e';
import { web } from '@e2e-dev/web';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const python = process.env.SIMON_PYTHON ?? 'python3';
const browser = (process.env.E2E_BROWSER ?? 'chromium') as 'chromium' | 'firefox' | 'webkit';

export default {
  tests: 'demo-tests/**/*.e2e.ts',
  targets: [{
    name: 'phone',
    engine: web({ browser, viewport: { width: 390, height: 844 }, screencast: { size: { width: 390, height: 844 } } }),
    app: {
      url: 'http://127.0.0.1:0',
      command: {
        executable: python,
        args: [
          '-m', 'simon_sees', 'serve', '--host', '127.0.0.1', '--port', '{port}',
          '--decider', process.env.SIMON_DEMO_DECIDER ?? 'd1', '--model', process.env.SIMON_DEMO_MODEL ?? 'LiquidAI/d1-omni-600M',
          '--segmenter', process.env.SIMON_DEMO_SEGMENTER ?? 'sam3', '--embedder', process.env.SIMON_DEMO_EMBEDDER ?? 'auto',
          '--device', process.env.SIMON_DEVICE ?? 'cpu',
          '--replay', path.join(repo, 'samples', 'people'),
          '--seed', process.env.SIMON_DEMO_SEED ?? '4', '--rounds', '4', '--pose-frames', '1',
          '--tts', 'browser', '--no-stt', '--fast', '--speech-scale', '1.0',
          '--log', path.join(repo, 'e2e', '.e2e', 'demo-decisions.jsonl'),
        ],
        env: { PYTHONPATH: repo, SIMON_LOGLEVEL: 'WARNING', PYTHONFAULTHANDLER: '1', PYTHONUNBUFFERED: '1', OMP_NUM_THREADS: process.env.OMP_NUM_THREADS ?? '2', ...(process.env.HF_HOME ? { HF_HOME: process.env.HF_HOME } : {}),
               ...(process.env.HF_HUB_OFFLINE ? { HF_HUB_OFFLINE: process.env.HF_HUB_OFFLINE } : {}),
               // gated weights (facebook/sam3): the server needs the token even when the runner filters the env
               ...(process.env.HF_TOKEN ? { HF_TOKEN: process.env.HF_TOKEN } : {}),
               // SIMON_DEMO_DECIDER=tinker: the fine-tuned referee on Tinker (SIMON_TINKER_MODEL + TINKER_API_KEY).
               // The runner drops the proxy vars (the browser must reach 127.0.0.1 directly); a sandbox that only
               // reaches Tinker through a proxy passes it as SIMON_DEMO_HTTPS_PROXY and it goes to the server alone.
               ...Object.fromEntries(['SIMON_TINKER_MODEL', 'TINKER_API_KEY', 'SIMON_TINKER_API_KEY', 'SIMON_TINKER_TOKENIZER',
                                      'SSL_CERT_FILE', 'REQUESTS_CA_BUNDLE']
                 .filter((k) => process.env[k]).map((k) => [k, process.env[k] as string])),
               ...(process.env.SIMON_DEMO_HTTPS_PROXY ? { HTTPS_PROXY: process.env.SIMON_DEMO_HTTPS_PROXY,
                    HTTP_PROXY: process.env.SIMON_DEMO_HTTPS_PROXY, NO_PROXY: '127.0.0.1,localhost' } : {}) },
        log: '.e2e/logs/demo-server.log',
        startupTimeout: 60_000,
      },
    },
  }],
  video: 'on',
  timeout: 1_200_000,
  assertionTimeout: 10_000,
  workers: 1,
} satisfies E2EConfig;
