# The fine-tuned referee (Qwen3.5-4B + LoRA, trained on Tinker)

The referee answers one yes/no question per player crop ("Look at the person in this photo. Is that person
standing on one foot? ... Answer with one word: yes or no.") and the game reads P(yes) from the logprobs of the
first answer token. Results, training details and what failed: `bench/results-tinker.md`. Data pipeline and
licences: `data/README.md`.

## Play with it on Tinker (what was measured)

```bash
pip install tinker
export TINKER_API_KEY=...                       # your Tinker key
export SIMON_TINKER_MODEL=tinker://<run>/sampler_weights/<step>   # the checkpoint in bench/results-tinker.md
python -m simon_sees serve                      # --decider auto: Tinker -> local weights -> d1
```

`--decider auto` (the default) uses Tinker when `SIMON_TINKER_MODEL` and a key are set, else local weights when
`SIMON_REFEREE_WEIGHTS` is set, else `LiquidAI/d1-omni-600M`/`d1-3B` on your own machine. Every fallback is logged.
Freeze rounds are judged by frame differencing either way (the referee sees one crop at a time).

A Tinker checkpoint is private to the account that trained it. To use the referee without that account,
use the local weights below.

## Run it without Tinker: local weights

```bash
# 1. download the LoRA (Tinker archive: adapter weights + adapter_config.json, ~290 MB)
python scripts/export_referee.py download tinker://<run>/sampler_weights/<step> --out models/referee-lora/tinker
# 2. convert to standard PEFT format (needs tinker-cookbook + the base model, Qwen/Qwen3.5-4B, ~9 GB once)
pip install git+https://github.com/thinking-machines-lab/tinker-cookbook peft
python scripts/export_referee.py peft models/referee-lora/tinker --out models/referee-lora/peft
# 3a. serve it in the game (transformers + peft, merged in memory at start-up)
SIMON_REFEREE_WEIGHTS=models/referee-lora/peft python -m simon_sees serve --decider local-qwen
# 3b. or merge into a full checkpoint once (bf16, ~9 GB) and point SIMON_REFEREE_WEIGHTS at it
python scripts/export_referee.py merge models/referee-lora/tinker --out models/referee-merged
SIMON_REFEREE_WEIGHTS=models/referee-merged python -m simon_sees serve --decider local-qwen
# 3c. or serve the adapter with vLLM (OpenAI-compatible): vllm serve Qwen/Qwen3.5-4B --enable-lora \
#     --lora-modules referee=models/referee-lora/peft
```

`models/` is git-ignored. Hardware: Qwen3.5-4B needs ~9 GB in bf16 (a GPU with 12 GB+, or an Apple-silicon
Mac with 16 GB+, `--device mps`); in fp32 on CPU it needs ~17 GB of RAM.

**Not tested here.** The machine this was built on has 8 GB of RAM and no GPU, so steps 2-3 (conversion,
merge, local serving) were not run. Step 1 (the download) was. The local path uses the same prompt and the
same P(yes) read-out as the Tinker path, but its accuracy has not been measured; treat it as untested until you
run `bench/referee_eval.py` against it.
