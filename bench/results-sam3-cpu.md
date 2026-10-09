# Real SAM 3 + DINOv2 on the sample frames (CPU, 2026-10-08)

Run with `bench/sam3_cpu_reid.py` in the build sandbox: linux aarch64, 2 vCPU, 8 GB RAM, no GPU,
torch 2.14.1+cpu, transformers 5.17.0, fp32, 2 threads. Raw output: `results-sam3-cpu.json`.

- **Segmenter:** `facebook/sam3` through transformers (`Sam3Model`, text prompt "person"), input resized to
  **560 px** (the CPU default; SAM 3 is trained at 1008 px). Masks on every detection.
- **Embedder:** DINOv3 was tried first and refused (gated repo, HTTP 403 for this account), so the pipeline fell
  back automatically to **`facebook/dinov2-small`** (labelled FALLBACK). No DINOv3 numbers exist yet.
- **Frames:** the 7 photos in `samples/people/` (960x717, fitted to 640 px long side). They are generated park
  photos of three people, not a field recording.
- Model load (SAM 3 + DINOv2, weights already cached): 14.1 s. The weights loaded straight from the network
  filesystem; no SIGBUS this time, so the copy-to-/tmp fallback was not needed.

## Person counts and SAM 3 latency

| frame | people found | SAM 3 time | scores (left to right) |
|---|---|---|---|
| 000_lineup | 3 | 18.5 s (first call) | 0.98, 0.974, 0.98 |
| 001_lineup | 3 | 17.3 s | 0.98, 0.974, 0.98 |
| 002_onefoot_hands-head_stand | 3 | 16.8 s | 0.982, 0.974, 0.979 |
| 003_arms-up_stand_arms-up | 3 | 16.9 s | 0.977, 0.972, 0.979 |
| 004_swapped-places_stand | 3 | 16.3 s | 0.981, 0.979, 0.98 |
| 005_airplane_squat_airplane | 3 | 16.4 s | 0.954, 0.962, 0.951 |
| 006_centre-player-left-frame | 2 | 16.2 s | 0.979, 0.982 |

Median SAM 3 time per frame after the first call: **16.6 s**. Every frame got the
right head count (3, and 2 in the frame where the centre player has left). DINOv2-small embedding of 3 crops:
about 0.3 s. Matching: about 2 ms.

## Re-identification (enrol on 000, then every later frame in order, tracking state carried over)

Players are numbered left to right at the line-up. For labelling only: P1 navy sweater, P2 floral dress,
P3 checked shirt (the pipeline never sees these as words).

| frame | truth, left to right | matched, left to right | | cosine to the player's gallery |
|---|---|---|---|---|
| 001_lineup | P1 P2 P3 | P1 P2 P3 | ✓ | P1 1.00, P2 1.00, P3 1.00 |
| 002_onefoot_hands-head_stand | P1 P2 P3 | P1 P2 P3 | ✓ | P1 0.85, P2 0.71, P3 0.99 |
| 003_arms-up_stand_arms-up | P1 P2 P3 | P1 P2 P3 | ✓ | P1 0.71, P2 0.96, P3 0.73 |
| 004_swapped-places_stand | P3 P1 P2 | P3 P1 P2 | ✓ | P1 0.96, P2 0.94, P3 0.94 |
| 005_airplane_squat_airplane | P1 P2 P3 | P1 P3 | ✗ | P1 0.66, P2 0.30, P3 0.65 |
| 006_centre-player-left-frame | P1 P3 | P1 P3 | ✓ | P1 0.94, P2 0.32, P3 0.92 |

- **004_swapped-places: correct.** All three people changed places and each was matched to the right identity
  with cosine 0.94-0.96; in a direct line-up-only check on the same frame the runner-up cosines were at most 0.60.
- **006: correct.** P2 left the frame and was reported "not seen" (best cosine 0.32), never confused with anyone.
- **005: one miss.** P2 squatting (straight after the swap, so tracking had her last box on the far right) only
  reached cosine 0.30 against her standing line-up photo, below the 0.45 threshold, so she was reported
  **not seen** that frame. The game never eliminates an unseen player (it warns them), so this costs a
  "get back in the picture" line, not a wrong elimination. P1 and P3 in airplane pose were matched (0.66, 0.65).

Overall: 5 of 6 later frames fully correct, 0 identity swaps, 1 missed sighting. This is 7 generated frames of 3
people, not a benchmark; outdoor numbers will come from a field test.
