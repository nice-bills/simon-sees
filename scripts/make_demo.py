"""Turn the e2e demo recording into demo/simon-sees-demo.mp4 (+ .gif) with Simon's voice and an honesty caption.

    cd e2e && npm run demo            # records .e2e/artifacts/**/video*.webm and .e2e/demo-says.json
    python scripts/make_demo.py       # from the repo root

1. Trims the model-loading wait off the front (keeps 3 s before Line up is tapped).
2. Speeds the picture up uniformly if the game ran past --max-seconds (the voice is NOT sped up; each
   line is placed at its rescaled time).
3. Synthesises every line Simon said with Kokoro (open TTS) and mixes them in at the times the phone
   received them. Falls back to a silent track if Kokoro isn't installed.
4. Burns in a caption saying what this is: a replay of sample frames, not the outdoor field test.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FONT = next((f for f in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                         "/System/Library/Fonts/Supplemental/Arial Bold.ttf"] if os.path.exists(f)), None)


def run(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"command failed: {' '.join(cmd)}\n{r.stderr[-2000:]}")
    return r.stdout


def duration(path: str) -> float:
    return float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path]).strip())


def short(m: str | None) -> str:
    m = (m or "?").replace("FALLBACK ", "").replace("-pretrain-lvd1689m", "")
    return m.split("/")[-1]


def caption_from_log(path: str) -> str:
    """'REPLAY DEMO: AI-generated sample photos. SAM 3 (560 px) + DINOv2 FALLBACK + d1-omni-600M on CPU. ...'
    built from the models the server actually logged, so the caption can't drift from what ran."""
    models = {}
    try:
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            if row.get("models"):
                models = row["models"]
    except OSError:
        pass
    seg, emb, dec = models.get("segmenter", "?"), models.get("embedder", "?"), models.get("decider", "?")
    seg_s = ("SAM 3" + (f" ({seg.split('@')[1].replace('px', ' px')})" if "@" in seg else "")) if "sam3" in seg and "3.1" not in seg \
        else ("SAM 3.1" if "sam3.1" in seg else f"{short(seg)} FALLBACK detector")
    emb_s = f"{'DINOv2 FALLBACK' if 'dinov2' in emb else 'DINOv3' if 'dinov3' in emb else short(emb)}"
    if "Tinker" in dec:  # the fine-tuned referee runs on Tinker's servers, the vision half on this CPU
        return (f"REPLAY DEMO: AI-generated sample photos. {seg_s} + {emb_s} on CPU, referee Qwen3.5-4B + LoRA "
                "on Tinker. Not the field test.")
    return (f"REPLAY DEMO: AI-generated sample photos, real models on CPU: {seg_s} + {emb_s} + {short(dec)}. "
            "Not the outdoor field test.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="recorded webm (default: newest under e2e/.e2e/artifacts)")
    ap.add_argument("--says", default=str(ROOT / "e2e" / ".e2e" / "demo-says.json"))
    ap.add_argument("--out", default=str(ROOT / "demo"))
    ap.add_argument("--max-seconds", type=float, default=58.0)
    ap.add_argument("--caption", default=None,
                    help="honesty caption; default names the models that actually ran (read from the demo log)")
    ap.add_argument("--log", default=str(ROOT / "e2e" / ".e2e" / "demo-decisions.jsonl"))
    ap.add_argument("--lead", type=float, default=3.0, help="seconds kept before Line up is tapped")
    ap.add_argument("--crop-top", type=int, default=0,
                    help="px to cut off the top of the recording (e.g. 100 if a browser notification bar got recorded; "
                         "the demo test opens the page with ?mute=1 so Firefox's speech-dispatcher bar should not appear)")
    a = ap.parse_args()

    video = a.video or max(glob.glob(str(ROOT / "e2e" / ".e2e" / "artifacts" / "**" / "*.webm"), recursive=True),
                           key=os.path.getmtime)
    meta = json.loads(Path(a.says).read_text())
    t0 = meta["videoStart"] / 1000.0
    start = max(0.0, meta.get("lineupClick", meta["startClick"]) / 1000.0 - t0 - a.lead)
    total = duration(video)
    keep = total - start
    speed = max(1.0, keep / a.max_seconds)
    if a.caption is None:
        a.caption = caption_from_log(a.log) + (f" Sped up x{speed:.1f}." if speed > 1.05 else "")
    print("caption:", a.caption)
    out_len = keep / speed
    print(f"video {video}: {total:.1f}s, trimming {start:.1f}s, speed x{speed:.2f} -> {out_len:.1f}s")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp())

    # --- voice
    clips: list[tuple[float, str]] = []
    try:
        from simon_sees.voice import Voice

        voice = Voice("kokoro")
        if not voice.available:
            raise RuntimeError("kokoro not installed")
        for i, s in enumerate(meta["says"]):
            at = (s["t"] / 1000.0 - t0 - start) / speed
            if at < 0 or at > out_len:
                continue
            wav = tmp / f"say{i:02d}.wav"
            wav.write_bytes(voice.synth(s["text"]))
            clips.append((at, str(wav)))
    except Exception as e:  # noqa: BLE001
        print(f"no voice track ({e}); writing a silent one")
    # never let two lines talk over each other: push a line back until the previous one has finished
    placed, free_at = [], 0.0
    for at, wav in clips:
        at = max(at, free_at)
        placed.append((at, wav))
        free_at = at + duration(wav) + 0.1

    # --- picture: trim, speed up, caption
    vf = [f"trim=start={start:.3f}", "setpts=PTS-STARTPTS", f"setpts=PTS/{speed:.4f}", "fps=25",
          *([f"crop=iw:ih-{a.crop_top}:0:{a.crop_top}"] if a.crop_top else []),
          "scale=390:-2:flags=lanczos", "pad=390:ceil(ih/2)*2"]
    if FONT:
        text = a.caption.replace(":", r"\:").replace("'", "")
        # the caption gets its own strip under the phone screen instead of covering the page's bottom edge
        vf.append("pad=iw:ih+82:0:0:color=#121212")
        words, lines, cur = text.split(), [], ""
        for w in words:  # wrap to ~44 characters for a 390 px wide frame
            if len(cur) + len(w) + 1 > 44:
                lines.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}".strip()
        lines.append(cur)
        for k, line in enumerate(lines[:4]):
            vf.append(f"drawtext=fontfile={FONT}:text='{line}':fontcolor=#ffd400:fontsize=14:x=(w-tw)/2:y=h-74+{k * 18}")
    picture = tmp / "picture.mp4"
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-vf", ",".join(vf), "-an", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "23", "-t", f"{out_len:.3f}", str(picture)])

    mp4 = out / "simon-sees-demo.mp4"
    if placed:
        inputs, filters = [], []
        for k, (at, wav) in enumerate(placed):
            inputs += ["-i", wav]
            ms = int(at * 1000)
            filters.append(f"[{k + 1}:a]adelay={ms}|{ms},aresample=44100[a{k}]")
        mix = "".join(f"[a{k}]" for k in range(len(placed)))
        filters.append(f"{mix}amix=inputs={len(placed)}:normalize=0,apad[aout]")
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(picture), *inputs, "-filter_complex", ";".join(filters),
             "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest", str(mp4)])
    else:
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(picture), "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
             "-c:v", "copy", "-c:a", "aac", "-shortest", str(mp4)])

    gif = out / "simon-sees-demo.gif"
    pal = tmp / "pal.png"
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vf", "fps=8,scale=300:-1:flags=lanczos,palettegen=max_colors=96",
         str(pal)])
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-i", str(pal), "-lavfi",
         "fps=8,scale=300:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4", str(gif)])
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {mp4} ({duration(str(mp4)):.1f}s, {mp4.stat().st_size // 1024} KB) and {gif} ({gif.stat().st_size // 1024} KB)")
    print(f"voice lines mixed: {len(placed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
