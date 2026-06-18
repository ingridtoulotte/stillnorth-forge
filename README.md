# ❄ StillNorth Forge

**A local, staged-batch pipeline that turns folders of HTML prompt files into long, loopable Nordic ambient video clips — one drag-and-drop, one Run button.**

```
HTML prompts ─▶ FLUX.2 images ─▶ upscale ×2 ─▶ CLIP classify (A/B/C/D)
   ─▶ Wan 2.2 clip 1 ─▶ last frame ─▶ upscale ×4 ─▶ Wan 2.2 clip 2 (continuation)
   ─▶ concat into one ~10–11 s loop ─▶ final upscale ×4
   ─▶ 🎞 Long-video assembler ─▶ randomised, faded 15 min … 12 h ambient compilations
```

Built for **THE STILL NORTH** (a faceless Nordic ambient YouTube production line) and wired around an existing ComfyUI install, FLUX.2 and Wan 2.2 14B. Pure standard library for everything except the optional classify stage. Windows-first, single GPU.

---

## Why "staged batch"?

The whole point. The pipeline **never** runs one prompt all the way down the chain before starting the next. Instead every item clears a stage before the next stage begins:

> all prompts → **all** FLUX images → **all** ×2 upscales → **all** classified → **all** first clips → **all** last frames → **all** ×4 upscales → **all** continuation clips → **all** concats → **all** final upscales

That means FLUX loads once for the whole image batch and Wan 2.2 loads once per video batch, instead of thrashing both models in and out of VRAM on every single prompt. On a 16 GB card that is the difference between hours and days.

### Rough throughput

On a single RTX 5070 Ti the **two Wan 2.2 clips dominate** the per-prompt cost (~2–5 min each); everything else is seconds. End-to-end one prompt — FLUX image → ×2 → classify → clip 1 → last frame → ×4 → clip 2 → concat → final ×4 — lands around **≈ 7–12 minutes**, almost all of it the two video renders. The exact number is GPU-, resolution- and LoRA-dependent; the UI now shows a live **avg/item** and **stage ETA** so you get your real number after the first few items.

## The 4 motion classes

A zero-shot **CLIP** classifier (mirrored from the project's `tsn_classify.py`, same model and class prompts) sorts each FLUX image by **content**. Each content class is then animated with the Wan 2.2 motion prompt that physically fits it:

| Class | Image content | Motion sent to Wan 2.2 |
|---|---|---|
| **A** | aerial forest canopy / snowy plains | low fog + soft cloud shadows + faint falling snow drifting over the canopy |
| **B** | mountain ridges, peaks, clouds | clouds & fog rolling horizontally across the ridge |
| **C** | rivers, lakes, open water | gentle current + ripples + thin mist over glass-smooth water |
| **D** | glaciers, frozen lakes, arctic ice | heavy snowfall cascade over the frozen landscape |

All four are broad, realistic and loopable so a single prompt fits every image in its class. Edit them freely in **`config/motion_prompts.json`** — they are sent verbatim.

**Direction continuity:** clip 1 picks a random camera move (`Pan Up/Left/Right`, `Zoom In/Out`) and records it. The continuation clip starts from clip 1's **upscaled last frame** and is **forced to the same camera move + speed**, so the two halves push the camera the same way and concatenate seamlessly (clip 2's first frame == clip 1's last frame, so it is dropped at the join).

---

## 🎞 Long-video assembler

Once you have a pile of ~11 s masters, the **Long video** panel stitches them into a ready-to-upload ambient compilation — pick a length and press **Build**:

- **Target length:** 15 min · 30 min · 1 · 2 · 3 · 4 · 6 · 8 · 10 · 12 hours.
- **A fade between every clip** (dip to black, default 1 s — set `assembler.fade_seconds`). 1080p by default (`assembler.target_height`).
- **Globally randomised order**, picked at random from the eligible pool — never "the first file in the folder". When a long target needs more slots than you have masters, the pool repeats with no back-to-back duplicates.
- **Usage-aware mixing.** Every master tracks how many times it has been put into a long video. The mixer (◀ ▶ steppers, auto-normalised to 100 %) sets what share of clips is drawn from each reuse level — **Never used / Used 1× / 2× / 3× / 4×+** — defaulting to a fresh-first 60 / 20 / 10 / 6 / 4.

### Usage buckets (durable, inspectable folders)

A never-used master lives in `09_final_up4/`. The first time it lands in a compilation it is **moved** into a sibling folder under the ComfyUI output root, and bumped one level on every later build it joins:

```
<output>/StillNorthForge/09_final_up4/   never used
<output>/11sec_used_1/                    used once
<output>/11sec_used_2/                    used twice
<output>/11sec_used_3/                    used 3×
<output>/11sec_used_4plus/                used 4× or more
```

The folder a clip sits in is the source of truth for its bucket, so you can inspect or reshuffle usage by hand. Exact counts and first-used timestamps are cached in `assembler_state.json`. The pipeline knows a moved master is *finished* work, so it is never re-rendered.

Compilations are written to `<output>/Compilations/North-<length>-<timestamp>.mp4`. To keep a 12 h render cheap, each distinct master is normalised+faded **once** into a cached MPEG-TS segment, then the timeline is stream-**copied** (no multi-hour re-encode) into the final file.

### Crash-safe builds & batches

- **Generation:** pressing **Run** records the intent. If the terminal is closed or the PC shuts down mid-batch, the next launch **auto-resumes** where it stopped (an explicit Cancel/pause does not). State writes are atomic and resume is filesystem-driven.
- **Long-video builds** are resumable too: the plan is persisted, segments live on disk (a resumed build skips finished ones), and the usage moves are recorded per-clip, so a half-finished 12 h render picks up cleanly on relaunch.

### Auto-delete (retention)

Intermediate build artifacts — FLUX images, ×2 upscales, first clips, last frames, ×4 last-frame upscales, continuation clips, concats — are **auto-deleted `assembler.retention_days` (default 7) days after a master is first used in a long video**. The finished ×4 masters (in `09_final_up4` / the used buckets) and the compilations are **never** touched. Runs on launch, after each build, and on the **Clean now** button. Set `assembler.autodelete_enabled: false` to keep it manual-only.

### Preview

The **Last done** card shows only the single most-recent artifact (latest image or short clip) — a minimal "what just finished" glance, not a gallery.

---

## Requirements

- **Python 3.9+** (3.12 recommended) — the server, queue, parser, ffmpeg orchestration and ComfyUI client are pure stdlib.
- **ComfyUI** running locally with the two API-format workflows in `workflows/` working (FLUX.2 + Wan 2.2 14B LoRA stack). Default address `127.0.0.1:8188`.
- **ffmpeg** (default `C:/ffmpeg/bin/ffmpeg.exe`). NVENC used by default; falls back to libx264 if you set `"nvenc": false`.
- **For the classify stage only:** `pip install open_clip_torch pillow` into the same environment as ComfyUI (torch already lives there). Weights (~1.7 GB) download once.

## Quick start

```powershell
git clone https://github.com/ingridtoulotte/stillnorth-forge
cd stillnorth-forge

# 1. (optional) point config/config.json at your machine — see below
# 2. make sure ComfyUI is running
.\scripts\run.ps1          # opens http://127.0.0.1:8787 in your browser
```

Then: **drag your HTML files onto the page → press Run**. That's it. You can drop more HTML files at any time, even mid-batch — they flow in from the FLUX stage on the next pass.

CLI extras:

```powershell
python -m stillnorth                 # launch the UI (same as run.ps1)
python -m stillnorth --no-browser    # headless
python -m stillnorth nodes workflows\image_flux2_text_to_image.json   # find node ids
```

## The UI

- **Drag-and-drop zone** — one or many `.html` files; add more anytime.
- **Run / Resume** (`R`), **Cancel** (`C`; pause, resumable), **Clear queue** (forget pending prompts, **keep** rendered media), **Purge outputs** (destructive — **delete** every rendered image/clip and reset to a clean slate).
- **Left sidebar** — live stats (prompts in set, finished masters, elapsed, avg/item, stage ETA, failures), VRAM gauge, the output-folder path (with copy), and the environment health pills.
- **🎞 Long-video assembler** — duration buttons (15 min → 12 h), a fresh-vs-reused **clip mix** with ◀ ▶ steppers, live usage-bucket counts, build progress, finished-compilation list, and a Clean-now button. See [above](#-long-video-assembler).
- **Last done** preview — the single most-recent artifact, nothing else.
- **Output gallery** — browse every rendered stage in-app (thumbnails for images, click-to-play loop preview for clips), copy any file's absolute path, lightbox view. Served sandboxed to the workspace with HTTP-range seeking.
- **Per-stage metrics** — each pipeline node shows its average render time and failure count.
- **Activity feed** — live colour-coded tail of `forge.log` with search, copy, and pause-autoscroll.
- **Command palette** (`⌘K` / `Ctrl-K`) and shortcuts: `R` run, `C` cancel, `T` theme, `G` gallery.
- **Light / dark theme** toggle (persisted), built on a semantic design-token system.
- **Live GPU VRAM** gauge (via `nvidia-smi`).
- **Progress bar + stage text** that reports exactly where you are: `image batch 42%`, `upscaling the flux images 80%`, `1st vid batch …`, `upscaling the lastframes (1st vid) …`, `2nd vid batch …`, `final upscale …`.
- **Pipeline strip** showing the live output count at every one of the 9 stages, with the active stage highlighted.
- **Queue list** of ingested files and prompt counts.

## Configuration

Everything tunable lives in `config/` — no Python edits needed.

- **`config/config.json`** — ComfyUI address, ffmpeg path (ffprobe is found beside it), ComfyUI input/output dirs, server host/port, upscale multipliers (`2` / `4` / `4`), fps, codec/quality, the Route-1 upscale filter chain, timeouts, and `submit_retries` / `retry_backoff_seconds` (each ComfyUI render is retried with exponential backoff before it's marked failed; failures are categorised — offline / workflow / GPU-timeout — in the log and status). The `assembler` block tunes the long-video feature: `fade_seconds`, `target_height`, `avg_clip_seconds`, `retention_days`, `autodelete_enabled`, and the default usage-bucket `default_weights`.
- **`config/workflows.json`** — the node-id map into each workflow. Re-exported a workflow and the ids changed? Run `python -m stillnorth nodes <file>` and fix them here.
- **`config/motion_prompts.json`** — the 4 motion prompts, the camera poses and per-pose speed.

Outputs are written under `"<ComfyUI output>/StillNorthForge/"` in numbered per-stage folders:

```
01_flux  02_flux_up2  03_classified  04_vid1  05_lastframe
06_lastframe_up4  07_vid2  08_concat  09_final_up4
forge_state.json   forge.log   assembler_state.json   _assembler_cache/
```

Final masters land in **`09_final_up4/`** as `<class>_<key>.mp4`; once used in a long video they migrate to `<output>/11sec_used_N/`, and compilations land in `<output>/Compilations/`.

## Resumability & "add anytime"

State is driven by the filesystem: every stage skips any item whose output already exists, so a crash, a Cancel, or closing the app loses nothing — press Run and it picks up where it stopped. **And if a batch was running when the terminal closed or the PC shut down, the next launch auto-resumes it** (intent is persisted; an explicit Cancel does not auto-resume). `forge_state.json` persists only what the filesystem can't recover: the prompt set, the class letters, the per-clip camera pose map, and the run intent. Dropping new HTML mid-run simply grows the set; the worker re-reads it each pass.

## Architecture

```
web/                 drag-drop UI (vanilla JS, no build step) ── polls /api/status
stillnorth/
  server.py          stdlib HTTP API + static host
  pipeline.py        orchestrator: 9 staged batches, queue, cancel, resume, state
  assembler.py       long-video builder: selection, fades, usage buckets, retention
  library.py         master library + usage-bucket folders/state (stdlib, testable)
  comfy.py           ComfyUI HTTP client (queue / wait / interrupt) + output rename
  html_prompts.py    robust prompt extraction (handles every TSN HTML layout)
  classify.py        zero-shot CLIP A/B/C/D (lazy torch import)
  media.py           ffmpeg upscale / last-frame / concat / fade-segment / VRAM
  config.py          config + path resolution
workflows/           the two ComfyUI API-format workflows
config/              all tunables
tests/               extraction, config, server, assembler tests (no GPU, no deps)
```

The worker is single-threaded by design: a single GPU renders one ComfyUI job at a time, so the pipeline submits one, waits, submits the next — which also makes progress reporting exact.

## Troubleshooting

- **`ComfyUI ✕`** in the header — start ComfyUI; the pipeline retries each pass and the status shows the error.
- **classify stage error about torch/open_clip** — `pip install open_clip_torch pillow` into ComfyUI's Python.
- **wrong node error / nothing renders** — your workflow node ids differ; run `python -m stillnorth nodes <workflow>` and update `config/workflows.json`.
- **no NVENC** — set `"nvenc": false` in `config/config.json` to use libx264.
- **logs** — `<workspace>/forge.log`.

## License

MIT © 2026 Ingrid Toulotte. Part of **THE STILL NORTH** local production stack.
