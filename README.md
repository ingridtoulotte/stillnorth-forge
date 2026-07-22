# ❄ StillNorth Forge

**A local, staged-batch pipeline that turns folders of HTML prompt files into long, loopable Nordic ambient video clips — one drag-and-drop, one Run button.**

```
HTML prompts ─▶ FLUX.2 images ─▶ upscale ×2 ─▶ CLIP classify (A/B/C/D)
   ─▶ Wan 2.2 clip 1 ─▶ native-overlap continuation (clip 1's last 17 frames, camera dropped)
   ─▶ gold join (RGB colour-match + speed-retime + hard cut at the matched frame) ─▶ one seamless ~10 s loop
   ─▶ ESRGAN finish (contrast de-drift ─▶ Real-ESRGAN super-res ─▶ colour-match back to source ─▶ UHD 2160p + crisp)
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
| **A** | aerial forest canopy / snowy plains | slow drone glide, stable weather, canopy detail held sharp |
| **B** | mountain ridges, peaks, clouds | slow glide along the ridge, existing clouds keep their coverage |
| **C** | rivers, lakes, open water | gentle current + ripples, banks and treeline held sharp |
| **D** | glaciers, frozen lakes, arctic ice | slow glide over the ice, cracks/texture held sharp, clear weather |

All four are broad, realistic and loopable so a single prompt fits every image in its class. Edit them freely in **`config/motion_prompts.json`** — they are sent verbatim.

> ⚠️ **Never ask Wan for volumetrics** (drifting fog, falling snow, rolling clouds): the sampler runs at **cfg 1** (4-step Seko distill), so the negative prompt is ignored and the positive prompt is the only steering. Volumetric wording makes Wan hallucinate soft blurry blobs that spawn/dissolve mid-clip and compound over the continuation (measured 2026-07-09: the old "heavy snowfall" / "low fog drifting" prompts were the direct cause of the spawn-despawn blob defect). Describe **camera glide + stable weather + sharp detail held steady** instead.

**Native-overlap continuation (default `continuation_mode: native_overlap`).** The proven recipe — verified seamless across forest, snow, autumn-lake, green-coast, golden-mountain and marsh sources. Instead of seeding clip 2 from a single still (which gives Wan no motion context → surge, colour pop, speed jump, soft edges), the pipeline feeds **clip 1's last `overlap_frames` (17) real frames** into `WanCameraImageToVideo.start_image`. Wan masks those as *known* and generates the rest as a true continuation of their actual motion — **no surge / colour / speed / sharpness seam by construction**, in one generation.

- `continuation_drop_camera` (default on) — drop the camera embedding on the continuation so it pans at the *inherited* rate from the 17 known frames, not camera-on-top. Stops the over-pan that revealed soft, hallucinated edges (progressive edge-blur). This was the single fix that flattened revealed-edge sharpness over time.
- **Gold join** (`finish.gold_join`) — Wan re-renders the seeded frames at the head of the continuation, so the join frame-matches clip 1's last real frame against the continuation head to find the **true cut** (the first genuinely new frame), then **hard-cuts** (no `xfade`). The old `xfade` cross-dissolved clip 1's tail against Wan's *re-rendered* copy of it (~1 frame ahead) → ghosted motion (blur) + a forward jump; the matched hard cut is continuous by construction. Before the cut the continuation is: de-drifted (`clip2_dedrift`), speed-resampled (`continuation_speed_match`), sharpened (`join_sharpen`); the whole join is then edge-cropped (`edge_crop`).
- `clip2_dedrift` (default on) — Wan's i2v continuation darkens / colour-shifts over its length (clip 2 "looks darker, not natural"). A per-channel quadratic trend, measured on the **visible** (post-crop) centre, is pinned to clip 1's tail colour — kills the drift while keeping natural frame-to-frame variation.
- `continuation_speed_match` (default on) — the continuation runs ~0.8–1.2× clip 1's speed; clip 2's frames are **resampled in memory** (not an ffmpeg `setpts`+`fps` filter, which dropped a frame at the boundary and *spiked* the seam) so its body speed matches clip 1.
- `edge_crop` (default 0.04) — Wan hallucinates soft/grainy content at the frame edges a pan reveals; trim that border and scale back to full frame.
- `esrgan_color_match` + `esrgan_saturation_match` (default on) — remacri-4x over-punches contrast/saturation (the *4k-only* "neon"); after super-res the per-channel mean+std **and** HSV saturation are pulled back onto the source clip.
- `overlap_frames: 17` — seed size; the cut frame itself is found by matching, not assumed (empirically Wan only reproduces ~1 seeded frame at the head).
- **Do not chain** more than one continuation: a second continuation built on a continuation compounds Wan's drift → end-of-clip hallucination. One continuation = ~10 s; go longer via the assembler, not by chaining.

**ESRGAN finish (default `final_upscaler: esrgan`, `finish.esrgan_finish`).** Real detail, not the soft lanczos chain:

1. **Per-frame contrast de-drift** (`contrast_flatten`, default on) — Wan drifts toward higher contrast/saturation over a continuation (a "neon" end). The finisher corrects against the **smoothed measured curve** of per-frame luma-std + saturation (not a linear fit — Wan's drift rises then eases, and a straight line left +10 % saturation swing in real masters) so the drift is removed while each frame keeps its natural variation. Target = clip 1's level × `contrast_target_boost` / `saturation_target_boost`.
1b. **Detail hold** (`detail_hold`, default on) — Wan progressively melts high-frequency texture over a clip (~-15 % Laplacian variance at 720p, which super-res amplifies to ~-30 % perceived at 4K — trees turn painterly). Before super-res, late frames get an adaptive unsharp that holds the clip's fine-detail level at its own early-window baseline (`detail_hold_max` caps the gain). Pairs with **`seed_restore`** (default on): the continuation is seeded with a detail-restored copy of clip 1's tail instead of the raw (already melted) frames — Wan carries its seed's detail level through the whole continuation, measured **+60 % sharper continuation** on matched-seed A/B at zero extra render time.
2. **Real-ESRGAN** per-frame super-res via Upscayl (`esrgan_model: high-fidelity-4x` — `ultramix-balanced-4x` fabricates a wire-mesh texture on open meadows and `remacri-4x` fabricates a chain-mail / vein mesh on dense conifer and moss; `high-fidelity-4x` stays clean across every tested content type), then scale to **UHD `final_height` (2160)** + crisp unsharp + light grain + a gentle final grade (`final_grade`, default `eq=contrast=0.93:gamma=1.04:saturation=0.97` — eases the FLUX-inherited punchy tone; `""` = off). The upscaler can't invent detail past the generation res, which is why `wan_width`/`wan_height` default to **1280×720**: 720p base × 4 = exact 3840×2160 and measured **+47–92 % sharper 2160p masters** vs the old 1104×624 (at ~+65 % render time). Drop back to 1104×624 for faster drafts.

Set `final_upscaler: lanczos` to fall back to the cheap ffmpeg chain on a box without the Upscayl binary. `continuation_mode: single_frame` restores the old single-still continuation (`clip2_color_match`, `trim_start_frames`, `clip2_sharpen` apply only in that mode).

> **Why not one long native pass?** This Wan **WanCameraEmbedding** workflow only animates ~81 frames (5 s). Forcing `length: 161` makes the camera move die out after the first 5 s — the clip stalls/reverses at the midpoint. So `native_long` defaults **false**; the overlap continuation is what extends to ~10 s cleanly.

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

## 🔁 Loop publishing (still-image slideshows)

A second, **GPU-free** way to fill the channel (the "loop pivot"). Instead of
rendering a catalog of hundreds of Wan clips, turn a handful of already-rendered
4K stills into one **seamless-looping** ambient video, then multiply it into many
uploads with ambient audio beds and duration tiers — all ffmpeg, no ComfyUI/GPU.
It does **not** touch the Wan/FLUX render core.

```bash
# build one loop from source-still hashes (found in 03_classified/)
python -m stillnorth loop --name boreal_journey \
    --stills A_eb92140e13914713,B_128f86009656c97a,D_f0019bc3fa9086eb \
    --audio wind
```

Outputs land in `12_slideshows/`: the base loop, one file per duration tier
(`_1min` / `_30min` / `_1h`, each with a fresh ambient bed), and a 9:16 Shorts cut.

**How the loop is seamless.** Each still gets a slow Ken Burns pass; consecutive
stills cross-dissolve; then a short **identical-frame wrap clip** (a static hold of
shot 1's first frame) is appended so the sequence's last frame is pixel-identical to
its first. `ffmpeg -stream_loop` then repeats it with an invisible hard cut — no
wrap-dissolve maths across the loop boundary (measured: loop-seam PSNR ≈ 39 dB, well
above the video's own frame-to-frame motion floor ≈ 33 dB).

**Audio is the multiplier.** Beds are procedural (`wind` / `rain` / `drone` /
`still`), generated fresh at each tier's full length (never looped, so no audio
seam). One visual loop × N beds = N videos at ~zero cost.

Knobs live in the `slideshow` block of `config/config.json` (`hold_seconds`,
`xfade_seconds`, `kenburns_zoom`, `height`, `audio_kind`, `tiers`). The
`--no-kenburns` flag renders static stills; `--no-shorts` skips the vertical cut.

### 🌊 `--motion dive` — depth-parallax instead of Ken Burns (optional)

An alternative shot motion that reads as **diving into the scene** — depth-based
parallax (near moves faster than far), not a flat pan/zoom. Evaluated and measured
in [`docs/spikes/2026-07-22-dive-motion-spike.md`](docs/spikes/2026-07-22-dive-motion-spike.md)
(the upscaler pick in [`…-dive-upscaler-shootout.md`](docs/spikes/2026-07-22-dive-upscaler-shootout.md)).

```bash
python -m stillnorth loop --name boreal_dive --motion dive --stills <hashes> --audio wind
```

Per shot: an **Ollama coherency review of the still** (before any GPU is spent) →
**upscale the still once to 4K** with the shootout-winning upscaler (`high-fidelity-4x`;
sharpest without fabricating mesh) → **DepthFlow** dolly-in from the sharp texture →
the pipeline's own `final_grade` + `final_unsharp`. The `1−cos` dolly returns to its
first frame, so each shot self-loops; the slideshow wrap trick still closes the whole
sequence. Measured: near/far flow ratio 1.9–5.6× (vs ~1.1× for Ken Burns), loop seam
≈ 38 dB. Costs ~1 min/shot (coherency + upscale + render), vs Ken Burns's seconds.

**DepthFlow is AGPL** — so it runs in its **own venv** and is invoked as a **CLI
subprocess** (`dive_render.py`), never imported, keeping this repo's license clean.
Rendered output is yours to monetize (the video is not a derivative of the code).
Configure the `dive` block in `config/config.json` (`venv_python`, `upscaler`, `dolly`,
`parallax`, `ssaa`, `judge_coherency`); default motion stays `kenburns`, so nothing
changes unless you pass `--motion dive`.

---

## Requirements

- **Python 3.9+** (3.12 recommended) — the server, queue, parser, ffmpeg orchestration and ComfyUI client are pure stdlib.
- **ComfyUI** running locally with the two API-format workflows in `workflows/` working (FLUX.2 + Wan 2.2 14B LoRA stack). Default address `127.0.0.1:8188`.
- **ffmpeg** (default `C:/ffmpeg/bin/ffmpeg.exe`). NVENC used by default; falls back to libx264 if you set `"nvenc": false`.
- **For the classify stage only:** `pip install open_clip_torch pillow` into the same environment as ComfyUI (torch already lives there). Weights (~1.7 GB) download once.
- **For the ESRGAN finish (default `final_upscaler: esrgan`):** [Upscayl](https://upscayl.org) (its bundled `upscayl-bin` + models) and `numpy`/`opencv-python` for the per-frame contrast de-drift. Point `esrgan_bin` / `esrgan_models_dir` at your install. If the binary is absent the finisher auto-falls back to the lanczos chain (set `final_upscaler: lanczos` to force it). **VideoHelperSuite** (`VHS_LoadVideoPath`) must be installed in ComfyUI for the native-overlap continuation to read clip 1's tail frames.

## Quick start

```powershell
git clone https://github.com/ingridtoulotte/stillnorth-forge
cd stillnorth-forge

# 1. (optional) point config/config.json at your machine — see below
# 2. make sure ComfyUI is running
.\scripts\run.ps1          # opens http://127.0.0.1:8790 in your browser
```

Then: **drag your HTML files onto the page → press Run**. That's it. You can drop more HTML files at any time, even mid-batch — they flow in from the FLUX stage on the next pass.

CLI extras:

```powershell
python -m stillnorth                 # launch the UI (same as run.ps1)
python -m stillnorth --no-browser    # headless
python -m stillnorth nodes workflows\image_flux2_text_to_image.json   # find node ids
```

## 🤖 AI judge (local Ollama)

Every rendered artefact is inspected by a local vision model (default
`huihui_ai/Qwen3.6-abliterated:35b` via Ollama) — no cloud, no cost:

- **FLUX stills** — the heavy gate. Gate 1 is deterministic CV
  **animate-risk**: a bright desaturated texture-free fog/cloud mass
  covering more than `fog_cover_max` (0.28) of the frame rejects (Wan
  animates big soft fog banks and smears whatever they pass over), and an
  outright blurry render below `image_min_sharp` rejects. Gate 2 is the
  VLM incoherence check (duplicated patches, melted regions, broken
  geometry, floating fragments, text). A reject **deletes the image** and
  the next pass regenerates it with a fresh seed (up to
  `max_image_retries`, then the last render is accepted so one stubborn
  prompt can't stall the batch). Quality steering lives HERE, where a
  reject costs seconds — not after 14 minutes of video render.
- **Finished masters** — by default (`video_check: "cv"`) only the cheap
  CV temporal-coherence gate runs (frame-to-frame shimmer + texture
  instability, catches "objects jumping" flicker a VLM can't see); the
  slow VLM master check is off. Set `video_check: "full"` to add the
  3-frame VLM content-coherence pass, or `false` to disable master
  checks entirely. A reject moves the master to **`10_review/`** (kept
  for your eyes, visible as a gallery tab) and deletes the whole chain
  including the FLUX still, so a completely fresh chain regenerates (up
  to `max_video_retries`, then the key is abandoned).

Ollama being down never blocks the batch — items pass unjudged and it is
noted in the log. Tune it in the `judge` block of `config/config.json`
(`enabled`, `video_check`, `model`, retries, `shimmer_max`/`instab_max`,
`fog_cover_max`, `image_min_sharp`).

## 🎯 Run modes

- **vids wanted (N)** — **N _more_** accepted masters, not N total. The run
  snapshots how many masters already sit in the catalog when you press Run,
  then keeps regenerating (replacing judge-rejected images and remaking
  reviewed vids) until **N new** ones are added on top of that baseline, then
  stops — the status line reads `X / N new · M in catalog`. Only the needed
  number of chains is in flight at once, most-advanced first.
- **time budget (minutes)** — produce as many accepted masters as possible
  in the window, then stop (the item being rendered is finished, not
  killed). Chains run in small waves so clips actually complete instead of
  the whole prompt set camping in the FLUX stage.
- **neither** — classic behaviour: process every queued prompt.

## 📥 Prompt-folder auto-load

Drop your prompt files into **`<workspace>/00_html_prompts/`** — the worker
scans the folder on every pass (new or edited files, tracked by mtime) and
ingests them automatically. It reads the FLUX.2 generator **`.html`** exports
and also plain **`.txt`** / **`.json`** files holding one or more balanced
`{"scene": "..."}` objects, so you can hand-write prompts without the
generator. No UI upload needed; the drag-drop zone still works too.

## The UI

- **Drag-and-drop zone** — one or many `.html` files; add more anytime.
- **Run-mode inputs** — `🎯 vids wanted` / `⏲ time budget (min)` next to Run (leave both empty for the classic full-set run); a status line shows `accepted / target`, time left, and the review count.
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

- **`config/config.json`** — ComfyUI address, ffmpeg path (ffprobe is found beside it), ComfyUI input/output dirs, server host/port, upscale multipliers (`2` / `4` / `4`), fps, codec/quality, the Route-1 upscale filter chain, `native_long` + `native_long_frames` (one-pass clip, the seam fix) and `continuation_seed`, timeouts, and `submit_retries` / `retry_backoff_seconds` (each ComfyUI render is retried with exponential backoff before it's marked failed; failures are categorised — offline / workflow / GPU-timeout — in the log and status). The `assembler` block tunes the long-video feature: `fade_seconds`, `target_height`, `avg_clip_seconds`, `retention_days`, `autodelete_enabled`, and the default usage-bucket `default_weights`.
- **`config/workflows.json`** — the node-id map into each workflow. Re-exported a workflow and the ids changed? Run `python -m stillnorth nodes <file>` and fix them here.
- **`config/motion_prompts.json`** — the 4 motion prompts, the camera poses and per-pose speed.

Outputs are written under `"<ComfyUI output>/StillNorthForge/"` in numbered per-stage folders:

```
00_html_prompts  01_flux  02_flux_up2  03_classified  04_vid1  05_lastframe
06_lastframe_up4  07_vid2  08_concat  09_final_up4  10_review
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
