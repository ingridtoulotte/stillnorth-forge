# StillNorth Forge — Slideshow Lane (design)

**Date:** 2026-07-22 **Branch:** `feature/slideshow-lane`
**Context:** §5E loop-publishing pivot. This adds a second, GPU-free way to fill the
channel: a seamless **still-image slideshow** (Ken Burns + crossfades) built from the
already-rendered still library, plus format-agnostic finishing (audio bed, duration
tiers, Shorts crop). It does **not** touch the render core (FLUX/Wan/gold_join/ESRGAN).

## Goal

Turn N curated 4K stills into one **seamless-looping** ambient video, mux an ambient
audio bed, and export duration tiers (1 min / 30 min / 1 h) + a 9:16 Shorts cut — all
with ffmpeg only, no ComfyUI/GPU. First deliverable ships from the existing 70-master
still library (each master has a matching crisp 2560×1440 source still in `03_classified`).

## Non-goals

- No new image generation for the first ship (850+ stills already on disk).
- No changes to the Wan/FLUX render pipeline.
- Day-cycle grade drift + multi-still "journey ordering UI": deferred (YAGNI until asked).

## Modules

### `stillnorth/slideshow.py` (stdlib + ffmpeg via `media.py`)
- `prep_still(cfg, src_png, dst_png, height)` — lanczos-scale a source still to
  `height` (2160) at 16:9, apply `cfg.final_grade` for cross-still tonal cohesion.
- `kenburns_clip(cfg, img, dst, seconds, zoom, pan, fps)` — render ONE still into a
  slow Ken Burns clip (zoompan, ≤`zoom` total magnification, gentle pan), NVENC.
- `xfade_chain_graph(n, seg_seconds, xfade)` — **pure** function returning the
  `-filter_complex` string + output label for chaining `n` equal-length clips with
  `xfade` dissolves. Offsets = `i*(seg_seconds - xfade)`. Unit-testable without ffmpeg.
- `build_slideshow(cfg, images, dst, hold, xfade, zoom)` — orchestrates: prep each
  still → kenburns_clip each → append an **identical-frame wrap clip** (a static clip
  of image[0] at its Ken Burns start-zoom, so the final frame is pixel-identical to
  frame 0) → xfade-chain all → base seamless loop. The identical end/start frame makes
  `-stream_loop` (tier export) seamless by construction.

### `stillnorth/publish.py` (format-agnostic finishing — reusable by any base loop)
- `make_audio_bed(cfg, kind, seconds, dst)` — procedural ambient via ffmpeg
  `anoisesrc` (pink) + band shaping + slow tremolo swell. Presets: `wind`/`rain`/`drone`.
- `mux_audio(cfg, video, audio, dst)` — mux (`-c:v copy`, aac, `-shortest`).
- `export_tier(cfg, loop, dst, seconds, audio_kind)` — `-stream_loop` the base loop to
  `seconds`, regenerate a fresh audio bed at the tier's full length (no audio seam), mux.
- `export_tiers(cfg, loop, out_dir, tiers, audio_kind)` — batch the above over a tier map.
- `shorts_crop(cfg, src, dst)` — center 9:16 (`crop=ih*9/16:ih` → scale 1080×1920).

### `media.py` additions (low-level, reused)
- Reuse existing `_codec_args`, `run_cancellable`, `probe_duration`. Add only thin
  helpers if a call doesn't fit an existing one; keep orchestration in the two modules.

## Config (`config.json` + `config.py`)
New optional `slideshow` block (all defaulted, back-compat):
`hold_seconds` (7.0), `xfade_seconds` (2.5), `kenburns_zoom` (1.08), `height` (2160),
`audio_kind` ("wind"), `tiers` ({"1min":60,"30min":1800,"1h":3600}).
Loaded in `Config.__init__` following the existing `assembler` block pattern.

## Server / UI (`server.py` + `web/`)
- `GET  /api/slideshow/stills` → the still library (hash, class, thumb path) for picking.
- `POST /api/slideshow/build` → {hashes, hold, xfade, zoom, audio_kind, tiers} → runs a
  build in a worker thread; progress via existing status/log surface.
- New **Slideshow** tab in the SPA: multi-select stills, set knobs, Build, preview result.
- Output lands in a new `12_slideshows/` workspace folder (kept out of the master library
  so the assembler/library bucketing is untouched).

## Testing
- `tests/test_slideshow.py` — pure `xfade_chain_graph` math (offsets, labels, total
  duration), prep/grade filter-string construction, wrap-clip inclusion. stdlib-only.
- `tests/test_publish.py` — tier-count math, audio-bed filter construction, shorts crop
  geometry (9:16 from 16:9). stdlib-only.
- ffmpeg-dependent end-to-end render checks guarded like `test_finish`/`test_flow_roi`
  (skip when ffmpeg/cv2 absent) so the stdlib CI suite stays green.

## First-ship flow (Priority 0)
10-still boreal arc (4×A → 2×B → 2×C → 2×D incl. `D_f0019bc3fa9086eb`), skipping the 3
flagged clips. Recipe validated by rendering 2–3 variants at 1080p, screenshot-compare,
pick, then render the chosen recipe at 4K. Wind bed. Export 1 min / 30 min / 1 h. User
watches, swaps any stills, approves.

## Risks / mitigations
- **zoompan jitter** → pre-scale still up before zoompan; cap zoom rate.
- **loop seam visible** → identical-frame wrap guarantees a clean `-stream_loop` cut;
  escalate to a split-dissolve wrap only if a viewer flags the settle-on-img0 as a pause.
- **NVENC concurrent-session cap** → render recipe variants at 1080p and/or sequentially.
- **audio seam on tiers** → audio regenerated per tier length, never looped.
