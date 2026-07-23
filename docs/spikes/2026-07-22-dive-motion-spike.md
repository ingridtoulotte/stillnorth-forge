# Spike report — Dive-motion animation (DepthFlow vs Ken Burns)

**Date:** 2026-07-22 · **Branch:** `spike/dive-motion` (off `e3f011d`) · **Machine:** RTX 5070 Ti, 16 GB
**Status:** contained R&D spike — *evaluated, not adopted.* No production file touched. Not merged. No PR.
**Priority note:** this is **parallel, optional, exploratory** work. Per the project sheet §6, Priority 1 remains **user feedback on the shipped `boreal_journey_v1` loop** — this spike does **not** jump that queue.

---

## TL;DR verdict

| Candidate | Result | One-line reason |
|---|---|---|
| **1. DepthFlow** | ✅ **PASS — winner** | Real depth-parallax dive, ≤12 s/shot, loops seamlessly, monetization-safe, not diffusion. One caveat: raw output ~10–20 % softer than Ken Burns — **fully recovered by the pipeline's own `final_unsharp`** (retention → 1.01–1.22). |
| 2. ComfyUI-Depthflow-Nodes | ⏭️ **Not rendered — same engine** | Byte-for-byte the same DepthFlow shader/depth model, wrapped for ComfyUI. Would reproduce candidate 1's numbers. For the *loop lane* it is a step **backward** (recouples the deliberately zero-dep lane to ComfyUI, pulls AGPL into that env). Deliberately **not installed** (§8: don't touch the ComfyUI env). |
| 3. parallax-maker | ❌ **Disqualified + not triggered** | Its standard pipeline **inpaints occlusion gaps with Stable Diffusion XL/SD3** → generative hallucination, which trips §2's hard "pixels displaced, not regenerated" filter. Also a manual per-shot Blender/glTF workflow. Its gating condition ("1 & 2 pass but motion feels generic") never fired — DepthFlow's motion is good. |

**Recommendation:** wire DepthFlow's dolly preset into `slideshow.py` as a `--motion dive` **alternative** to `--motion kenburns`, gated behind a flag, **no change to the existing default**, invoked as a **CLI subprocess** (never `import depthflow` — keeps the repo's own license clean of AGPL). **Do not implement yet** — this report is for review.

---

## 0. Setup / hygiene (what was and wasn't touched)

- **Isolated env:** everything runs from a self-contained venv at `output\StillNorthForge\99_dive_spike\venv` (Python 3.12). The project's Python env, `requirements`, and the ComfyUI/Wan install were **not** modified. `depthflow==1.0.0` pulled a **CPU** torch 2.13 wheel (no CUDA index) — which sidesteps the RTX 5070 Ti Blackwell `sm_120` problem entirely: depth estimation runs on CPU, the animation render is an OpenGL/moderngl GLSL shader that uses the GPU through WGL (confirmed: *"OpenGL Renderer: NVIDIA GeForce RTX 5070 Ti"*).
- **Read-only source:** the three test stills were **copied** out of `03_classified\` into the scratch folder; the originals were never moved or mutated.
- **Outputs isolated:** all renders, depth maps, montages, code and venv live under a new `output\StillNorthForge\99_dive_spike\` scratch folder — nothing written into `12_slideshows\` or any numbered stage folder.
- **D:\ write constraint:** all files built in the C:\ scratchpad first, then copied onto D:\ via PowerShell (the `Write`/`Bash`-redirect route is blocked on D:\ in this environment).
- **rtk:** absent on this machine (checked both shells) — plain python/git used, consistent with the project sheet.
- **Disk footprint (removable):** 1507 MB total (venv 1163 MB + 21 render files 230 MB). Delete `99_dive_spike\` to reclaim.

## 1. Test material (3 stills spanning classes)

| Class | Hash | Scene | Depth character |
|---|---|---|---|
| A | `A_014f6c8b2c9bd6db` | aerial boreal coast/forest | **depth-shallow** (flat aerial, compressed near→far range) |
| B | `B_01db9fb467cb4514` | aerial tundra + lakes, misty | strong near-tussock → far-fog range |
| D | `D_f0019bc3fa9086eb` | **hero glacier** (braided delta) | rich: near river → glacier → far mountains |

## 2. Success criteria (defined numerically *before* testing, per §4)

1. **Render time** ≤ 40 s/shot (this machine). Logged per attempt.
2. **Sharpness retention** ≥ 0.80 — variance-of-Laplacian of output mid-clip frames vs the source still, resolution-normalised (both resized to 1280 px wide before the Laplacian, the project's `struct_ratio` calibration width). *Method mirrors `finish._sharp_metric` / `judge._sharp`.*
3. **Depth differential (the actual dive test)** — mean Farneback optical-flow magnitude (params identical to `flow_roi.roi_flow`: `0.5,3,21,3,5,1.2,0`) in a **hand-picked foreground ROI ÷ background ROI**, same frame pairs, over the mid-clip band. A real dive shows FG ≫ BG; a flat pan/zoom shows FG ≈ BG. **Ken Burns rendered as the control** on the same stills + same ROIs — the honest baseline for "is this a dive, or Ken Burns in disguise?"
4. **Edge integrity** — visual check for tearing/stretching/ghosting at occlusion boundaries across frames (no numeric gate).
5. **License** — must be compatible with a monetized (AdSense) channel; reported explicitly per candidate.
6. **Not diffusion** (hard filter) — output pixels must be the *displaced source pixels*, not regenerated.

Metrics tool: `docs/spikes/dive_motion_metrics.py` (stdlib + opencv/numpy, self-tested — `python dive_motion_metrics.py selftest` → ALL PASS). Full driver + raw clips under `99_dive_spike\`.

---

## 3. Candidate 1 — DepthFlow (full results)

**What it is (verified by reading the installed source, not the docs):** DepthFlow 1.0.0 = DepthAnythingV2 depth map → a ray-marched GLSL parallax shader that *displaces* the source pixels. **Not** a diffusion/generative model — passes §2's hard filter. In 1.0.0 the old `depthflow dolly …` preset subcommands are **gone** and the default `DepthScene.update()` is empty (the bare CLI makes a *static* video); motion is the Python API — subclass `DepthScene`, set `self.state.*` per frame. The "dive" is the docs `Dolly` recipe: `state.dolly = amp·(1−cos(cycle))` moves the camera ray-origins **through depth** (near parallax > far), and `1−cos` returns to 0 at the end → **the clip loops to its own first frame by construction**.

### 3.1 Primary metrics — default settings (dolly 1.0 / parallax 0.6), 3840×2160 @ 16 fps, ~9.5 s

| Still | Render s (cold) | Sharpness retention (raw / **+`final_unsharp`**) | **Dive ratio** (FG/BG) | KB control ratio | Edge integrity | Loop seam PSNR | Verdict |
|---|---|---|---|---|---|---|---|
| A (shallow aerial) | 11.6 | 0.61 / **1.06** | ~1.0 *(weak)* | 0.53–1.36 | clean | 43.6 dB | ⚠️ weak dive (scene-limited) |
| B (tundra) | 11.6 | 0.61 / **1.01** | **3.77** | 1.21 | clean | 43.1 dB | ✅ pass |
| D (hero glacier) | 11.6 | 0.66 / **1.19** | **1.86** | 1.12 | clean | 41.6 dB | ✅ pass |

*Camera-subtracted differential (`ratio_rel`, isolates parallax from the global move): B 3.99, D 2.16 — even stronger.*

### 3.2 Criterion-by-criterion

**① Render time — PASS decisively.** ~11.6 s/shot cold at 4K (CPU depth-est ~7.3 s + GLSL shader/encode ~4.1 s), ~9.1 s cold at native 2560×1440, and DepthFlow **disk-caches depth by image hash** so a re-render of the same still is ~2–4 s. Ken Burns control ~4.5–5.0 s. DepthFlow is ~2.5× slower than Ken Burns but clears the 40 s budget by ~3.5×. (Moving depth-est to GPU — needs a Blackwell `cu128` torch, not installed — would cut it to ~2 s; not needed.)

**② Sharpness — the one real caveat, and it is recoverable.**
- Raw DepthFlow retention is **0.56–0.67** vs the source (below 0.80). *But so is Ken Burns (0.70–0.74).* A **static re-encode of the still through the identical yuv420p/crf16 path scores 0.998** — so the encode is not the floor; both animators genuinely soften via resampling (DepthFlow's depth-warp samples bilinearly every frame; Ken Burns's zoompan resamples too). DepthFlow is the softer of the two: **~89–91 % of Ken Burns mid-clip sharpness, ~79–85 % of its frame-0 sharpness** — a real, measured ~10–20 % softness, the same *category* of finding as Wan's video-from-still softness in the seam wars.
- **Applying the pipeline's own `final_unsharp` (`5:5:0.9:5:5:0.0`) lifts retention to 1.01–1.22 — comfortably past 0.80, on all three stills.** This is the exact finish pass the project already runs on every (soft) Wan clip, for exactly this reason. So sharpness is a **solved, tunable knob**, not a wall. (0.9 slightly over-sharpens for a still source; ~0.5 would look more natural — a future tuning choice, not a blocker.)

**③ Depth differential — PASS on depth-rich scenes; scene-dependent.** B (3.77×) and D (1.86×) show a strong FG≫BG differential far above the Ken Burns control (1.1–1.2×) — genuine motion parallax, a real dive. **A is weak (~1.0×)**: it's a flat aerial with a compressed depth range, so there is little near/far separation for the shader to exploit — and cranking dolly to 2.5 doesn't fix it (A stayed 0.95; *no dolly can manufacture depth that isn't in the map*). Honest takeaway: **DepthFlow's dive strength tracks the scene's actual depth range** — great on foreground-vs-distance shots (rivers/tussock/ridges against far mountains/fog), subtle on flat aerials. Curate accordingly.

**④ Edge integrity — PASS.** At occlusion boundaries (glacier-vs-mountain ridgeline on D, foreground-tundra-vs-fog on B), early-vs-late dolly frames show **no tearing, ghosting, or rubber-sheet stretching** — texture stays coherent. These boreal landscapes have *gradient-like* depth (no hard foreground silhouettes against far backgrounds), so DepthFlow's known weakness (edge stretch at hard occlusions) barely manifests — a favourable content match. See `99_dive_spike\montages\edge_B.png`, `edge_D.png`.

**⑤ License — AGPL-3.0 — PASS for output monetization, with an integration caveat.** AGPL copyleft binds *software distribution*, **not rendered output**: locally rendering videos and uploading them to a monetized channel does **not** trigger any source-disclosure obligation (the video is not a derivative work of the code, and nothing is offered as a network service). ⚠️ The caveat is for any future *code* integration: invoke DepthFlow as a **CLI subprocess** (as the repo already does for ffmpeg/upscayl), **never `import depthflow`** — importing an AGPL library into `stillnorth-forge` could pull AGPL onto the whole (public) repo. Subprocess invocation is aggregation, not linking. This is the same "open-source ≠ automatically commercial-safe" caution the project already applied to MusicGen/Suno audio.

**⑥ Not diffusion — PASS (confirmed by source).** DepthAnythingV2 depth + GLSL displacement shader. The output is the source photo's pixels displaced — no generative model paints new content.

**Bonus — seamless loop for free.** The `1−cos` dolly returns to its first frame; measured **loop-boundary PSNR 41.6–43.6 dB** (the loop lane treats 39 dB as an excellent seam, 33 dB as the everyday-motion floor). So a DepthFlow dive clip is **drop-in loop-lane material with no wrap-frame trick needed** — the wrap is intrinsic to the motion.

### 3.3 Tuning log (5-iteration cap respected — 3 used, clear winner reached)

| # | What was tried | Result / decision |
|---|---|---|
| 1 | dolly 1.0 / parallax 0.6 @ 4K + Ken Burns control | Baseline. Strong dive on B/D; A's FG ROI landed on smooth water → flow under-read (fixed in #2). |
| 2 | native 2560×1440 re-render; A's FG ROI moved to textured shore | Isolated the sharpness confound (native KB f0 only 0.71 → the low absolute retention is resampling, not just upscale). A remains weak even with a fair ROI → scene-limited. |
| 3 | static-encode floor (0.998) · `final_unsharp` recovery · higher dolly 2.5/0.85 · edge crops | Unsharp recovers sharpness to 1.01–1.22 ✅. Higher dolly does **not** help (A 0.95; D ratio 1.73 and loop seam fell to 38 dB) → **dolly 1.0 / parallax 0.6 is the sweet spot.** Edges clean. **Winner confirmed — stopped.** |

**Example clips** (under `99_dive_spike\renders\`): `dive_B_01db9fb467cb4514.mp4` (strongest dive), `dive_D_f0019bc3fa9086eb.mp4` (hero glacier). Contact montages (source · depth · dive@25% · dive@75%): `99_dive_spike\montages\m_*.png`.

---

## 4. Candidate 2 — ComfyUI-Depthflow-Nodes — not rendered (same engine)

AGPL-3.0. The README states it is *"an implementation of Depthflow in ComfyUI"* — it **wraps the same BrokenSource/DepthFlow shader + DepthAnythingV2**, exposing the same zoom/dolly/orbital presets. Output would be **identical to candidate 1** — rendering it separately would only reproduce the numbers above. Installing it requires cloning into `ComfyUI/custom_nodes` and modifying that Python env; per §8 I **did not** touch the ComfyUI env. For the loop-publishing lane specifically it is the *wrong* vehicle: that lane is deliberately zero-dependency / ffmpeg-only and decoupled from the ComfyUI/Wan core (§4.2 of the project sheet) — the ComfyUI wrapper would recouple it and pull AGPL into the render-core env. **Recommend the standalone CLI (candidate 1), not this wrapper**, for the loop lane. (The wrapper stays worth remembering *only* if a future goal is running dive inside a ComfyUI graph alongside Wan — a different use case than this spike.)

## 5. Candidate 3 — parallax-maker — disqualified (diffusion) + gate not triggered

AGPL-3.0. Its core is depth-segmentation into layered cards + glTF export (Blender/Unreal) — but its standard pipeline **inpaints the occlusion gaps revealed at card boundaries using Stable Diffusion XL / SD3**. That is exactly the generative hallucination §2 makes a hard disqualifier ("output pixels should still be recognizably the source photo's pixels, displaced — not new content hallucinated by a generative model" → *"If a candidate approach turns out to secretly be diffusion-based, disqualify it immediately and say so"*). Independently, it is the **manual per-shot Blender/glTF camera-path** candidate, explicitly gated behind "*test this only if 1 and 2 clear the quality bar but the motion still feels generic*" — that condition never fired (DepthFlow's dive is good and tunable). **Not pursued**, for two independent reasons.

---

## 6. Recommendation (do not implement yet — for review)

1. **Adopt DepthFlow as an optional `--motion dive` mode** in `slideshow.py`, a sibling of the existing `--motion kenburns` (which stays the untouched default). Gate behind a flag; render each still into a dive clip instead of a Ken Burns clip, then feed into the *existing* xfade/wrap/loop chain unchanged.
2. **Invoke it as a CLI subprocess** from an isolated venv (like ffmpeg/upscayl) — **never `import depthflow`** — to keep `stillnorth-forge`'s license unaffected by AGPL.
3. **Default knobs:** dolly 1.0 / parallax 0.6, render at **native 2560×1440** (not 4K — let the existing finish/upscale path do the 4K, same as for stills), then the pipeline's existing **`final_unsharp`** to restore sharpness. Loops need no wrap clip (the dolly self-loops).
4. **Curate for depth:** apply dive to depth-rich shots (foreground-vs-distance); keep flat aerials on Ken Burns (dive is weak there).
5. **Cost of adoption:** ~12 s GPU/CPU per shot, zero new model in the render core, one AGPL tool invoked out-of-process.

## 7. Reproduce

```powershell
$py='D:\IA_PROJET\System\ComfyUI\output\StillNorthForge\99_dive_spike\venv\Scripts\python.exe'
$C='D:\IA_PROJET\System\ComfyUI\output\StillNorthForge\99_dive_spike\code'
$env:HF_HOME='D:\IA_PROJET\System\ComfyUI\output\StillNorthForge\99_dive_spike\hfcache'
# render a dive
& $py $C\render_depthflow.py --image <still.png> --out dive.mp4 --time 9.5 `
   --width 2560 --height 1440 --fps 16 --ssaa 2.0 --dolly 1.0 --parallax 0.6
# measure everything
& $py $C\measure_all.py    # 4K dive vs KB: retention + depth-diff + loop seam
& $py $C\measure2.py       # native retention + revised ROIs
& $py $C\measure3.py       # encode floor + unsharp recovery + higher dolly
```

Non-package scratch (drivers, venv, raw clips) lives under `99_dive_spike\`; only this report and `dive_motion_metrics.py` are committed. Delete `99_dive_spike\` to reclaim 1.5 GB.
