# Upscaler shootout — dive-lane 4K path

**Date:** 2026-07-22 · **Branch:** `spike/dive-motion` · **Machine:** RTX 5070 Ti
**Question:** which upscaler takes a 2560×1440 classified still → 4K best, to feed
DepthFlow a sharp texture (dive renders at 4K from the upscaled still — one upscale
per shot, not per frame). Winner is baked as `dive.upscaler` default.

## Method
3 content-diverse stills (B tundra/grass — the mesh-fabrication-prone case; C misty
marsh; D hero glacier gravel/ice) × 8 methods → 3840×2160. Per output: Laplacian
variance on a fixed native-res 1024 crop over textured ground (`crop_lap` = real
detail *and* where fabrication shows), the project's own `struct_ratio`, and an
**Ollama coherency review** (the configured `Qwen3.6:35b` VLM) fed the **native crop**
(not the downsampled 4K frame — the VLM downsamples ~1k and would miss pixel-scale
mesh), asked only about upscaler artifacts (mesh/veins/plastic/halos): NONE/MINOR/SEVERE.

## Results (mean crop-sharpness + coherency across the 3 stills)

| Method | mean crop_lap | VLM coherency | Visual read (1:1 crops) |
|---|--:|---|---|
| **high-fidelity-4x** ✅ | 603 | all MINOR | **sharp + natural across grass/water/gravel; no visible mesh** |
| ultrasharp-4x | 1008 | all MINOR | sharpest, but **hexagonal "chicken-wire" mesh** on organic grass |
| upscayl-standard-4x | 1258 | all MINOR | very sharp, but **repetitive pebble "tiling"** on stream/grass |
| ultramix-balanced-4x | 381 | all MINOR | fine **pixel grid** + slight plastic smoothing |
| upscayl-lite-4x | 138 | all MINOR | "painted"/plastic clumping |
| digital-art-4x | 71 | all MINOR | plastic smoothing + faint banding |
| remacri-4x | 44 | all MINOR | soft (low detail), no mesh |
| lanczos | 32 | all MINOR | softest (honest, no fabrication) |

## Findings
1. **Winner: `high-fidelity-4x`.** Sharpest of the *artifact-free* models — meaningfully
   crisper than lanczos/remacri without the chicken-wire mesh (ultrasharp,
   upscayl-standard) or plastic grid (ultramix) the others fabricate on organic
   content. Confirms the core pipeline's own hard-won SR pick, now validated for the
   dive lane. Paired with `final_unsharp`, the integrated dive measured mid-clip
   sharpness 669 (B) / 847 (D) — sharper than the source still — with the parallax
   *strengthened* (B near/far flow ratio 3.77→5.60: sharper texture = more trackable
   flow), loop seam ≈ 38 dB.
2. **High numeric sharpness ≈ fabricated mesh here.** The top-`crop_lap` models win the
   number by inventing texture, which is exactly what the dive lane's charter forbids
   (pixels displaced, not regenerated). `crop_lap` alone is a trap; the 1:1 visual +
   the VLM's artifact notes are the real discriminators.
3. **The 35B VLM did not cleanly gate** — it flagged *every* method MINOR (the
   over-strict "parrot" behaviour already documented for this model in the project's
   seam-war history). Its *value* was the notes (it correctly named ultrasharp's
   chicken-wire and upscayl-standard's tiling), not a pass/fail. So the dive lane uses
   it as the project does elsewhere: a **lenient** coherency gate (reject only gross /
   IMPOSSIBLE), not a fabrication detector.

## SeedVR2 / UltimateSDUpscale (diffusion) — not run, by design
Both are **diffusion** upscalers → they hallucinate detail, which the shootout shows is
already the failure mode of the sharpest ESRGAN models — a diffusion model would
fabricate *more*, directly violating the dive lane's preserve-pixels charter. SeedVR2
is also not installed (only `comfyui_ultimatesdupscale` is present), and installing its
CUDA deps into the shared ComfyUI env risks the production Wan pipeline. **Recommendation:
skip for this lane.** If a diffusion-upscale comparison is still wanted, it should be a
*sandboxed* follow-up (separate env, its own GPU test), reported with the
hallucination-vs-fidelity tradeoff explicit — not wired into the loop lane.

## Decision
`config.json` `dive.upscaler = "high-fidelity-4x"` (default). `"lanczos"` remains a
zero-fabrication fallback (used automatically if the Upscayl bin/model is missing).
