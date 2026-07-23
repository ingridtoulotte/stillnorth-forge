# Batch-judge benchmark — one-by-one vs N-images-per-call Ollama coherency

**Date:** 2026-07-22 · **Branch:** `spike/dive-motion` · **Model:** `huihui_ai/Qwen3.6-abliterated:35b` (the project's configured judge)
**Question:** does sending 5-50 images in ONE Ollama call, instead of one call per image, save real time without losing real accuracy?

## A real bug surfaced first

The first two attempts at this benchmark returned garbage: every solo call AND every batch call failed with `HTTP 400 exceed_context_size_error` — **Ollama's default served context (4096 tokens) doesn't even fit ONE image + the production `IMAGE_PROMPT`** (measured: 4223 tokens needed). `judge.judge_image_ex`'s `except Exception` handler was silently catching this and returning `(True, "judge unavailable… — passed unjudged", …)` — i.e. **every image-stage coherency judge call in the project has likely been silently no-op'ing**, with no visible error anywhere. This affects the main pipeline's FLUX-still judging, not just the dive lane's `coherency_ok()`.

**Fixed**: `stillnorth/judge.py` — `_chat()` now passes an explicit `num_ctx`, sized by image count (`_judge_num_ctx`, floored at the proven-working 6144 for one image). Verified live through the real `judge.judge_image_ex()` call path: real verdict, real per-image motion/obstacle text, no more fail-open. **This fix stands on its own regardless of the batching question below.**

## Method (with the fix in place)

24 real classified stills (6 each of class A/B/C/D). **Solo baseline** = the project's own `judge.IMAGE_PROMPT` + `judge._parse_verdict`, one Ollama call per still (this IS production behavior, now working). **Batched** = a custom prompt asking for N distinct `<index>: <NONE|MINOR|IMPOSSIBLE>` lines in one call, `num_ctx` scaled per batch size. Model kept warm (`keep_alive=10m`) for the whole run so no batch size was penalized by a cold reload.

## Results

| Mode | Total (24 stills) | Speedup | Agreement w/ solo | Disagreement direction |
|---|--:|--:|--:|---|
| Solo (1/call) | 169.1s (7.05s/img) | 1.0× | — (ground truth) | 24/24 real verdicts, all NONE |
| Batch 5 (ctx 16384) | 94.4s | 1.79× | 79% (19/24) | 5 false-rejects |
| Batch 10 (ctx 24576) | 80.7s | 2.10× | 88% (21/24) | 3 false-rejects |
| Batch 12 (ctx 28672) | 80.7s | 2.10× | 92% (22/24) | 2 false-rejects |
| Batch 20 | — | — | — | **HTTP 413 Payload Too Large** — hard wall unrelated to context size, hit regardless of num_ctx |

**Every disagreement, at every batch size, was the same direction**: a still solo correctly judged NONE got wrongly flagged IMPOSSIBLE when crowded with others in one call — attention dilution, exactly the theoretical risk. Zero false-passes were observed, but this sample happened to be 100% clean per the solo baseline, so the reverse failure mode (a genuinely bad still slipping through under batching) is **untested, not ruled out** — flagging honestly rather than overclaiming.

Batch 20 hits a hard HTTP-layer request-size limit (413), independent of `num_ctx` — batching much past ~12-15 images per call isn't viable on this server without separately raising that limit.

## Why this isn't wired into the per-shot dive/loop path

The per-shot dive pipeline is dominated by upscale (~20-50s) and DepthFlow render (~15s); solo coherency is only ~7s of that. Batching saves ~3.6s/shot — for a typical 6-10 shot loop, that's ~20-35s out of several minutes, marginal. The 8-12% extra-false-reject rate isn't worth it there: a false reject just means the loop-build script has to work with the per-shot solo `coherency_ok()` gate, well-tested and unchanged in this branch (see `docs/spikes/2026-07-22-dive-motion-spike.md`).

Where batching **does** pay off is exactly the "bulk-screen the 828-still backlog before curating loops" scenario: 828 × 3.6s ≈ **50 minutes saved**, real money on this machine's time.

## What shipped

- **`judge._chat` num_ctx fix** — real bug fix, benefits the whole pipeline's image-stage judging, not gated behind anything.
- **`judge.judge_images_batch(cfg, paths)`** / **`judge.judge_stills_prefilter(cfg, paths, batch_size=12)`** — new, tested, standalone bulk pre-screen functions. Fails open on any Ollama/network error (never silently drops a still).
- **`python -m stillnorth judge-stills --stills all --batch-size 12`** — new CLI command exposing the bulk pre-screen. `--batch-size 1` (default) = exact solo judging, matches production. `--batch-size 10-12` = the measured best speed/accuracy point if bulk-screening a large pool.
- **Not changed**: `--motion dive`'s per-shot coherency gate (`dive.coherency_ok`) — still solo, still the tested default from the prior spike.

## Recommendation

Use `--batch-size 1` (solo, exact) for anything where a wrongly-skipped still matters (small curated loops — the material is scarce per-loop). Use `--batch-size 10-12` only for bulk pre-screening a large pool where a false reject just means "the operator picks a different still from an abundant pool" — e.g. before building many loops from the 828-still backlog. Never go above ~15 (hard 413 wall observed at 20).
