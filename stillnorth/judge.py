"""Local-AI quality judge (Ollama vision model).

Two checks, both MERCILESS by instruction — any hesitation is a reject:

* judge_image(cfg, path)  — one FLUX still. Looks for semantic incoherence
  (warped/melted structures, duplicated landforms, impossible geometry,
  mangled objects, text/watermarks, obvious AI artifacts).
* judge_video(cfg, path)  — one finished master. Samples 3 frames spanning
  ~1s at a RANDOM point of the clip and asks whether the scene stays
  coherent frame-to-frame (objects/shadows must not jump, warp, appear or
  disappear).

Design notes:
* Local VLMs cannot see pixel-level texture defects (the vision encoder
  downsizes the frame), so the judge targets composition-scale coherence
  only; pixel quality is owned by the render/SR recipe and CV metrics.
* Frames are downscaled to <=1024px before base64 — the encoder would do it
  anyway, this just saves bandwidth and VRAM.
* keep_alive is configurable (default 0) so the judge model releases VRAM
  for FLUX/Wan as soon as a judging batch finishes.
"""
import base64
import json
import os
import random
import subprocess
import tempfile
import urllib.request

# Calibrated on 16 real FLUX stills + synthetic defects (2026-07-09):
# leading "defect checklist" prompts made every local VLM parrot the list and
# reject 100% of clean images. The working shape is a NEUTRAL plausibility
# question with a severity gate — NONE / MINOR / IMPOSSIBLE — where only
# IMPOSSIBLE rejects, and natural repetition is explicitly ruled out
# (uniform ripples/mist were the dominant false-positive source).
IMAGE_PROMPT = (
    "Look at this nature photograph and answer in exactly three lines.\n"
    "Line 1 — from WHERE was the camera when this was taken? Reply exactly "
    "one of:\n"
    "VANTAGE: AIR — camera clearly airborne (drone/helicopter), high above "
    "the ground, looking down or out over a wide landscape\n"
    "VANTAGE: GROUND — camera at or near ground level: a close-up of "
    "flowers/moss/water, a subject a few meters away, an eye-level view\n"
    "Line 2 — classify the image content:\n"
    "- If it is plausible as a real photograph, reply exactly: NONE\n"
    "- If something looks odd but could occur in a real photo, reply: "
    "MINOR: <what>\n"
    "- ONLY if something could absolutely not exist in a real photograph, "
    "reply: IMPOSSIBLE: <what and where>\n"
    "IMPORTANT: repetitive or uniform NATURAL patterns (ripples, waves, "
    "grass, sand, mist, tree rows, sediment lines) occur in real nature "
    "photos all the time and are NEVER impossible. IMPOSSIBLE is reserved "
    "for things a photo editor would call broken: duplicated copy-pasted "
    "rectangles, melted or smeared objects, geometry fused together, "
    "floating disconnected fragments, text overlays, half-formed animals "
    "or structures, and distinct vertical COLUMNS of steam or smoke rising "
    "from the ground like geysers or chimneys where nothing could produce "
    "them (horizontal fog banks and valley mist are fine and normal).\n"
    "Line 3 — imagine this is the first frame of a slow aerial drone shot and "
    "the drone could glide FORWARD (deeper into the scene), LEFT, RIGHT or UP. "
    "Name only the direction(s) that would immediately push it into a large "
    "solid mass right at that edge of the frame — a cliff wall, rock face, "
    "ridge or dense wall of trees filling the foreground or one whole side. "
    "If the view is open in every direction, reply exactly: MOVE-BLOCK: NONE. "
    "Otherwise reply: MOVE-BLOCK: <FORWARD/LEFT/RIGHT/UP, comma-separated>.\n"
    "Three lines only."
)

VIDEO_PROMPT = (
    "These are 3 frames taken about half a second apart from ONE continuous "
    "real drone shot over Nordic nature. The camera moves, so the framing "
    "shifts slightly between frames — that is normal. Compare the frames "
    "and classify the sequence:\n"
    "- If it is plausible as continuous real footage, reply exactly: NONE\n"
    "- If something looks odd but could still be real (lighting shift, "
    "motion blur, parallax), reply: MINOR: <what>\n"
    "- ONLY if the content breaks between frames in a way real footage "
    "never could, reply: IMPOSSIBLE: <what and where>\n"
    "IMPOSSIBLE is reserved for: objects or shadows that teleport or jump "
    "to a different place, things that appear/disappear/duplicate between "
    "frames, landforms that change shape, an outright scene change, large "
    "soft cloudy blobs materialising out of nowhere or dissolving away, or "
    "frames so blurry, smeared or low-resolution that no professional 4K "
    "drone camera could have produced them (thin natural mist that stays "
    "put is fine). One line only."
)


def _b64_image(path, max_side=1024):
    """Read an image, downscale to max_side, return base64 PNG."""
    try:
        import cv2
        img = cv2.imread(path)
        if img is not None:
            h, w = img.shape[:2]
            m = max(h, w)
            if m > max_side:
                s = max_side / m
                img = cv2.resize(img, (int(w * s), int(h * s)),
                                 interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".png", img)
            if ok:
                return base64.b64encode(buf.tobytes()).decode()
    except Exception:
        pass
    with open(path, "rb") as fh:               # fallback: send as-is
        return base64.b64encode(fh.read()).decode()


def _chat(cfg, prompt, images):
    """One Ollama /api/chat round. Returns the raw text reply ('' on none).
    Freeform (no format=json): the severity protocol parses by prefix, and
    forced-JSON mode degraded both tested VLMs into boilerplate verdicts."""
    body = json.dumps({
        "model": cfg.judge_model,
        "messages": [{"role": "user", "content": prompt, "images": images}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0},
        "keep_alive": cfg.judge_keep_alive,
    }).encode()
    req = urllib.request.Request(cfg.ollama_url + "/api/chat", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=cfg.judge_timeout) as r:
        return (json.load(r).get("message", {}).get("content") or "").strip()


def _parse_verdict(text):
    """Severity protocol: only an IMPOSSIBLE verdict rejects. An empty reply
    counts as REJECT (the judge did not vouch for the frame); NONE/MINOR and
    anything unclassifiable-but-worded pass — false rejects were measured to
    be the dominant failure mode of local VLMs, not false passes."""
    t = (text or "").strip()
    if not t:
        return False, "empty judge reply"
    first = t.splitlines()[0].strip().upper()
    if first.startswith("IMPOSSIBLE") or "\nIMPOSSIBLE:" in t.upper():
        return False, t.replace("\n", " ")[:160]
    # Composition off-brief: the vantage-first protocol asks the VLM to
    # commit to AIR vs GROUND before judging content — a GROUND verdict
    # (ground-level close-up / macro) rejects so the source regenerates
    # instead of shipping a "drone" master that was never an aerial shot.
    # (Calibrated 2026-07-10: catches the ground-level tulip-field case;
    # asking the same thing as a NOTDRONE severity option was ignored.)
    up = t.upper()
    if first.startswith("NOTDRONE") or "\nNOTDRONE:" in up:
        return False, t.replace("\n", " ")[:160]
    if "VANTAGE: GROUND" in up or first.startswith("VANTAGE: GROUND"):
        return False, ("off-brief: " + t.replace("\n", " "))[:160]
    return True, t.replace("\n", " ")[:160]


# camera-glide directions the WanCameraEmbedding vocabulary can move
_MOVE_DIRS = ("FORWARD", "LEFT", "RIGHT", "UP")


def _parse_obstacles(text):
    """Directions the drone should NOT glide because a large near mass fills
    that side (the VLM's Line-3 MOVE-BLOCK answer). Subset of _MOVE_DIRS.
    Best-effort: no MOVE-BLOCK line, or 'NONE', returns [] (camera picks
    freely) — a missing/garbled obstacle answer must never block a render."""
    for line in (text or "").splitlines():
        u = line.strip().upper()
        if u.startswith("MOVE-BLOCK:") or u.startswith("MOVE BLOCK:"):
            rest = u.split(":", 1)[1]
            if "NONE" in rest:
                return []
            return [d for d in _MOVE_DIRS if d in rest]
    return []


def available(cfg):
    """True if Ollama answers and the judge model is installed."""
    try:
        with urllib.request.urlopen(cfg.ollama_url + "/api/tags", timeout=5) as r:
            models = [m.get("name", "") for m in json.load(r).get("models", [])]
        want = cfg.judge_model
        return any(m == want or m.split(":")[0] == want.split(":")[0]
                   for m in models)
    except Exception:
        return False


def image_risk_metrics(path, scale_w=768):
    """Deterministic animate-risk metrics for one FLUX still (no VLM).

    fog_cover — fraction of the frame that is bright, desaturated and
    texture-free (a dense fog/cloud bank). Wan cannot hold a big soft fog
    mass in place over a long clip: it animates it, smearing everything the
    fog passes over. Measured on real production sources: the clip whose
    master came back with fog-smear covered ~half the frame; clean sources
    measure near zero.
    sharp — global Laplacian variance; catches an outright blurry FLUX
    render before 14 minutes of GPU time get spent animating it.
    struct_ratio — coarse-structure gradient energy over median fine-tile
    energy. A frame that is a uniform field of thousands of near-identical
    micro-elements (flower meadow, tulip field) gives Wan no stable
    structure to track: it temporally averages the texture into a soft
    "240p" mush from the very first frame. Calibrated on the labeled
    production library (2026-07-10): mushed sources read 0.47/0.67, every
    clean source reads >= 1.3 — a 2x margin."""
    import cv2
    import numpy as np
    orig = cv2.imread(path)
    if orig is None:
        return {"fog_cover": 0.0, "sharp": 1e9, "struct_ratio": 1e9}
    h, w = orig.shape[:2]
    img = orig
    if w > scale_w:
        img = cv2.resize(orig, (scale_w, int(h * scale_w / w)),
                         interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(g, cv2.CV_64F)
    tex = cv2.blur(np.abs(lap), (15, 15))
    fog = (g > 165) & (hsv[..., 1] < 45) & (tex < 2.0)

    # struct_ratio needs the fine texture the 768px resize destroys —
    # measure at the Wan render scale (1280 wide), where it was calibrated.
    g12 = cv2.cvtColor(cv2.resize(orig, (1280, int(h * 1280 / w)),
                                  interpolation=cv2.INTER_AREA),
                       cv2.COLOR_BGR2GRAY)
    gf = g12.astype(np.float32)
    hf = gf - cv2.GaussianBlur(gf, (0, 0), 1.5)
    E = hf * hf
    hh, ww = E.shape
    th, tw = max(1, hh // 24), max(1, ww // 40)
    tiles = [E[y * th:(y + 1) * th, x * tw:(x + 1) * tw].mean()
             for y in range(24) for x in range(40)]
    med_fine = float(np.median(tiles))
    coarse = cv2.GaussianBlur(gf, (0, 0), 8)
    gx = cv2.Sobel(coarse, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(coarse, cv2.CV_32F, 0, 1)
    struct = float((gx * gx + gy * gy).mean())
    ratio = struct / max(med_fine, 1e-6) if med_fine > 25 else 1e9
    # (low fine energy = no dense micro-texture at all -> not this failure
    # mode; the sharp/fog gates own those cases)
    return {"fog_cover": float(fog.mean()), "sharp": float(lap.var()),
            "struct_ratio": ratio}


# Version of the deterministic CV image gate (cv_gate below). BUMP this whenever
# the gate's checks or thresholds change (a new metric, a retuned threshold). The
# pipeline stamps each accepted still with the version that approved it and
# re-gates any still stamped older on resume — so a source accepted before a gate
# existed can never grandfather itself past the gate added later. History:
#   v1 = fog_cover + sharp (pre-2026-07-10)
#   v2 = + struct_ratio micro-texture gate, sharp threshold raised 60->110
CV_GATE_VERSION = 2


def cv_gate(cfg, path):
    """Deterministic CV animate-risk gate for one FLUX still (fog / blur /
    uniform-micro-texture). No VLM — cheap and repeatable, so it is safe to
    re-run on already-accepted stills. Returns (ok, reason); best-effort: any
    error reading/metricing the image passes (True, "") rather than blocking
    the batch. This is gate 1 of judge_image AND the pipeline's stale-still
    re-gate (Pipeline._regate_stale_stills)."""
    try:
        rm = image_risk_metrics(path)
        if rm["fog_cover"] > cfg.judge_fog_cover_max:
            return False, (f"animate-risk: fog/cloud bank covers "
                           f"{rm['fog_cover'] * 100:.0f}% of frame "
                           f"(max {cfg.judge_fog_cover_max * 100:.0f}%) — "
                           "Wan smears large soft fog masses")
        if rm["sharp"] < cfg.judge_image_min_sharp:
            return False, (f"animate-risk: image too soft "
                           f"(sharpness {rm['sharp']:.0f} < "
                           f"{cfg.judge_image_min_sharp:.0f})")
        if rm["struct_ratio"] < cfg.judge_min_struct_ratio:
            return False, (f"animate-risk: uniform micro-texture, no coarse "
                           f"structure (ratio {rm['struct_ratio']:.2f} < "
                           f"{cfg.judge_min_struct_ratio:.2f}) — Wan "
                           "temporally averages dense identical elements "
                           "into mush")
    except Exception:
        return True, ""                          # CV gate is best-effort
    return True, ""


def judge_image(cfg, path):
    """(ok, reason) for one FLUX still. Gate 1 is the deterministic CV
    animate-risk check (cv_gate) — this is where bad videos are prevented, at
    the cheap image stage, instead of detected after 14 minutes of render.
    Gate 2 is the VLM plausibility check. Errors talking to Ollama do NOT
    reject work (the batch must survive a stopped Ollama): they pass with a
    logged reason instead."""
    ok, reason, _ = judge_image_ex(cfg, path)
    return ok, reason


def judge_image_ex(cfg, path):
    """Like judge_image but ALSO returns the VLM's move-block directions —
    which way a near foreground mass rules out gliding (Line-3 MOVE-BLOCK).
    Same single VLM call, additive parse: the camera stage reads these to
    avoid panning/dollying straight into a wall. Returns (ok, reason,
    blocked_dirs). CV-gate failure or an Ollama error returns []."""
    ok, reason = cv_gate(cfg, path)
    if not ok:
        return False, reason, []
    try:
        text = _chat(cfg, IMAGE_PROMPT, [_b64_image(path)])
        ok, reason = _parse_verdict(text)
        return ok, reason, _parse_obstacles(text)
    except Exception as e:
        return (True, f"judge unavailable ({e.__class__.__name__}) — passed "
                "unjudged", [])


def _duration(cfg, path):
    try:
        out = subprocess.run(
            [cfg.ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=60).stdout.strip()
        return float(out.splitlines()[0])
    except Exception:
        return 0.0


def sample_frames(cfg, path, n=3, spacing=0.5, rng=None):
    """Extract n frames starting at a random point, `spacing` seconds apart.
    Returns list of temp PNG paths (caller need not clean up — tempdir)."""
    dur = _duration(cfg, path)
    span = spacing * (n - 1)
    lo, hi = 0.5, max(0.6, dur - span - 0.5)
    t0 = (rng or random).uniform(lo, hi) if hi > lo else lo
    d = tempfile.mkdtemp(prefix="snf_judge_")
    out = []
    for i in range(n):
        p = os.path.join(d, f"f{i}.png")
        r = subprocess.run(
            [cfg.ffmpeg, "-y", "-loglevel", "error", "-ss", f"{t0 + i * spacing:.3f}",
             "-i", path, "-frames:v", "1", p],
            capture_output=True, timeout=120)
        if r.returncode == 0 and os.path.exists(p):
            out.append(p)
    return out


def flicker_metrics(path, scale_w=960, max_frames=200):
    """Temporal-coherence CV metrics (higher = worse): mean |Δ| in the
    frame regions optical flow calls static ('shimmer' — objects/shadows
    jittering frame to frame) and the relative instability of a fixed
    center box's Laplacian variance over time ('tex_instab'). This is what
    actually catches per-frame jumping — a VLM shown sampled stills cannot
    see it (verified on a known-jumpy master: VLM passed it, this didn't)."""
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(path)
    prev = None
    shimmer_vals, lapvars = [], []
    n = 0
    while n < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        h, w = fr.shape[:2]
        if w > scale_w:
            fr = cv2.resize(fr, (scale_w, int(h * scale_w / w)))
        g8 = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        g = g8.astype(np.float32)
        hh, ww = g8.shape
        lapvars.append(cv2.Laplacian(
            g8[hh // 4:3 * hh // 4, ww // 4:3 * ww // 4], cv2.CV_64F).var())
        if prev is not None:
            fl = cv2.calcOpticalFlowFarneback(prev, g, None,
                                              0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.sqrt(fl[..., 0] ** 2 + fl[..., 1] ** 2)
            static = mag < 0.35
            if static.sum() > 500:
                shimmer_vals.append(float(np.abs(g - prev)[static].mean()))
        prev = g
        n += 1
    cap.release()
    lv = np.array(lapvars) if lapvars else np.array([0.0])
    return {
        "shimmer": float(np.mean(shimmer_vals)) if shimmer_vals else 0.0,
        "tex_instab": float(lv.std() / max(lv.mean(), 1e-3)),
        "frames": n,
    }


def judge_video(cfg, path, rng=None, spacing=0.5):
    """(ok, reason) for one finished master. Two gates:
    1. CV temporal coherence — shimmer AND texture instability must BOTH
       exceed their thresholds to reject (single-signal spikes happen on
       fast pans over fine texture; real per-frame jumping trips both).
       Thresholds calibrated on user-verdicted masters: goods measured
       <=3.01/<=0.10, jumping-shadows bad 4.17/0.21, SR-wire bad 3.79/0.22.
    2. VLM semantic check — 3 frames at a random point, content coherence.
       SKIPPED when judge.video_check is "cv": quality steering belongs at
       the image stage (judge_image risk gates), where a reject costs
       seconds, not a re-render of the whole 14-minute chain. The cheap CV
       gate stays as the deterministic safety net on masters.
    Ollama being down skips only gate 2 (CV gate always runs)."""
    try:
        fm = flicker_metrics(path)
        if (fm["shimmer"] > cfg.judge_shimmer_max and
                fm["tex_instab"] > cfg.judge_instab_max):
            return False, (f"temporal flicker: shimmer {fm['shimmer']:.2f} "
                           f"+ instability {fm['tex_instab']:.3f} "
                           "(objects/texture jump between frames)")
    except Exception:
        pass                                     # CV gate is best-effort
    if getattr(cfg, "judge_video_mode", "full") == "cv":
        return True, "CV gate clean (VLM video check off — image-stage gating)"
    frames = sample_frames(cfg, path, spacing=spacing, rng=rng)
    if len(frames) < 3:
        return True, "frame sampling failed — passed unjudged"
    try:
        imgs = [_b64_image(f) for f in frames]
        ok, reason = _parse_verdict(_chat(cfg, VIDEO_PROMPT, imgs))
        return ok, reason
    except Exception as e:
        return True, f"judge unavailable ({e.__class__.__name__}) — passed unjudged"
