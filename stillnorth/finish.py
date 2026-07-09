"""Native-overlap continuation join + ESRGAN/contrast UHD finisher.

This is the proven "The Still North" recipe:
  - GOLD JOIN: Wan re-renders the seeded frames at the head of the continuation.
    Find which output frame reproduces clip 1's last real frame (frame match) ->
    the next one is the first genuinely NEW frame = the true cut. Colour-match the
    continuation to clip 1 (per-channel RGB mean+std), sharpen it to match, retime
    the new frames to clip 1's motion speed (Farneback flow ratio), then HARD-CUT
    concat. No xfade: cross-dissolving clip 1's tail with Wan's re-rendered copy of
    it ghosted ~1s of motion (blur) and snapped (forward jump). The cut is
    continuous by construction, so a straight join is seamless once colour matches.
  - FINISH: per-frame progressive contrast+saturation de-drift (Wan drifts toward
    higher contrast over a continuation -> "neon end"; corrected against the linear
    trend), then Real-ESRGAN (Upscayl) per-frame super-res, then colour-match the
    super-res back onto the source clip (remacri over-punches colour = the 4k-only
    neon), then UHD scale + crisp + grain.

numpy/cv2 are imported lazily so the package still loads on a lanczos-only box.
"""
import os, glob, shutil, subprocess, tempfile


def _np_cv2():
    import numpy as np
    import cv2
    return np, cv2


def _codec(cfg, cq):
    if cfg.nvenc:
        return ["-c:v", "hevc_nvenc", "-preset", "p7", "-cq", str(cq),
                "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "slow", "-crf", str(cq),
            "-pix_fmt", "yuv420p"]


def _run(args):
    return subprocess.run(args, stdout=subprocess.DEVNULL,
                          stderr=subprocess.PIPE).returncode == 0


def _frame_count(path):
    _, cv2 = _np_cv2()
    c = cv2.VideoCapture(path)
    n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
    c.release()
    return n


def _rgb_stats(path, lo, hi):
    np, cv2 = _np_cv2()
    cap = cv2.VideoCapture(path)
    i = 0
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if lo <= i <= hi:
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(float))
        if i > hi:
            break
        i += 1
    cap.release()
    a = np.stack(frames)
    return a.mean(axis=(0, 1, 2)), a.std(axis=(0, 1, 2))


def _flow(path, lo=0, hi=10 ** 9):
    """Mean Farneback optical-flow magnitude over frames [lo, hi] = motion speed.
    Downscaled to 416x240 for speed. Sharpness-independent, unlike a frame diff,
    so it measures real movement not texture."""
    np, cv2 = _np_cv2()
    cap = cv2.VideoCapture(path)
    i = 0
    prev = None
    mags = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if lo <= i <= hi:
            g = cv2.cvtColor(cv2.resize(fr, (416, 240)), cv2.COLOR_BGR2GRAY)
            if prev is not None:
                fl = cv2.calcOpticalFlowFarneback(
                    prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mags.append(float(np.sqrt(fl[..., 0] ** 2 + fl[..., 1] ** 2).mean()))
            prev = g
        if i > hi:
            break
        i += 1
    cap.release()
    return float(sum(mags) / len(mags)) if mags else 0.0


def _last_frame_gray(path, n, size=(128, 72)):
    np, cv2 = _np_cv2()
    cap = cv2.VideoCapture(path)
    last = None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        last = fr
    cap.release()
    return cv2.cvtColor(cv2.resize(last, size), cv2.COLOR_BGR2GRAY).astype(float)


def _match_offset(clip1, raw, n1, search):
    """Find the continuation frame that reproduces clip1's LAST real frame.

    WanCameraImageToVideo masks the seeded frames as KNOWN and re-renders them at
    the head of the output; how many it reproduces (N vs N+3) is opaque, so we
    don't trust a fixed overlap count. Match clip1's final frame against the first
    `search` continuation frames; the frame AFTER the best match is the first
    genuinely NEW frame, i.e. the true cut point. A wrong fixed offset is exactly
    what shows as a forward jump at the seam."""
    np, cv2 = _np_cv2()
    last = _last_frame_gray(clip1, n1)
    cap = cv2.VideoCapture(raw)
    i = 0
    best, bestd = 0, 1e18
    while i <= search:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(fr, (128, 72)), cv2.COLOR_BGR2GRAY).astype(float)
        d = float(((g - last) ** 2).mean())
        if d < bestd:
            bestd, best = d, i
        i += 1
    cap.release()
    return best


def _read_all(path):
    np, cv2 = _np_cv2()
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(fr.astype(float))
    cap.release()
    return out


def _dedrift_clip2(new, ref_bgr, box=None):
    """Remove the continuation's per-channel brightness/colour DRIFT in place.

    Wan's i2v continuation slowly darkens / shifts colour over its length, so a
    single colour anchor at the seam can't hold it -> clip2 ends visibly darker and
    'not natural'. Fit each channel's per-frame-mean trend and shift every frame so
    the TREND sits on clip1's tail colour (ref_bgr), keeping each frame's own
    deviation (natural variation, not a flat exposure clamp).

    `box`=(x,y,w,h) measures the trend on the CENTRE region that survives the edge
    crop (the part the viewer actually sees); the shift is applied to the whole
    frame. Matching the visible region is what removes the 'clip2 looks darker'."""
    np, _ = _np_cv2()
    n = len(new)
    if n < 2:
        return new
    if box:
        x, y, bw, bh = box
        roi = [f[y:y + bh, x:x + bw] for f in new]
    else:
        roi = new
    idx = np.arange(n)
    means = np.array([[r[..., c].mean() for c in range(3)] for r in roi])
    # quadratic, not linear: Wan's darkening accelerates toward the end, so a
    # straight-line trend leaves the END below it (clip2 still ends darker). A
    # degree-2 fit follows the curve and pins the whole clip to clip1's tail.
    deg = 2 if n >= 5 else 1
    for c in range(3):
        trend = np.polyval(np.polyfit(idx, means[:, c], deg), idx)
        for i in range(n):
            new[i][..., c] = np.clip(new[i][..., c] + (ref_bgr[c] - trend[i]), 0, 255)
    return new


def _sharp_metric(frames, box):
    """Mean Laplacian variance over the visible (post-crop) region -- texture/
    grain level, independent of colour. Used to detect a seam grain mismatch."""
    np, cv2 = _np_cv2()
    if not frames:
        return 0.0
    x, y, w, h = box
    vals = []
    for f in frames:
        roi = np.clip(f[y:y + h, x:x + w], 0, 255).astype("uint8")
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        vals.append(cv2.Laplacian(g, cv2.CV_64F).var())
    return float(np.mean(vals))


def _match_seam_sharpness(new, ref_sharp, box, K=8):
    """clip2's raw continuation is consistently grainier than clip1's diffused
    tail at the cut -- a hard cut between the two reads as a 'blur to sharp' /
    contrast jump even though colour is already matched. Blur clip2 down
    toward clip1's texture level (or sharpen it up, if it's the softer one)
    so the cut is a colour-only transition, not a texture-level one too.

    Laplacian variance falls off non-linearly with Gaussian blur sigma (and
    rises non-linearly with unsharp amount), so a fixed formula overshoots
    badly at large mismatches -- binary-search the actual parameter against
    the measured effect instead of guessing it."""
    np, cv2 = _np_cv2()
    sample = new[:min(K, len(new))]
    cur = _sharp_metric(sample, box)
    if cur <= 1e-3 or ref_sharp <= 1e-3:
        return new
    ratio = cur / ref_sharp
    if 0.85 <= ratio <= 1.15:
        return new
    if ratio > 1.15:
        lo, hi = 0.0, 6.0
        for _ in range(10):
            mid = (lo + hi) / 2
            test = [cv2.GaussianBlur(f, (0, 0), mid) for f in sample]
            (lo, hi) = (mid, hi) if _sharp_metric(test, box) > ref_sharp else (lo, mid)
        sigma = hi
        return [cv2.GaussianBlur(f, (0, 0), sigma) for f in new] if sigma > 0.05 else new
    lo, hi = 0.0, 3.0
    for _ in range(10):
        mid = (lo + hi) / 2
        test = [np.clip(f + (f - cv2.GaussianBlur(f, (0, 0), 3)) * mid, 0, 255)
                for f in sample]
        (lo, hi) = (mid, hi) if _sharp_metric(test, box) < ref_sharp else (lo, mid)
    amt = hi
    if amt <= 0.02:
        return new
    return [np.clip(f + (f - cv2.GaussianBlur(f, (0, 0), 3)) * amt, 0, 255) for f in new]


def _struct_metric(frames, box):
    """Structure sharpness, grain-blind: varLap after a sigma-1.5 Gaussian.
    Plain varLap counts grain as sharpness -- a noisy-but-smeared clip1 tail
    measured SHARPER than a clean continuation, inverting the equalize
    decision. Blurring first suppresses grain; real edges survive."""
    _, cv2 = _np_cv2()
    return _sharp_metric([cv2.GaussianBlur(f, (0, 0), 1.5) for f in frames],
                         box)


def _sharpen_to_struct(frames, target, box, K=10):
    """Binary-search ONE unsharp amount lifting `frames` structure to
    `target` (sharpen only -- never blurs)."""
    np, cv2 = _np_cv2()
    cur = _struct_metric(frames[:K], box)
    if cur >= target or cur <= 1e-3:
        return frames
    sample = frames[:K]
    lo, hi = 0.0, 3.0
    for _ in range(10):
        mid = (lo + hi) / 2
        test = [np.clip(f + (f - cv2.GaussianBlur(f, (0, 0), 3)) * mid, 0, 255)
                for f in sample]
        (lo, hi) = (mid, hi) if _struct_metric(test, box) < target else (lo, mid)
    amt = hi
    if amt <= 0.02:
        return frames
    return [np.clip(f + (f - cv2.GaussianBlur(f, (0, 0), 3)) * amt, 0, 255)
            for f in frames]


def _contrast_sat(frames, box):
    """Mean (luma std, HSV saturation mean) over the visible region -- the
    same two numbers Wan's continuation pass consistently overshoots on."""
    np, cv2 = _np_cv2()
    x, y, w, h = box
    cons, sats = [], []
    for f in frames:
        roi = np.clip(f[y:y + h, x:x + w], 0, 255).astype("uint8")
        cons.append(float(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).std()))
        sats.append(float(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[..., 1].mean()))
    return float(np.mean(cons)) if cons else 0.0, float(np.mean(sats)) if sats else 0.0


def _match_seam_contrast(new, ref_con, ref_sat, box, K=8):
    """clip2's continuation pass is consistently 13-22% punchier (contrast
    AND saturation) than clip1's tail, measured across multiple sources --
    not a config setting, the model itself overshoots on the second pass.
    Contrast/saturation scale linearly (unlike blur), so a direct ratio
    works without a search."""
    np, cv2 = _np_cv2()
    sample = new[:min(K, len(new))]
    cur_con, cur_sat = _contrast_sat(sample, box)
    k_con = float(np.clip(ref_con / max(cur_con, 1e-3), 0.5, 1.0)) if ref_con > 0 else 1.0
    k_sat = float(np.clip(ref_sat / max(cur_sat, 1e-3), 0.5, 1.0)) if ref_sat > 0 else 1.0
    if abs(k_con - 1.0) < 0.03 and abs(k_sat - 1.0) < 0.03:
        return new
    out = []
    for f in new:
        g = f[..., 0] * 0.114 + f[..., 1] * 0.587 + f[..., 2] * 0.299
        m = g.mean()
        f2 = np.clip(m + (f - m) * k_con, 0, 255) if abs(k_con - 1.0) >= 0.03 else f
        if abs(k_sat - 1.0) >= 0.03:
            hsv = cv2.cvtColor(np.clip(f2, 0, 255).astype("uint8"), cv2.COLOR_BGR2HSV).astype(float)
            hsv[..., 1] = np.clip(hsv[..., 1] * k_sat, 0, 255)
            f2 = cv2.cvtColor(np.clip(hsv, 0, 255).astype("uint8"), cv2.COLOR_HSV2BGR).astype(float)
        out.append(f2)
    return out


def _ramp_toward(orig, corrected, p=1.0):
    """Cross-fade `orig` into `corrected` along a 0->1 ramp across the list --
    a smooth ease, not a cut, so retouching clip1's own tail doesn't introduce
    a second internal seam. Weight 1.0 only lands on the very last frame.
    `p` < 1 rises faster early (sqrt ramp) so a correction covers more of the
    tail instead of only kicking in on the final frames."""
    np, _ = _np_cv2()
    n = len(orig)
    if n == 0:
        return orig
    w = np.linspace(0.0, 1.0, n) ** p
    return [np.clip((1 - wi) * o + wi * c, 0, 255) for o, c, wi in zip(orig, corrected, w)]


def _box_stats(frames, box):
    """Per-channel BGR mean+std over the visible box across `frames`."""
    np, _ = _np_cv2()
    x, y, w, h = box
    a = np.stack([f[y:y + h, x:x + w] for f in frames])
    return a.mean(axis=(0, 1, 2)), a.std(axis=(0, 1, 2))


def _affine_match(frames, cur_m, cur_sd, T_m, T_sd):
    """ONE per-channel affine map onto the target mean+std.

    Replaces the old luma-contrast scale + HSV-saturation scale pair: scaling
    around the luma mean already reduces chroma by ~k_con, then k_sat cut it
    AGAIN -- clip2 landed 20-25% undersaturated (the visible colour step at
    the seam). A single affine per channel matches brightness, contrast AND
    colour axes coherently, nothing is double-counted."""
    np, _ = _np_cv2()
    g = np.clip(T_sd / np.maximum(cur_sd, 1e-3), 0.6, 1.6)
    return [np.clip((f - cur_m) * g + T_m, 0, 255) for f in frames]


def _windowed_sharp(new, T_sharp, box, W):
    """Match texture only across the first W frames, correction ramped 1->0.

    The old full-clip match Gaussian-blurred ALL of clip2 down to clip1's
    motion-diffused TAIL level (the softest frames of clip1) -- the entire
    second half turned to mush. Seam-local matching keeps the body's native
    detail."""
    np, _ = _np_cv2()
    if len(new) <= 2 or W <= 0:
        return new
    W = min(W, len(new))
    head = [f.copy() for f in new[:W]]
    corr = _match_seam_sharpness(head, T_sharp, box)
    ramp = np.linspace(1.0, 0.0, W)
    out = [np.clip(wi * c + (1 - wi) * o, 0, 255)
           for o, c, wi in zip(new[:W], corr, ramp)]
    return out + new[W:]


def _body_sharpen(new, c1, box):
    """Bring clip2's BODY texture up to clip1's BODY level with one
    binary-searched unsharp. The camera-kept continuation renders slightly
    softer overall; this recovers it without touching the seam logic."""
    np, cv2 = _np_cv2()
    body1 = c1[10:max(11, len(c1) - 10):5]
    body2 = new[8::5]
    ref = _sharp_metric(body1, box)
    cur = _sharp_metric(body2, box)
    if cur >= ref * 0.9 or cur <= 1e-3 or not body2:
        return new
    sample = body2[:10]
    lo, hi = 0.0, 3.0
    for _ in range(10):
        mid = (lo + hi) / 2
        test = [np.clip(f + (f - cv2.GaussianBlur(f, (0, 0), 3)) * mid, 0, 255)
                for f in sample]
        (lo, hi) = (mid, hi) if _sharp_metric(test, box) < ref else (lo, mid)
    amt = hi
    if amt <= 0.02:
        return new
    return [np.clip(f + (f - cv2.GaussianBlur(f, (0, 0), 3)) * amt, 0, 255)
            for f in new]


def _even(x):
    x = int(round(x))
    return x - (x % 2)


def _resample_speed(frames, F):
    """Uniformly resample `frames` to change speed by factor F (F>1 = faster =
    fewer frames). Done in-memory by nearest-frame pick, NOT an ffmpeg setpts+fps
    filter -- that resample dropped/duped a frame right at the concat boundary and
    spiked the seam (made the speed step worse, not better)."""
    n = len(frames)
    if n < 2 or abs(F - 1.0) < 0.03:
        return frames
    tgt = max(2, int(round(n / F)))
    return [frames[min(n - 1, int(round(i * F)))] for i in range(tgt)]


def _resample_blend(frames, F):
    """Retime by factor F with FRACTIONAL frame blending.

    Nearest-frame picking creates a skip CADENCE (every ~1/(F-1) frames the
    motion double-steps -- measured as a periodic flow spike, reads as
    objects/shadows jumping). Sampling at fractional positions and linearly
    blending the two neighbours keeps motion even; static texture is
    untouched (identical neighbours blend to themselves) and slow drone pans
    only pick up sub-pixel motion blur. Full minterpolate re-synthesis was
    tried and rejected: it softened the whole clip."""
    np, _ = _np_cv2()
    n = len(frames)
    if n < 2 or abs(F - 1.0) < 0.03:
        return frames
    tgt = max(2, int(round(n / F)))
    out = []
    for i in range(tgt):
        pos = min(n - 1.0001, i * F)
        k = int(pos)
        a = pos - k
        if a < 0.02 or k + 1 >= n:
            out.append(frames[k])
        else:
            out.append(np.clip((1 - a) * frames[k] + a * frames[k + 1], 0, 255))
    return out


def _dedrift_bands(new, c1_tail, box, n_bands=3, alpha=0.0):
    """Kill DIVERGENT per-band colour drift after the global match.

    Measured on real clips: after the cut the sky band drifts lighter while
    the foreground drifts darker -- opposite directions, so no global affine
    can hold both. Fit each horizontal band's per-channel mean trend over
    clip2, pin it to clip1's tail band means, and apply the per-band offsets
    blended smoothly across the frame height (no visible band edges)."""
    np, _ = _np_cv2()
    if len(new) < 5 or not c1_tail:
        return new
    x, y, w, h = box
    H = new[0].shape[0]
    edges = [int(round(H * k / n_bands)) for k in range(n_bands + 1)]

    def band_means(frame):
        return np.array([[frame[edges[b]:edges[b + 1], x:x + w, c].mean()
                          for c in range(3)] for b in range(n_bands)])

    ref = np.mean([band_means(f) for f in c1_tail[-8:]], axis=0)
    per_frame = np.array([band_means(f) for f in new])       # (n, bands, 3)
    n = len(new)
    # target = the same alpha-blended midpoint the affine match used -- pinning
    # to clip1's raw tail here would overshoot past the affine target and
    # reintroduce a step at the cut (measured -2L when this used raw ref).
    head = per_frame[:min(8, n)].mean(axis=0)
    T = ref + alpha * (head - ref)
    idx = np.arange(n)
    centers = np.array([(edges[b] + edges[b + 1]) / 2 for b in range(n_bands)])
    rows = np.arange(H)
    deg = 2 if n >= 8 else 1
    for b in range(n_bands):
        for c in range(3):
            trend = np.polyval(np.polyfit(idx, per_frame[:, b, c], deg), idx)
            per_frame[:, b, c] = T[b, c] - trend             # offset to apply
    for i, f in enumerate(new):
        for c in range(3):
            off_rows = np.interp(rows, centers, per_frame[i, :, c])
            f[..., c] = np.clip(f[..., c] + off_rows[:, None], 0, 255)
    return new


def _join_graph(fps, js, cw, ch, cx, cy, w, h):
    """Filter graph: clip1 (input 0) hard-cut to the de-drifted/resampled
    continuation png sequence (input 1, already the right frames at the right
    speed), clip2 sharpen, concat, then crop the hallucinated pan-revealed border
    and scale back to full frame. NO xfade and NO mid-stream retime -- both caused
    a boundary hiccup; the cut is continuous by construction."""
    sh = f"unsharp=5:5:{js:.2f}:5:5:0.0," if js and js > 0 else ""
    a = f"[0:v]{sh}fps={fps},settb=AVTB,setpts=PTS-STARTPTS[a];"
    b = f"[1:v]{sh}fps={fps},settb=AVTB,setpts=PTS-STARTPTS[b];"
    crop = (f"crop={cw}:{ch}:{cx}:{cy},scale={w}:{h}:flags=lanczos"
            if (cw < w or ch < h) else f"scale={w}:{h}")
    return a + b + f"[a][b]concat=n=2:v=1,{crop}[o]"


def gold_join(cfg, clip1, raw, dst):
    """clip1 + de-drifted/colour-matched/speed-resampled continuation, hard-cut,
    edge-cropped -> ~10s with no seam jump, speed step, edge grain or colour drift.

    Contrast/saturation/sharpness meet in the MIDDLE: a fixed full-strength
    match-clip2-to-clip1 made clip2 look like the one being 'fixed' and clip1
    untouched -- asymmetric, reads as one half corrected and one half not.
    Blend the target toward clip2 by `seam_blend_alpha` (clip2 gets LESS
    correction) and ease clip1's own last few frames toward that same target
    (clip1 gets a SMALL nudge), so both halves move toward a shared midpoint
    instead of clip2 alone being pulled all the way to clip1."""
    np, cv2 = _np_cv2()
    ov = cfg.overlap_frames
    c1 = _read_all(clip1)
    c2 = _read_all(raw)
    if not c1 or not c2:
        return False
    n1 = len(c1)
    h, w = c1[0].shape[:2]
    m = _match_offset(clip1, raw, n1, 2 * ov + 6)    # cont frame == clip1's last
    cut = m + 1                                       # first genuinely new frame
    # edge crop dims (also the visible region for colour matching)
    ec = float(getattr(cfg, "edge_crop", 0.0))
    cw, ch = _even(w * (1 - 2 * ec)), _even(h * (1 - 2 * ec))
    cx, cy = (w - cw) // 2, (h - ch) // 2
    box = (cx, cy, cw, ch)
    # clip1 tail colour reference, measured on the VISIBLE (post-crop) centre so
    # clip2's visible brightness matches even as the camera moves content around.
    K = min(8, n1)
    TAIL = min(n1, max(16, 2 * K))   # window of clip1's own last frames to ease
    ref_bgr = np.mean([f[cy:cy + ch, cx:cx + cw] for f in c1[-K:]], axis=(0, 1, 2))
    new = c2[cut:]
    if not new:
        return False
    alpha = float(getattr(cfg, "seam_blend_alpha", 0.35))
    c1_tail = [f.copy() for f in c1[-TAIL:]]
    if getattr(cfg, "clip2_dedrift", True):
        new = _dedrift_clip2(new, ref_bgr, box)
        ref_m, ref_sd = _box_stats(c1[-K:], box)
        cur_m, cur_sd = _box_stats(new[:K], box)
        T_m = ref_m + alpha * (cur_m - ref_m)
        T_sd = ref_sd + alpha * (cur_sd - ref_sd)
        new = _affine_match(new, cur_m, cur_sd, T_m, T_sd)
        if getattr(cfg, "band_dedrift", True):
            new = _dedrift_bands(new, c1, box, alpha=alpha)
        if alpha > 0.001:
            t_m, t_sd = _box_stats(c1_tail, box)
            tail_corr = _affine_match([f.copy() for f in c1_tail],
                                      t_m, t_sd, T_m, T_sd)
            c1_tail = _ramp_toward(c1_tail, tail_corr)
    # speed: match clip1's BODY speed to clip2's BODY speed (both excluding the
    # seam transient). Applied by in-memory resample. The camera-kept
    # continuation runs ~0.85-0.9x; the wide clamp also covers the occasional
    # slow render without letting a bad flow reading halve the clip.
    if getattr(cfg, "continuation_speed_match", False):
        f1 = _flow(clip1, 4, n1 - 4)
        f2 = _flow(raw, cut + 3, 10 ** 9)
        F = float(np.clip(f1 / max(f2, 1e-3), 0.8,
                          float(getattr(cfg, "speed_clamp_hi", 1.8))))
        if getattr(cfg, "speed_retime_mc", True):
            new = _resample_blend(new, F)
        else:
            new = _resample_speed(new, F)
    if getattr(cfg, "body_sharpen", True):
        new = _body_sharpen(new, c1, box)
    if getattr(cfg, "clip2_dedrift", True):
        # STRUCTURE equalize-up: when the continuation's grain-blind structure
        # is cleaner than clip1's motion-diffused tail, sharpen the tail UP to
        # meet it (never blur the clean side down) -- a midpoint at the blur
        # reads as "end of clip1 goes soft, clip2 snaps clean". Plain varLap
        # cannot make this call (grain counts as sharpness), hence the
        # dedicated grain-blind metric.
        # grain targets measured BEFORE any tail edit -- the struct sharpen
        # below raises the tail's grain reading, and chaining the window
        # matcher off that pumped clip2's head into a sharpness bump.
        ref_sharp = _sharp_metric(c1_tail[-K:], box)
        cur_sharp = _sharp_metric(new[:K], box)
        T_sharp = ref_sharp + alpha * (cur_sharp - ref_sharp)
        ref_st = _struct_metric(c1_tail[-K:], box)
        cur_st = _struct_metric(new[:K], box)
        struct_fired = cur_st > ref_st * 1.05
        if struct_fired:
            a_up = float(getattr(cfg, "seam_sharp_alpha_up", 0.85))
            T_st = ref_st + a_up * (cur_st - ref_st)
            tail_corr = _sharpen_to_struct([f.copy() for f in c1_tail],
                                           T_st, box)
            c1_tail = _ramp_toward(c1_tail, tail_corr,
                                   float(getattr(cfg, "tail_ramp_pow", 0.5)))
        new = _windowed_sharp(new, T_sharp, box,
                              int(getattr(cfg, "seam_sharp_window", 16)))
        if alpha > 0.001 and not struct_fired:
            # tail grain-pull would Gaussian the freshly sharpened tail back
            # down -- only run it when the struct equalize did not.
            tail_corr = _match_seam_sharpness([f.copy() for f in c1_tail],
                                               T_sharp, box, K=TAIL)
            c1_tail = _ramp_toward(c1_tail, tail_corr)
    c1_mod = c1[:-TAIL] + c1_tail
    tmp = tempfile.mkdtemp(prefix="snf_join_")
    try:
        for i, f in enumerate(c1_mod):
            cv2.imwrite(os.path.join(tmp, f"c{i:05d}.png"),
                        np.clip(f, 0, 255).astype("uint8"))
        for i, f in enumerate(new):
            cv2.imwrite(os.path.join(tmp, f"n{i:05d}.png"),
                        np.clip(f, 0, 255).astype("uint8"))
        fc = _join_graph(cfg.fps, cfg.join_sharpen, cw, ch, cx, cy, w, h)
        return _run([cfg.ffmpeg, "-y", "-loglevel", "error",
                     "-framerate", str(cfg.fps), "-i", os.path.join(tmp, "c%05d.png"),
                     "-framerate", str(cfg.fps), "-i", os.path.join(tmp, "n%05d.png"),
                     "-filter_complex", fc, "-map", "[o]", "-r", str(cfg.fps),
                     "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", dst]) \
        and os.path.exists(dst)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _progressive_contrast(files, cfg):
    """Flatten Wan's contrast/saturation drift against its linear trend, in place.
    Gentle early, stronger late; keeps each frame's natural deviation."""
    np, cv2 = _np_cv2()
    n = len(files)

    def luma(b):
        return b[..., 2] * 0.299 + b[..., 1] * 0.587 + b[..., 0] * 0.114

    S = np.zeros(n)
    SAT = np.zeros(n)
    imgs = []
    for i, p in enumerate(files):
        f = cv2.imread(p).astype(float)
        imgs.append(f)
        S[i] = luma(f).std()
        SAT[i] = cv2.cvtColor(np.clip(f, 0, 255).astype("uint8"),
                              cv2.COLOR_BGR2HSV)[..., 1].mean()
    idx = np.arange(n)
    bs, as_ = np.polyfit(idx, S, 1)
    Ts = as_ + bs * idx
    bt, at_ = np.polyfit(idx, SAT, 1)
    Tsat = at_ + bt * idx
    hi = min(50, n)
    tgtS = float(np.mean(S[10:hi])) * cfg.contrast_boost
    tgtSAT = float(np.mean(SAT[10:hi])) * cfg.saturation_boost
    for i, f in enumerate(imgs):
        cs = float(np.clip(tgtS / max(Ts[i], 1e-3), 0.7, 1.2))
        for c in range(3):
            m = f[..., c].mean()
            f[..., c] = (f[..., c] - m) * cs + m
        ss = float(np.clip(tgtSAT / max(Tsat[i], 1e-3), 0.7, 1.2))
        L = luma(f)[..., None]
        f = L + (f - L) * ss
        cv2.imwrite(files[i], np.clip(f, 0, 255).astype("uint8"))


def esrgan_available(cfg):
    return bool(cfg.esrgan_bin) and os.path.exists(cfg.esrgan_bin) \
        and os.path.isdir(cfg.esrgan_models_dir)


def _global_stats_video(path, sample=16):
    """Per-channel RGB mean+std over `sample` frames spread across the clip."""
    np, cv2 = _np_cv2()
    n = _frame_count(path) or 1
    idx = set(int(round(x)) for x in np.linspace(0, max(0, n - 1), sample))
    cap = cv2.VideoCapture(path)
    i = 0
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in idx:
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(float))
        i += 1
    cap.release()
    a = np.stack(frames)
    return a.mean(axis=(0, 1, 2)), a.std(axis=(0, 1, 2))


def _global_stats_pngs(files, sample=16):
    np, cv2 = _np_cv2()
    step = max(1, len(files) // sample)
    frames = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB).astype(float)
              for p in files[::step]]
    a = np.stack(frames)
    return a.mean(axis=(0, 1, 2)), a.std(axis=(0, 1, 2))


def _mean_saturation_video(path, sample=16):
    np, cv2 = _np_cv2()
    n = _frame_count(path) or 1
    idx = set(int(round(x)) for x in np.linspace(0, max(0, n - 1), sample))
    cap = cv2.VideoCapture(path)
    i = 0
    s = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in idx:
            s.append(float(cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)[..., 1].mean()))
        i += 1
    cap.release()
    return float(np.mean(s)) if s else 0.0


def _mean_saturation_pngs(files, sample=16):
    np, cv2 = _np_cv2()
    step = max(1, len(files) // sample)
    s = [float(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2HSV)[..., 1].mean())
         for p in files[::step]]
    return float(np.mean(s)) if s else 0.0


def _norm_lut(src_m, src_sd, cur_m, cur_sd):
    """lutrgb that maps the CURRENT colour distribution back onto the SOURCE's
    per-channel mean+std (val' = (val-cur)*src_sd/cur_sd + src). remacri-4x punches
    contrast/saturation ('neon'); this undoes it against the real source clip."""
    np, _ = _np_cv2()
    g = np.clip(src_sd / np.maximum(cur_sd, 1e-3), 0.5, 1.5)
    ch = "rgb"
    parts = [f"{ch[c]}=clip((val-{cur_m[c]:.2f})*{g[c]:.3f}+{src_m[c]:.2f}\\,0\\,255)"
             for c in range(3)]
    return "lutrgb=" + ":".join(parts)


def esrgan_finish(cfg, src, dst):
    """Per-frame de-drift -> Real-ESRGAN super-res -> colour-match back to source
    (kills remacri 'neon') -> UHD scale + crisp + grain."""
    np, _ = _np_cv2()
    tmp = tempfile.mkdtemp(prefix="snf_finish_")
    inF = os.path.join(tmp, "in")
    outF = os.path.join(tmp, "out")
    os.makedirs(inF)
    os.makedirs(outF)
    try:
        out_fps = cfg.fps
        final_fps = int(getattr(cfg, "final_fps", 0))
        if final_fps and final_fps != cfg.fps:
            # motion-compensated interpolation BEFORE super-res: Wan's native
            # 16fps judders on a big screen; mci doubles it cleanly on slow
            # pans (checked for warp artifacts on real footage).
            interp = os.path.join(tmp, "interp.mp4")
            if not _run([cfg.ffmpeg, "-y", "-loglevel", "error", "-i", src,
                         "-vf",
                         f"minterpolate=fps={final_fps}:mi_mode=mci:"
                         "mc_mode=aobmc:me_mode=bidir:vsbmc=1",
                         "-c:v", "libx264", "-crf", "14", "-pix_fmt",
                         "yuv420p", interp]):
                return False
            src = interp
            out_fps = final_fps
        if not _run([cfg.ffmpeg, "-y", "-loglevel", "error", "-i", src,
                     os.path.join(inF, "f%05d.png")]):
            return False
        files = sorted(glob.glob(os.path.join(inF, "*.png")))
        if not files:
            return False
        # source colour BEFORE the de-drift edits the input frames in place
        src_m, src_sd = (_global_stats_video(src)
                         if cfg.esrgan_color_match else (None, None))
        src_sat = (_mean_saturation_video(src)
                   if cfg.esrgan_saturation_match else None)
        if cfg.contrast_flatten:
            _progressive_contrast(files, cfg)
        if subprocess.run([cfg.esrgan_bin, "-i", inF, "-o", outF, "-n",
                           cfg.esrgan_model, "-m", cfg.esrgan_models_dir,
                           "-s", "4", "-f", "png"],
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode != 0:
            return False
        chain = []
        sr_files = sorted(glob.glob(os.path.join(outF, "*.png")))
        if cfg.esrgan_color_match:
            cur_m, cur_sd = _global_stats_pngs(sr_files)
            chain.append(_norm_lut(src_m, src_sd, cur_m, cur_sd))
        if cfg.esrgan_saturation_match:
            # remacri over-saturates ('neon'); pull HSV saturation back to the
            # source clip's level (per-channel mean/std alone left a residual).
            sr_sat = _mean_saturation_pngs(sr_files)
            ssat = float(np.clip(src_sat / max(sr_sat, 1e-3), 0.55, 1.05))
            chain.append(f"eq=saturation={ssat:.3f}")
        if cfg.final_tdenoise:
            chain.append(f"hqdn3d={cfg.final_tdenoise}")
        chain.append(f"scale=-2:{cfg.final_height}:flags=lanczos")
        if cfg.final_unsharp and cfg.final_unsharp != "0:0:0:0:0:0":
            chain.append(f"unsharp={cfg.final_unsharp}")
        if cfg.final_grain:
            chain.append(f"noise={cfg.final_grain}")
        if getattr(cfg, "final_grade", ""):
            chain.append(cfg.final_grade)
        vf = ",".join(chain)
        return _run([cfg.ffmpeg, "-y", "-loglevel", "error", "-framerate",
                     str(out_fps), "-i", os.path.join(outF, "f%05d.png"),
                     "-vf", vf] + _codec(cfg, cfg.final_cq) + [dst]) \
            and os.path.exists(dst)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
