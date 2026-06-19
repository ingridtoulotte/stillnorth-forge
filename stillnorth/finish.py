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


def _join_graph(fps, js, cw, ch, cx, cy, w, h):
    """Filter graph: clip1 (input 0) hard-cut to the de-drifted/resampled
    continuation png sequence (input 1, already the right frames at the right
    speed), clip2 sharpen, concat, then crop the hallucinated pan-revealed border
    and scale back to full frame. NO xfade and NO mid-stream retime -- both caused
    a boundary hiccup; the cut is continuous by construction."""
    a = f"[0:v]fps={fps},settb=AVTB,setpts=PTS-STARTPTS[a];"
    sh = f"unsharp=5:5:{js:.2f}:5:5:0.0," if js and js > 0 else ""
    b = f"[1:v]{sh}fps={fps},settb=AVTB,setpts=PTS-STARTPTS[b];"
    crop = (f"crop={cw}:{ch}:{cx}:{cy},scale={w}:{h}:flags=lanczos"
            if (cw < w or ch < h) else f"scale={w}:{h}")
    return a + b + f"[a][b]concat=n=2:v=1,{crop}[o]"


def gold_join(cfg, clip1, raw, dst):
    """clip1 + de-drifted/colour-matched/speed-resampled continuation, hard-cut,
    edge-cropped -> ~10s with no seam jump, speed step, edge grain or colour drift."""
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
    # clip1 tail colour reference, measured on the VISIBLE (post-crop) centre so
    # clip2's visible brightness matches even as the camera moves content around.
    K = min(8, n1)
    ref_bgr = np.mean([f[cy:cy + ch, cx:cx + cw] for f in c1[-K:]], axis=(0, 1, 2))
    new = c2[cut:]
    if not new:
        return False
    if getattr(cfg, "clip2_dedrift", True):
        new = _dedrift_clip2(new, ref_bgr, (cx, cy, cw, ch))
    # speed: match clip1's BODY speed to clip2's BODY speed (both excluding the
    # seam transient). Gentle, tightly clamped, applied by in-memory resample.
    if getattr(cfg, "continuation_speed_match", False):
        f1 = _flow(clip1, 4, n1 - 4)
        f2 = _flow(raw, cut + 3, 10 ** 9)
        F = float(np.clip(f1 / max(f2, 1e-3), 0.8, 1.25))
        new = _resample_speed(new, F)
    tmp = tempfile.mkdtemp(prefix="snf_join_")
    try:
        for i, f in enumerate(new):
            cv2.imwrite(os.path.join(tmp, f"n{i:05d}.png"),
                        np.clip(f, 0, 255).astype("uint8"))
        fc = _join_graph(cfg.fps, cfg.join_sharpen, cw, ch, cx, cy, w, h)
        return _run([cfg.ffmpeg, "-y", "-loglevel", "error",
                     "-i", clip1, "-framerate", str(cfg.fps),
                     "-i", os.path.join(tmp, "n%05d.png"),
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
        vf = ",".join(chain)
        return _run([cfg.ffmpeg, "-y", "-loglevel", "error", "-framerate",
                     str(cfg.fps), "-i", os.path.join(outF, "f%05d.png"),
                     "-vf", vf] + _codec(cfg, cfg.final_cq) + [dst]) \
            and os.path.exists(dst)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
