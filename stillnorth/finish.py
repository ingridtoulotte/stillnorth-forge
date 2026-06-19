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


def _join_graph(cor, cut, fps, retime):
    """Hard-cut concat: clip1 in full, then the continuation from frame `cut`
    (its first genuinely new frame) colour-matched + optionally retimed to clip1's
    motion speed. NO xfade -- cross-dissolving clip1's tail with Wan's re-rendered
    copy of the same frames ghosted ~1s of motion (read as blur) and snapped at the
    end (forward jump). The cut is continuous by construction (cut == the frame
    after clip1's last), so a straight join is seamless once colour is matched."""
    a = "[0:v]settb=AVTB,setpts=PTS-STARTPTS[a];"
    if retime and abs(retime - 1.0) >= 1e-3:
        b = (f"[1:v]{cor},trim=start_frame={cut},"
             f"setpts=(PTS-STARTPTS)/{retime:.3f},fps={fps},"
             f"settb=AVTB,setpts=PTS-STARTPTS[b];")
    else:
        b = (f"[1:v]{cor},trim=start_frame={cut},"
             f"setpts=PTS-STARTPTS,settb=AVTB[b];")
    return a + b + "[a][b]concat=n=2:v=1[o]"


def gold_join(cfg, clip1, raw, dst):
    """clip1 + colour-matched/(retimed) continuation, hard-cut -> seamless ~10s."""
    np, _ = _np_cv2()
    ov = cfg.overlap_frames
    n1 = _frame_count(clip1)
    m = _match_offset(clip1, raw, n1, 2 * ov + 6)    # cont frame == clip1's last
    cut = m + 1                                       # first genuinely new frame
    mlen = m + 1                                      # frames in the matched overlap
    cm, cs = _rgb_stats(clip1, n1 - mlen, n1 - 1)     # clip1 tail (matched window)
    om, osd = _rgb_stats(raw, 0, m)                   # continuation's copy of it
    g = np.clip(cs / np.maximum(osd, 1e-3), 0.6, 1.7)
    lut = (f"r=clip((val-{om[0]:.2f})*{g[0]:.3f}+{cm[0]:.2f}\\,0\\,255):"
           f"g=clip((val-{om[1]:.2f})*{g[1]:.3f}+{cm[1]:.2f}\\,0\\,255):"
           f"b=clip((val-{om[2]:.2f})*{g[2]:.3f}+{cm[2]:.2f}\\,0\\,255)")
    cor = f"lutrgb={lut},unsharp=5:5:{cfg.join_sharpen:.2f}:5:5:0.0"
    F = None
    if getattr(cfg, "continuation_speed_match", False):
        f1 = _flow(clip1)              # clip1 = established motion speed
        f2 = _flow(raw, cut)          # the NEW continuation frames
        F = float(np.clip(f1 / max(f2, 1e-3), 0.7, 1.7))
    fc = _join_graph(cor, cut, cfg.fps, F)
    return _run([cfg.ffmpeg, "-y", "-loglevel", "error", "-i", clip1, "-i", raw,
                 "-filter_complex", fc, "-map", "[o]", "-r", str(cfg.fps),
                 "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", dst]) \
        and os.path.exists(dst)


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
        if cfg.contrast_flatten:
            _progressive_contrast(files, cfg)
        if subprocess.run([cfg.esrgan_bin, "-i", inF, "-o", outF, "-n",
                           cfg.esrgan_model, "-m", cfg.esrgan_models_dir,
                           "-s", "4", "-f", "png"],
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode != 0:
            return False
        chain = []
        if cfg.esrgan_color_match:
            sr_files = sorted(glob.glob(os.path.join(outF, "*.png")))
            cur_m, cur_sd = _global_stats_pngs(sr_files)
            chain.append(_norm_lut(src_m, src_sd, cur_m, cur_sd))
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
