"""Native-overlap continuation join + ESRGAN/contrast UHD finisher.

This is the proven "The Still North" recipe, verified across 6 sources:
  - GOLD JOIN: the continuation's first `overlap` frames are the SAME content as
    clip 1's last `overlap` frames (Wan reproduced them as known context), so they
    are an exact colour reference. Match the continuation to clip 1 (per-channel
    RGB mean+std), light unsharp, then xfade ACROSS the overlap -> invisible morph
    (not a content fade).
  - FINISH: per-frame progressive contrast+saturation de-drift (Wan drifts toward
    higher contrast over a continuation -> "neon end"; corrected against the linear
    trend, gentle early / stronger late, keeping natural variation), then
    Real-ESRGAN (Upscayl) per-frame super-res, then UHD scale + crisp + grain.

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


def gold_join(cfg, clip1, raw, dst):
    """clip1 + overlap-matched/xfaded continuation -> seamless ~10s native clip."""
    np, _ = _np_cv2()
    ov = cfg.overlap_frames
    n1 = _frame_count(clip1)
    cm, cs = _rgb_stats(clip1, n1 - ov, n1 - 1)      # clip1 tail (the overlap)
    om, osd = _rgb_stats(raw, 0, ov - 1)             # continuation head (same content)
    g = np.clip(cs / np.maximum(osd, 1e-3), 0.6, 1.7)
    lut = (f"r=clip((val-{om[0]:.2f})*{g[0]:.3f}+{cm[0]:.2f}\\,0\\,255):"
           f"g=clip((val-{om[1]:.2f})*{g[1]:.3f}+{cm[1]:.2f}\\,0\\,255):"
           f"b=clip((val-{om[2]:.2f})*{g[2]:.3f}+{cm[2]:.2f}\\,0\\,255)")
    cor = f"lutrgb={lut},unsharp=5:5:0.5:5:5:0.0"
    dur = ov / cfg.fps
    off = (n1 - ov) / cfg.fps
    fc = (f"[1:v]{cor},setpts=PTS-STARTPTS[b];"
          f"[0:v][b]xfade=transition=fade:duration={dur:.4f}:offset={off:.4f}[o]")
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


def esrgan_finish(cfg, src, dst):
    """Per-frame de-drift -> Real-ESRGAN super-res -> UHD scale + crisp + grain."""
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
        if cfg.contrast_flatten:
            _progressive_contrast(files, cfg)
        if subprocess.run([cfg.esrgan_bin, "-i", inF, "-o", outF, "-n",
                           cfg.esrgan_model, "-m", cfg.esrgan_models_dir,
                           "-s", "4", "-f", "png"],
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode != 0:
            return False
        chain = []
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
