"""ffmpeg helpers (upscale / last-frame / concat) and GPU VRAM polling.

Upscaling is multiplier-based (scale=iw*N:ih*N) so it is resolution-agnostic and
honours the literal "x2 / x4" the pipeline asks for. The video chain reuses the
project's proven Route-1 recipe: hqdn3d -> lanczos -> unsharp -> filmic grain.
"""
import os
import subprocess


def _codec_args(cfg):
    if cfg.nvenc:
        return ["-c:v", "hevc_nvenc", "-preset", "p7", "-cq", str(cfg.cq),
                "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "slow", "-crf", str(cfg.cq),
            "-pix_fmt", "yuv420p"]


def _run(args):
    return subprocess.run(args, stdout=subprocess.DEVNULL,
                          stderr=subprocess.PIPE).returncode == 0


def vram():
    """(used_mb, total_mb, name) via nvidia-smi, or None if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip()
        used, total, name = [x.strip() for x in out.splitlines()[0].split(",")]
        return int(used), int(total), name
    except Exception:
        return None


def upscale_image(cfg, src, dst, mult):
    """Lanczos x`mult` upscale of a still image (+ subtle unsharp)."""
    vf = f"scale=iw*{mult}:ih*{mult}:flags=lanczos,unsharp=3:3:0.4:3:3:0.0"
    return _run([cfg.ffmpeg, "-y", "-loglevel", "error", "-i", src,
                 "-vf", vf, dst]) and os.path.exists(dst)


def upscale_video(cfg, src, dst, mult):
    """x`mult` video upscale with the Route-1 cleanup + grain chain."""
    chain = []
    if cfg.denoise:
        chain.append(f"hqdn3d={cfg.denoise}")
    chain.append(f"scale=iw*{mult}:ih*{mult}:flags=lanczos")
    if cfg.sharp and cfg.sharp != "0:0:0:0:0:0":
        chain.append(f"unsharp={cfg.sharp}")
    if cfg.grain:
        chain.append(f"noise={cfg.grain}")
    vf = ",".join(chain)
    return _run([cfg.ffmpeg, "-y", "-loglevel", "error", "-i", src,
                 "-vf", vf] + _codec_args(cfg) + [dst]) and os.path.exists(dst)


def last_frame(cfg, clip, dst):
    """Grab the very last frame (keep-last trick) as a PNG."""
    return _run([cfg.ffmpeg, "-y", "-loglevel", "error", "-sseof", "-3",
                 "-i", clip, "-update", "1", dst]) and os.path.exists(dst)


def concat_pair(cfg, a, b, dst):
    """Join clip a + clip b. b's frame 0 == a's last frame (b was generated
    from it), so drop b frame 0 to avoid a 1-frame freeze at the seam."""
    fc = ("[1:v]trim=start_frame=1,setpts=N/FRAME_RATE/TB[b];"
          "[0:v][b]concat=n=2:v=1[o]")
    return _run([cfg.ffmpeg, "-y", "-loglevel", "error", "-i", a, "-i", b,
                 "-filter_complex", fc, "-map", "[o]", "-r", str(cfg.fps)]
                + _codec_args(cfg) + [dst]) and os.path.exists(dst)
