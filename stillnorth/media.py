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


def run_cancellable(args, cancel=None, poll=0.5):
    """Run ffmpeg as a child process that can be aborted mid-encode.

    Polls `cancel()` while it runs; on cancel the process is terminated and
    (False, "cancelled") is returned. Lets a long compilation stop promptly
    instead of blocking on one multi-minute ffmpeg call."""
    import time
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    try:
        while proc.poll() is None:
            if cancel and cancel():
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
                return False, "cancelled"
            time.sleep(poll)
    finally:
        if proc.poll() is None:
            proc.kill()
    return (proc.returncode == 0), ("ok" if proc.returncode == 0 else "ffmpeg error")


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
    """Join clip a + clip b into the ~10s master.

    - clip a: drop the first `trim_start` frames (the img2vid start glitch).
    - clip b: drop frame 0 (it == a's last frame, b was generated from it) and
      sharpen to match a -- the continuation clip drifts slightly soft, so a
      light unsharp brings its detail back up to clip 1's level.
    """
    n = max(0, int(getattr(cfg, "trim_start", 0)))
    a_chain = f"[0:v]trim=start_frame={n},setpts=N/FRAME_RATE/TB[a]" if n > 0 \
        else "[0:v]setpts=N/FRAME_RATE/TB[a]"
    b_chain = "[1:v]trim=start_frame=1,setpts=N/FRAME_RATE/TB"
    if getattr(cfg, "clip2_sharpen", ""):
        b_chain += f",unsharp={cfg.clip2_sharpen}"
    b_chain += "[b]"
    fc = f"{a_chain};{b_chain};[a][b]concat=n=2:v=1[o]"
    return _run([cfg.ffmpeg, "-y", "-loglevel", "error", "-i", a, "-i", b,
                 "-filter_complex", fc, "-map", "[o]", "-r", str(cfg.fps)]
                + _codec_args(cfg) + [dst]) and os.path.exists(dst)


def probe_duration(cfg, path):
    """Duration of a clip in seconds (float), or None."""
    try:
        out = subprocess.run(
            [cfg.ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, timeout=30).stdout.strip()
        return float(out.splitlines()[0]) if out else None
    except Exception:
        return None


def normalize_segment(cfg, src, dst_ts, height, fps, fade, duration, cancel=None):
    """Render one library clip into a normalised, faded MPEG-TS segment so that
    many of them concatenate by stream-copy (no re-encode) into the long video.

    Scales to `height` (keeping aspect, even width), forces constant `fps`, and
    fades to/from black for `fade` seconds at each end. All segments share the
    same codec, pixel format and timebase, which is what makes the final
    concat a cheap copy instead of a multi-hour re-encode."""
    fade = max(0.0, float(fade))
    out_start = max(0.0, (duration or 0.0) - fade)
    chain = [f"scale=-2:{int(height)}:flags=lanczos", f"fps={fps}", "format=yuv420p"]
    if fade > 0 and duration:
        chain.append(f"fade=t=in:st=0:d={fade:g}")
        chain.append(f"fade=t=out:st={out_start:g}:d={fade:g}")
    vf = ",".join(chain)
    args = [cfg.ffmpeg, "-y", "-loglevel", "error", "-i", src, "-an",
            "-vf", vf, "-r", str(fps)] + _codec_args(cfg) + [
            "-g", str(fps * 2), "-keyint_min", str(fps * 2), "-sc_threshold", "0",
            "-video_track_timescale", "90000", "-f", "mpegts", dst_ts]
    ok, _ = run_cancellable(args, cancel=cancel)
    return ok and os.path.exists(dst_ts) and os.path.getsize(dst_ts) > 0


def concat_copy(cfg, list_file, dst, cancel=None):
    """Concatenate the MPEG-TS segments named in `list_file` (ffmpeg concat
    demuxer) by stream copy into the final mp4 -- fast, no re-encode."""
    args = [cfg.ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
            "-i", list_file, "-c", "copy", "-movflags", "+faststart", dst]
    ok, _ = run_cancellable(args, cancel=cancel)
    return ok and os.path.exists(dst) and os.path.getsize(dst) > 0
