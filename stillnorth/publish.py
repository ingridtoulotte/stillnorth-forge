"""Format-agnostic finishing for the loop-publishing pivot (§5E).

Given ANY seamless base loop (slideshow.py output, or a future video loop), this
adds an ambient audio bed, expands it to duration tiers, and cuts a 9:16 Shorts
version -- all ffmpeg, no GPU.

Audio is PROCEDURAL (ffmpeg `anoisesrc` shaped per preset), so a silent visual
loop becomes a set of distinct videos at ~zero cost -- the biggest §5E multiplier
("one visual loop x N soundscapes = N videos"). A duration tier is built by
stream-COPYING the base loop up to the target length and generating a FRESH audio
bed at that full length (never looping the audio), so tiers have no audio seam.
"""
import math
import os

from . import media


# preset -> (anoisesrc color, amplitude, post -af chain)
AUDIO_PRESETS = {
    # airy pink noise, rolled off = wind through pines, with a slow swell
    "wind": ("pink", 0.9, "highpass=f=45,lowpass=f=700,tremolo=f=0.1:d=0.7,volume=0.40"),
    # brighter, band-limited hiss = steady rain
    "rain": ("white", 0.6, "highpass=f=500,lowpass=f=7000,volume=0.28"),
    # deep brown-noise bed = low ambient drone
    "drone": ("brown", 1.0, "lowpass=f=180,volume=0.50"),
    # near-silent airflow for a very calm bed
    "still": ("brown", 0.5, "lowpass=f=120,volume=0.22"),
}

DEFAULT_TIERS = {"1min": 60, "30min": 1800, "1h": 3600}


def _run(cfg, args, cancel=None):
    ok, _ = media.run_cancellable([cfg.ffmpeg, "-y", "-loglevel", "error"] + args,
                                  cancel=cancel)
    return ok


def _audio_input(kind, seconds):
    """(-f lavfi -i ... , af_chain) for a procedural bed of `seconds`."""
    color, amp, chain = AUDIO_PRESETS.get(kind, AUDIO_PRESETS["wind"])
    src = f"anoisesrc=color={color}:amplitude={amp}:duration={seconds:g}"
    return ["-f", "lavfi", "-i", src], chain


def make_audio_bed(cfg, kind, seconds, dst, cancel=None):
    """Render a standalone procedural ambient bed (wav) of `seconds`."""
    inp, chain = _audio_input(kind, seconds)
    args = inp + ["-af", chain, "-t", f"{seconds:g}", dst]
    return _run(cfg, args, cancel=cancel) and os.path.exists(dst)


def mux_audio(cfg, video, audio, dst, cancel=None):
    """Mux an audio file onto a video (video stream-copied, aac audio)."""
    args = ["-i", video, "-i", audio, "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
            "-movflags", "+faststart", dst]
    return _run(cfg, args, cancel=cancel) and os.path.exists(dst)


def loop_reps(target_seconds, loop_seconds):
    """`-stream_loop` count (EXTRA repeats) needed to cover target_seconds.

    stream_loop N plays the input N+1 times, so we need N such that
    (N+1)*loop >= target -> N = ceil(target/loop) - 1, floored at 0, plus one
    extra loop of headroom so the final -t trim always has material."""
    if loop_seconds <= 0:
        return 0
    return max(0, int(math.ceil(target_seconds / loop_seconds)) - 1) + 1


def export_tier(cfg, loop, dst, seconds, audio_kind="wind", loop_seconds=None,
                cancel=None):
    """Expand a base loop to `seconds` with a fresh full-length audio bed.

    The video is stream-COPIED (`-c:v copy`, no re-encode -- scales to 8h cheap);
    the audio is generated inline at the tier's full length so there is no audio
    loop seam."""
    if loop_seconds is None:
        loop_seconds = media.probe_duration(cfg, loop) or 0.0
    reps = loop_reps(seconds, loop_seconds)
    ainp, chain = _audio_input(audio_kind, seconds)
    args = ["-stream_loop", str(reps), "-i", loop] + ainp + [
        "-t", f"{seconds:g}",
        "-filter_complex", f"[1:a]{chain}[aud]",
        "-map", "0:v:0", "-map", "[aud]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart", dst]
    return _run(cfg, args, cancel=cancel) and os.path.exists(dst)


def export_tiers(cfg, loop, out_dir, tiers=None, audio_kind="wind", stem="loop",
                 cancel=None, log=None):
    """Export a base loop at every duration tier. Returns {tier_key: path}."""
    tiers = tiers or DEFAULT_TIERS
    os.makedirs(out_dir, exist_ok=True)
    loop_seconds = media.probe_duration(cfg, loop) or 0.0
    out = {}
    for key, seconds in tiers.items():
        if cancel and cancel():
            break
        dst = os.path.join(out_dir, f"{stem}_{key}.mp4")
        if export_tier(cfg, loop, dst, float(seconds), audio_kind=audio_kind,
                       loop_seconds=loop_seconds, cancel=cancel):
            out[key] = dst
            if log:
                log(f"tier {key} exported ({seconds}s)")
        elif log:
            log(f"tier {key} FAILED")
    return out


def shorts_crop(cfg, src, dst, seconds=None, cancel=None):
    """Cut a 9:16 vertical (1080x1920) Shorts version from a 16:9 source.

    Centre-crops to 9:16 then scales to 1080x1920; keeps/copies audio if present.
    Pass `seconds` to cap the length (Shorts are <=60s)."""
    vf = "crop=ih*9/16:ih,scale=1080:1920:flags=lanczos,setsar=1"
    codec = (["-c:v", "hevc_nvenc", "-preset", "p7", "-cq", str(cfg.final_cq),
              "-pix_fmt", "yuv420p"] if cfg.nvenc else
             ["-c:v", "libx264", "-preset", "slow", "-crf", str(cfg.final_cq),
              "-pix_fmt", "yuv420p"])
    args = ["-i", src]
    if seconds:
        args += ["-t", f"{seconds:g}"]
    args += ["-vf", vf, "-map", "0:v:0"] + codec
    # carry audio through if the source has any (the trailing ? makes it optional)
    args += ["-map", "0:a:0?", "-c:a", "aac", "-b:a", "192k",
             "-movflags", "+faststart", dst]
    return _run(cfg, args, cancel=cancel) and os.path.exists(dst)
