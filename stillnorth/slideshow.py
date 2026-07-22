"""Still-image slideshow lane (GPU-free) for the loop-publishing pivot (§5E).

Turns a list of curated 4K stills into ONE seamless-looping ambient clip:

    prep each still (lanczos -> target res + tonal grade for cohesion)
      -> render each into a slow Ken Burns clip (zoompan)
      -> append an identical-frame WRAP clip (a static hold of clip-0's first
         frame) so the sequence's last frame == its first frame
      -> xfade-dissolve the whole chain together

Because the last output frame is pixel-identical to the first, `-stream_loop`
(duration-tier export in publish.py) repeats it with an invisible hard cut --
the loop is seamless by construction, no wrap-dissolve math across the boundary.

Two-pass (render-each-then-chain) mirrors assembler.py's normalise-then-concat:
each pass is simple, cancellable, and the chain-graph builder is a pure function
that unit-tests without ffmpeg. Nothing here touches the Wan/FLUX render core.
"""
import os

from . import media


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _dims(height):
    """(width, height) for a 16:9 frame of the given height, both even
    (yuv420p requires even dimensions on both axes)."""
    h = int(height)
    h -= h % 2
    w = int(round(h * 16 / 9 / 2)) * 2
    return w, h


def _codec(cfg, cq):
    """media._codec_args, but with an explicit cq (KB intermediates want a
    tighter cq than the video_cq default so the xfade re-encode stays crisp)."""
    if cfg.nvenc:
        return ["-c:v", "hevc_nvenc", "-preset", "p7", "-cq", str(cq),
                "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "slow", "-crf", str(cq),
            "-pix_fmt", "yuv420p"]


def _run(cfg, args, cancel=None):
    ok, _ = media.run_cancellable([cfg.ffmpeg, "-y", "-loglevel", "error"] + args,
                                  cancel=cancel)
    return ok


# ---------------------------------------------------------------------------
# per-still preparation + Ken Burns
# ---------------------------------------------------------------------------
def prep_still(cfg, src, dst, height=2160):
    """Scale a source still to a 16:9 frame of `height` (cover+crop so any
    aspect works) and apply cfg.final_grade so every still shares one tone."""
    w, h = _dims(height)
    chain = [f"scale={w}:{h}:flags=lanczos:force_original_aspect_ratio=increase",
             f"crop={w}:{h}"]
    grade = (cfg.final_grade or "").strip()
    if grade:
        chain.append(grade)
    vf = ",".join(chain)
    return _run(cfg, ["-i", src, "-vf", vf, dst]) and os.path.exists(dst)


def kenburns_clip(cfg, img, dst, seconds, zoom=1.08, mode="in", pan=0.0,
                  height=2160, fps=None, cancel=None):
    """Render one prepared still into a slow Ken Burns clip.

    zoom  -- total magnification reached over the clip (1.08 = +8%).
    mode  -- "in" (1.0 -> zoom) or "out" (zoom -> 1.0).
    pan   -- horizontal drift, fraction of frame width (-1..1), applied gently.
    The still is pre-scaled 2x before zoompan so the sub-pixel zoom is smooth
    (zoompan on the native frame stair-steps)."""
    w, h = _dims(height)
    fps = int(fps or cfg.fps)
    frames = max(2, int(round(seconds * fps)))
    rate = (float(zoom) - 1.0) / (frames - 1)
    if mode == "out":
        # start zoomed on frame 0, ease back toward 1.0
        z = f"if(eq(on,0),{zoom:.4f},max(zoom-{rate:.6f},1.0))"
    else:
        z = f"min(zoom+{rate:.6f},{zoom:.4f})"
    # centre, plus a gentle linear horizontal drift of up to ~6% frame width
    x = f"iw/2-(iw/zoom/2)+({pan:.4f})*(on/{frames - 1})*iw*0.06"
    y = "ih/2-(ih/zoom/2)"
    vf = (f"scale={w * 2}:{h * 2}:flags=lanczos,"
          f"zoompan=z='{z}':d={frames}:x='{x}':y='{y}':s={w}x{h}:fps={fps},"
          f"format=yuv420p,setsar=1")
    args = ["-loop", "1", "-i", img, "-t", f"{seconds:.4f}", "-vf", vf] \
        + _codec(cfg, cfg.final_cq) + [dst]
    return _run(cfg, args, cancel=cancel) and os.path.exists(dst)


def static_clip(cfg, img, dst, seconds, height=2160, fps=None, cancel=None):
    """A dead-still hold of `img` (used for the identical-frame wrap clip)."""
    w, h = _dims(height)
    fps = int(fps or cfg.fps)
    vf = (f"scale={w}:{h}:flags=lanczos:force_original_aspect_ratio=increase,"
          f"crop={w}:{h},fps={fps},format=yuv420p,setsar=1")
    args = ["-loop", "1", "-i", img, "-t", f"{seconds:.4f}", "-vf", vf] \
        + _codec(cfg, cfg.final_cq) + [dst]
    return _run(cfg, args, cancel=cancel) and os.path.exists(dst)


def first_frame(cfg, clip, dst):
    """Extract the very first frame of a clip as a PNG."""
    return _run(cfg, ["-i", clip, "-frames:v", "1", dst]) and os.path.exists(dst)


# ---------------------------------------------------------------------------
# xfade chain graph -- PURE (unit-testable without ffmpeg)
# ---------------------------------------------------------------------------
def xfade_chain_graph(durations, xfade, transition="fade"):
    """Build the -filter_complex string that xfade-chains N clips.

    `durations` -- per-clip seconds (clips may differ in length).
    Returns (graph, out_label, total_seconds). For the i-th dissolve the offset
    is (cumulative length of the composite so far) - xfade, which for equal-length
    clips reduces to i*(dur-xfade). Total = sum(durations) - (N-1)*xfade.
    N==1 returns the single input passed through with no filter."""
    n = len(durations)
    if n == 0:
        raise ValueError("need at least one clip")
    if n == 1:
        return "", "[0:v]", float(durations[0])
    xf = float(xfade)
    parts = []
    prev = "[0:v]"
    cum = float(durations[0])          # length of composite built so far
    for i in range(1, n):
        offset = cum - xf
        out = f"[x{i}]" if i < n - 1 else "[vout]"
        parts.append(f"{prev}[{i}:v]xfade=transition={transition}:"
                     f"duration={xf:g}:offset={offset:g}{out}")
        cum = cum + float(durations[i]) - xf
        prev = out
    return ";".join(parts), "[vout]", cum


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def build_slideshow(cfg, images, dst, hold=7.0, xfade=2.5, zoom=1.08,
                    height=2160, workdir=None, kenburns=True, cancel=None,
                    log=None):
    """Build one seamless-looping slideshow from prepared/unprepared stills.

    images   -- list of source still paths (any resolution/aspect).
    hold     -- seconds each still is on screen at full frame (excl. dissolve).
    xfade    -- dissolve seconds between consecutive stills.
    zoom     -- Ken Burns magnification (ignored if kenburns=False).
    Returns (dst_path, total_seconds) on success, else (None, 0.0).
    """
    def _log(m):
        if log:
            log(m)

    if not images:
        return None, 0.0
    workdir = workdir or os.path.join(cfg.stage_dir("slideshow"), "_work")
    os.makedirs(workdir, exist_ok=True)
    fps = int(cfg.fps)
    seg = float(hold) + float(xfade)          # real Ken Burns clip length
    clips, durations = [], []

    for i, src in enumerate(images):
        if cancel and cancel():
            return None, 0.0
        prepped = os.path.join(workdir, f"prep_{i:03d}.png")
        if not prep_still(cfg, src, prepped, height):
            _log(f"prep failed: {src}")
            return None, 0.0
        clip = os.path.join(workdir, f"clip_{i:03d}.mp4")
        if kenburns:
            mode = "in" if i % 2 == 0 else "out"
            pan = 1.0 if i % 4 in (0, 3) else -1.0
            ok = kenburns_clip(cfg, prepped, clip, seg, zoom=zoom, mode=mode,
                               pan=pan, height=height, fps=fps, cancel=cancel)
        else:
            ok = static_clip(cfg, prepped, clip, seg, height=height, fps=fps,
                             cancel=cancel)
        if not ok:
            _log(f"kenburns failed: {src}")
            return None, 0.0
        clips.append(clip)
        durations.append(seg)
        _log(f"clip {i + 1}/{len(images)} rendered")

    # identical-frame wrap: a short static hold of clip-0's FIRST frame, so the
    # chain's last frame == its first frame -> -stream_loop is seamless.
    wrap_png = os.path.join(workdir, "wrap_first.png")
    wrap_clip = os.path.join(workdir, "clip_wrap.mp4")
    wrap_hold = float(xfade) + 0.8
    if not first_frame(cfg, clips[0], wrap_png):
        _log("wrap frame extract failed")
        return None, 0.0
    if not static_clip(cfg, wrap_png, wrap_clip, wrap_hold, height=height,
                       fps=fps, cancel=cancel):
        _log("wrap clip failed")
        return None, 0.0
    clips.append(wrap_clip)
    durations.append(wrap_hold)

    graph, out_label, total = xfade_chain_graph(durations, xfade)
    inputs = []
    for c in clips:
        inputs += ["-i", c]
    fc = ["-filter_complex", graph, "-map", out_label] if graph \
        else ["-map", out_label]
    args = inputs + fc + ["-r", str(fps)] + _codec(cfg, cfg.final_cq) \
        + ["-movflags", "+faststart", dst]
    if not _run(cfg, args, cancel=cancel) or not os.path.exists(dst):
        _log("xfade assembly failed")
        return None, 0.0
    _log(f"slideshow built: {os.path.basename(dst)} ~{total:.1f}s")
    return dst, total
