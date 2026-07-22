"""One-call orchestration of the slideshow loop-publishing pipeline (§5E).

Ties slideshow.py (build the seamless base loop) + publish.py (audio bed,
duration tiers, Shorts) into a single job, used by both the `loop` CLI command
and the web UI. Given a list of source-still hashes it produces, in
`12_slideshows/`: the base loop, one file per duration tier (with an ambient
bed), and optionally a 9:16 Shorts cut. Returns a result dict of paths.
"""
import os

from . import slideshow, publish


def resolve_stills(cfg, hashes):
    """Map classified-still hashes (with or without .png) to file paths.

    Raises FileNotFoundError naming the first missing hash so a typo fails
    loudly before any render time is spent."""
    d = cfg.stage_dir("classified", ensure=False)
    out = []
    for h in hashes:
        name = h if h.lower().endswith(".png") else h + ".png"
        p = os.path.join(d, name)
        if not os.path.exists(p):
            raise FileNotFoundError(f"source still not found: {name} (in {d})")
        out.append(p)
    return out


def build_loop_job(cfg, hashes, name="loop", audio_kind=None, tiers=None,
                   make_shorts=True, hold=None, xfade=None, zoom=None,
                   kenburns=True, log=None, cancel=None):
    """Build a seamless loop from `hashes` and publish tiers + Shorts.

    Returns {"base","duration","tiers":{key:path},"short"} on success, or None
    if the base render failed / was cancelled. Recipe knobs fall back to the
    config `slideshow` block when not given."""
    imgs = resolve_stills(cfg, hashes)
    out_dir = cfg.stage_dir("slideshow")
    base = os.path.join(out_dir, f"{name}.mp4")
    work = os.path.join(out_dir, f"_work_{name}")
    b, total = slideshow.build_slideshow(
        cfg, imgs, base,
        hold=cfg.ss_hold if hold is None else hold,
        xfade=cfg.ss_xfade if xfade is None else xfade,
        zoom=cfg.ss_zoom if zoom is None else zoom,
        height=cfg.ss_height, workdir=work, kenburns=kenburns,
        log=log, cancel=cancel)
    if not b:
        return None
    kind = audio_kind or cfg.ss_audio_kind
    tiers = tiers or cfg.ss_tiers
    tier_paths = publish.export_tiers(cfg, base, out_dir, tiers=tiers,
                                      audio_kind=kind, stem=name,
                                      cancel=cancel, log=log)
    short = None
    if make_shorts and not (cancel and cancel()):
        sp = os.path.join(out_dir, f"{name}_short.mp4")
        if publish.shorts_crop(cfg, base, sp, seconds=20, cancel=cancel):
            short = sp
    return {"base": base, "duration": total, "tiers": tier_paths, "short": short}
