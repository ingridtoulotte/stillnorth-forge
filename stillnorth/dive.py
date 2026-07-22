"""DepthFlow 'dive' motion for the loop lane -- an OPTIONAL alternative to Ken Burns.

Evaluated in docs/spikes/2026-07-22-dive-motion-spike.md: DepthFlow displaces the
SOURCE pixels along an estimated depth map (no diffusion, no new content), giving
a "diving into the scene" parallax that a flat zoompan cannot. The default motion
stays kenburns; dive is opt-in per build.

DepthFlow is AGPL, so it is invoked as a CLI SUBPROCESS in its own venv (config
`dive.venv_python`) running dive_render.py -- never imported -- keeping this repo
clear of AGPL.

Per shot (`dive_shot`): coherency-judge the still (BEFORE spending upscale/GPU) ->
upscale the still ONCE to 4K with the chosen upscaler -> DepthFlow dive at 4K from
the sharp texture -> the pipeline's final_grade + final_unsharp (which erases
DepthFlow's mild depth-warp softness). One upscale per shot, not per frame.
"""
import os
import subprocess

from . import slideshow


# --------------------------------------------------------------------------- #
def dive_available(cfg):
    """True when the external DepthFlow venv + render script are present."""
    py = cfg.dive_venv_python
    return bool(py) and os.path.exists(py) and os.path.exists(cfg.dive_render_script)


def dive_cmd(cfg, still, dst, seconds, width, height):
    """The DepthFlow subprocess arg list -- pure, unit-testable."""
    return [cfg.dive_venv_python, cfg.dive_render_script,
            "--image", still, "--out", dst,
            "--time", f"{float(seconds):.3f}",
            "--width", str(int(width)), "--height", str(int(height)),
            "--fps", str(cfg.fps), "--ssaa", str(cfg.dive_ssaa),
            "--dolly", str(cfg.dive_dolly), "--parallax", str(cfg.dive_parallax)]


def _render_dive(cfg, still, dst, seconds, width, height):
    env = dict(os.environ)
    if cfg.dive_hf_home:
        env["HF_HOME"] = cfg.dive_hf_home
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    try:
        r = subprocess.run(dive_cmd(cfg, still, dst, seconds, width, height),
                           env=env, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
    except OSError:                       # venv python missing/misconfigured
        return False
    return r.returncode == 0 and os.path.exists(dst)


def _upscale_still(cfg, src, dst, w, h):
    """Upscale a still to WxH once (cover-scale + crop, so ANY source aspect works
    -- mirrors slideshow.prep_still). `cfg.dive_upscaler`:
      'lanczos'  -> ffmpeg lanczos only (honest, no fabrication).
      <model>    -> Upscayl/ESRGAN 4x (that model) then lanczos-downscale
                    (supersampled). Falls back to lanczos if the bin/model is
                    missing or the call fails."""
    model = (cfg.dive_upscaler or "lanczos").strip()
    scale = (f"scale={int(w)}:{int(h)}:flags=lanczos:"
             f"force_original_aspect_ratio=increase,crop={int(w)}:{int(h)}")
    if model == "lanczos" or not (cfg.esrgan_bin and os.path.exists(cfg.esrgan_bin)):
        return slideshow._run(cfg, ["-i", src, "-vf", scale, dst]) and os.path.exists(dst)
    tmp = dst + ".4x.png"
    try:
        rc = subprocess.run(
            [cfg.esrgan_bin, "-i", src, "-o", tmp, "-n", model,
             "-m", cfg.esrgan_models_dir, "-s", "4", "-f", "png"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    except OSError:
        rc = 1
    try:
        up = tmp if (rc == 0 and os.path.exists(tmp)) else src   # esrgan out, else fall back
        return slideshow._run(cfg, ["-i", up, "-vf", scale, dst]) and os.path.exists(dst)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def coherency_ok(cfg, still, log=None):
    """Ollama coherency review of the SOURCE STILL, before any upscale/GPU spend.
    Reuses the project's image judge. Never blocks on infra: if coherency judging
    is disabled, Ollama is down, or anything errors, it passes (True)."""
    if not cfg.dive_judge_coherency:
        return True
    try:
        from . import judge
        if not judge.available(cfg):
            return True
        ok, reason = judge.judge_image(cfg, still)
        if not ok and log:
            log(f"dive: still failed coherency judge -> skipped ({reason})")
        return ok
    except Exception:
        return True


def dive_shot(cfg, still, dst, seconds, height=2160, cancel=None, log=None):
    """Render ONE dive shot from a source still.

    Returns:
      True  -- rendered OK (dst written).
      None  -- the coherency judge REJECTED the still; the caller should SKIP it
               (drop from the loop), not abort the whole build.
      False -- a hard stage failure (upscale / render / encode)."""
    w, h = slideshow._dims(height)
    if not coherency_ok(cfg, still, log=log):
        return None
    work = os.path.dirname(dst) or "."
    stem = os.path.splitext(os.path.basename(dst))[0]
    up = os.path.join(work, f"_diveup_{stem}.png")
    nat = os.path.join(work, f"_divenat_{stem}.mp4")
    try:
        if cancel and cancel():
            return False
        if not _upscale_still(cfg, still, up, w, h):
            if log:
                log(f"dive: upscale failed for {os.path.basename(still)}")
            return False
        if not _render_dive(cfg, up, nat, seconds, w, h):
            if log:
                log(f"dive: DepthFlow render failed for {os.path.basename(still)}")
            return False
        grade = (cfg.final_grade or "").strip()
        chain = ([grade] if grade else []) + \
            [f"unsharp={cfg.final_unsharp}", "format=yuv420p", "setsar=1"]
        ok = slideshow._run(
            cfg, ["-i", nat, "-vf", ",".join(chain)]
            + slideshow._codec(cfg, cfg.final_cq) + ["-movflags", "+faststart", dst],
            cancel=cancel) and os.path.exists(dst)
        return ok
    finally:
        for t in (up, nat):
            try:
                os.remove(t)
            except OSError:
                pass
