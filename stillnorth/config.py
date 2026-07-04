"""Configuration loader for StillNorth Forge.

All tunables live in the JSON files under config/ so the pipeline can be
re-pointed at a different machine, ComfyUI install or workflow export without
touching Python. This module loads them, resolves Windows paths and exposes
the per-stage working directories.
"""
import json
import os

# repo root = parent of this package
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(ROOT, "config")

# Ordered pipeline stages and the folder each one writes into (under the
# ComfyUI output workspace). Order matters: the orchestrator walks it top down.
STAGE_DIRS = {
    "flux":      "01_flux",            # FLUX.2 images
    "img_up":    "02_flux_up2",        # images upscaled x2
    "classified":"03_classified",      # x2 images renamed <letter>_<key>.png
    "vid1":      "04_vid1",            # first Wan 2.2 clips
    "lastframe": "05_lastframe",       # last frame of each first clip
    "lf_up":     "06_lastframe_up4",   # last frames upscaled x4
    "vid2":      "07_vid2",            # continuation Wan 2.2 clips
    "concat":    "08_concat",          # clip1 + clip2 joined (~10-11s)
    "final_up":  "09_final_up4",       # final clips upscaled x4
}


def _load(name):
    with open(os.path.join(CONFIG_DIR, name), "r", encoding="utf-8") as fh:
        return json.load(fh)


class Config:
    def __init__(self):
        self.reload()

    def reload(self):
        self.raw = _load("config.json")
        self.workflows = _load("workflows.json")
        self.motion = _load("motion_prompts.json")

        self.comfy_server = self.raw["comfy_server"]
        self.ffmpeg = os.path.normpath(self.raw["ffmpeg"])
        self.ffprobe = self._sibling_tool(self.ffmpeg, "ffprobe")
        self.comfy_input = os.path.normpath(self.raw["comfy_input_dir"])
        self.comfy_output = os.path.normpath(self.raw["comfy_output_dir"])
        self.workspace_subdir = self.raw["workspace_subdir"]
        self.workspace = os.path.join(self.comfy_output, self.workspace_subdir)

        self.host = self.raw["server_host"]
        self.port = int(self.raw["server_port"])

        self.timeout_img = int(self.raw["render_timeout_img"])
        self.timeout_vid = int(self.raw["render_timeout_vid"])
        self.poll = float(self.raw["poll_seconds"])
        self.submit_retries = int(self.raw.get("submit_retries", 2))
        self.retry_backoff = float(self.raw.get("retry_backoff_seconds", 5))

        self.img_mult = int(self.raw["image_upscale_mult"])
        self.lf_mult = int(self.raw["lastframe_upscale_mult"])
        self.final_mult = int(self.raw["final_upscale_mult"])
        # how the continuation clip is seeded: "native" feeds Wan the crisp
        # native-resolution last frame (proven seamless recipe); "upscaled"
        # feeds the x4 lanczos lastframe (round-trips a blur -> softer clip 2).
        self.cont_seed = str(self.raw.get("continuation_seed", "native")).lower()
        # native_long: render the whole ~10s clip in ONE Wan pass instead of
        # clip1 + a softer continuation clip2. No seam, no quality step -- the
        # proven "11s natif" fix. Falls back to continuation if disabled.
        self.native_long = bool(self.raw.get("native_long", False))
        self.native_frames = int(self.raw.get("native_long_frames", 161))
        # continuation polish: drop the first few frames of clip 1 (img2vid
        # start glitch) and sharpen clip 2 to match clip 1 (the continuation
        # clip drifts slightly soft). Empty clip2_sharpen disables sharpening.
        self.trim_start = max(0, int(self.raw.get("trim_start_frames", 3)))
        self.clip2_sharpen = str(self.raw.get("clip2_sharpen", "")).strip()
        # clip2_color_match: measure the brightness/chroma the continuation clip
        # drifts to (Wan re-exposes when seeded from a single still) and pull
        # clip 2's head back onto clip 1's tail so the cut has no tonal step.
        # This is the real seam fix; a one-sided unsharp only hid softness and
        # actually popped contrast at the join.
        self.clip2_color_match = bool(self.raw.get("clip2_color_match", True))
        # native_overlap (default): seed clip 2 with clip 1's LAST `overlap_frames`
        # real frames (camera embedding dropped) so Wan continues the actual motion
        # -> no surge/color/speed/blur seam by construction. "single_frame" = old
        # one-still i2v continuation (kept as fallback).
        self.continuation_mode = str(
            self.raw.get("continuation_mode", "native_overlap")).lower()
        self.overlap_frames = max(1, int(self.raw.get("overlap_frames", 17)))
        self.continuation_length = int(self.raw.get("continuation_length", 97))
        self.continuation_drop_camera = bool(
            self.raw.get("continuation_drop_camera", True))
        # continuation_speed_match: Wan's continuation drifts slightly slower than
        # clip 1 (flow ratio ~0.73-0.95). Measure both halves' Farneback optical
        # flow and time-stretch only the NEW continuation frames so clip2's motion
        # speed matches clip1 -> no speed step at the seam. Frame-diff would be
        # confounded by sharpness; optical flow is not.
        self.continuation_speed_match = bool(
            self.raw.get("continuation_speed_match", True))
        # join_sharpen: unsharp amount applied to clip2 at the join (the
        # continuation drifts slightly soft vs clip1). The join is now a hard cut
        # (no xfade), so this is the only seam-side sharpening.
        self.join_sharpen = float(self.raw.get("join_sharpen", 0.8))
        # clip2_dedrift: remove the continuation's per-channel brightness/colour
        # drift (Wan i2v darkens over its length) by fitting each channel's trend
        # and pinning it to clip1's tail colour -> clip2 no longer ends darker.
        self.clip2_dedrift = bool(self.raw.get("clip2_dedrift", True))
        # edge_crop: fraction trimmed off EACH side then scaled back, to drop the
        # soft/grainy hallucinated border a pan reveals at the frame edges.
        self.edge_crop = float(self.raw.get("edge_crop", 0.04))
        # seam_blend_alpha: how far the contrast/sat/sharp seam TARGET sits from
        # clip1 (0.0) toward clip2 (1.0). 1.0 = old behaviour (clip2 pulled all
        # the way to clip1, clip1 untouched -- reads as one half "fixed" and one
        # not). <1.0 meets in the middle: clip2 gets a smaller correction and
        # clip1's own last frames ease toward the same shared target.
        self.seam_blend_alpha = float(self.raw.get("seam_blend_alpha", 0.35))
        # seam_sharp_window: frames of clip2 whose texture is eased toward the
        # seam target (ramped). 0 = off. Full-clip matching is gone -- it was
        # blurring the entire continuation down to clip1's softest frames.
        self.seam_sharp_window = int(self.raw.get("seam_sharp_window", 16))
        # body_sharpen: one binary-searched unsharp bringing clip2's body
        # texture up to clip1's body level (camera-kept renders slightly soft).
        self.body_sharpen = bool(self.raw.get("body_sharpen", True))
        # speed_clamp_hi: upper clamp for the seam speed resample factor.
        self.speed_clamp_hi = float(self.raw.get("speed_clamp_hi", 1.8))
        # speed_retime_mc: motion-compensated retime (setpts + minterpolate)
        # instead of nearest-frame picking, which stuttered (periodic double
        # steps read as objects jumping every few frames).
        self.speed_retime_mc = bool(self.raw.get("speed_retime_mc", True))
        # band_dedrift: per-band (sky/mid/ground) colour drift removal after
        # the global affine -- bands drift in OPPOSITE directions on some
        # scenes, which a global correction cannot hold.
        self.band_dedrift = bool(self.raw.get("band_dedrift", True))
        # seam_sharp_alpha_up: sharpness target position when the continuation
        # is SHARPER than clip1's diffused tail (equalise up, not to the blur).
        self.seam_sharp_alpha_up = float(self.raw.get("seam_sharp_alpha_up", 1.0))
        # tail_ramp_pow: ramp exponent for the tail sharpen-up (<1 = correction
        # reaches most of the tail, not just the last frames).
        self.tail_ramp_pow = float(self.raw.get("tail_ramp_pow", 0.5))
        # Wan render resolution, injected into the camera-embedding node at
        # submit time (overrides whatever the workflow export carries).
        # 1104x624 = 1.72x the pixels of 832x480; measured ~1.8x render time,
        # visibly more real detail after the x4 super-res -- the blur ceiling
        # was the 480p base, not the upscaler. 0 = leave workflow untouched.
        self.wan_width = int(self.raw.get("wan_width", 0))
        self.wan_height = int(self.raw.get("wan_height", 0))
        self.fps = int(self.raw["fps"])
        self.cq = int(self.raw["video_cq"])
        self.nvenc = bool(self.raw["nvenc"])
        self.denoise = self.raw["upscale_denoise"]
        self.sharp = self.raw["upscale_sharp"]
        self.grain = self.raw["upscale_grain"]

        # final finisher: "esrgan" = real-detail Upscayl/Real-ESRGAN per-frame +
        # progressive contrast/saturation de-drift + UHD crisp; "lanczos" = the
        # cheap ffmpeg chain (fallback / no-GPU-upscaler box).
        self.final_upscaler = str(self.raw.get("final_upscaler", "esrgan")).lower()
        self.esrgan_bin = self.raw.get("esrgan_bin", "")
        self.esrgan_models_dir = self.raw.get("esrgan_models_dir", "")
        self.esrgan_model = self.raw.get("esrgan_model", "remacri-4x")
        self.final_height = int(self.raw.get("final_height", 2160))
        self.final_unsharp = self.raw.get("final_unsharp", "5:5:0.6:5:5:0.0")
        self.final_grain = self.raw.get("final_grain", "alls=4:allf=t+u")
        self.final_tdenoise = self.raw.get("final_temporal_denoise", "0:0:6:6")
        self.final_cq = int(self.raw.get("final_cq", 17))
        # final_fps: motion-interpolate masters to this rate before super-res
        # (0 = keep the native Wan rate). 32 kills the 16fps judder on TVs.
        self.final_fps = int(self.raw.get("final_fps", 0))
        self.contrast_flatten = bool(self.raw.get("contrast_flatten", True))
        self.contrast_boost = float(self.raw.get("contrast_target_boost", 1.0))
        self.saturation_boost = float(self.raw.get("saturation_target_boost", 1.0))
        # esrgan_color_match: after super-res, pull the upscaled colour
        # distribution back onto the source clip (remacri-4x over-punches
        # contrast/saturation -> "neon"; this is what made only the 4k look off).
        self.esrgan_color_match = bool(self.raw.get("esrgan_color_match", True))
        # esrgan_saturation_match: per-channel mean/std match left a residual neon;
        # also pull HSV saturation back to the source clip's level after super-res.
        self.esrgan_saturation_match = bool(
            self.raw.get("esrgan_saturation_match", True))

        self.poses = self.motion["poses"]
        self.speed_by_pose = self.motion["speed_by_pose"]
        self.classes = self.motion["classes"]

        # long-video assembler (all optional, with safe defaults)
        a = self.raw.get("assembler", {})
        self.asm_fade = float(a.get("fade_seconds", 1.0))
        self.asm_height = int(a.get("target_height", 1080))
        self.asm_clip_len = float(a.get("avg_clip_seconds", 10.0))
        self.retention_days = int(a.get("retention_days", 7))
        self.autodelete = bool(a.get("autodelete_enabled", True))
        self.asm_default_weights = a.get(
            "default_weights", {"0": 60, "1": 20, "2": 10, "3": 6, "4": 4})

    @staticmethod
    def _sibling_tool(ffmpeg_path, name):
        """Path to a tool that ships beside ffmpeg (e.g. ffprobe), keeping the
        original extension so it resolves on Windows (.exe) and POSIX alike."""
        d, base = os.path.split(ffmpeg_path)
        ext = os.path.splitext(base)[1]
        return os.path.join(d, name + ext)

    # -- paths --------------------------------------------------------------
    def stage_dir(self, stage, ensure=True):
        """Absolute filesystem path for a stage's output folder."""
        d = os.path.join(self.workspace, STAGE_DIRS[stage])
        if ensure:
            os.makedirs(d, exist_ok=True)
        return d

    def comfy_prefix(self, stage, name):
        """filename_prefix for a ComfyUI Save node (relative to output dir,
        forward slashes -- ComfyUI is picky)."""
        return f"{self.workspace_subdir}/{STAGE_DIRS[stage]}/{name}"

    def workflow_path(self, which):
        return os.path.join(ROOT, self.workflows[which]["file"])

    def motion_text(self, letter):
        return self.classes[letter]["motion"]

    def state_path(self):
        os.makedirs(self.workspace, exist_ok=True)
        return os.path.join(self.workspace, "forge_state.json")

    def log_path(self):
        os.makedirs(self.workspace, exist_ok=True)
        return os.path.join(self.workspace, "forge.log")


_CFG = None


def get_config():
    global _CFG
    if _CFG is None:
        _CFG = Config()
    return _CFG
