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
        self.comfy_input = os.path.normpath(self.raw["comfy_input_dir"])
        self.comfy_output = os.path.normpath(self.raw["comfy_output_dir"])
        self.workspace_subdir = self.raw["workspace_subdir"]
        self.workspace = os.path.join(self.comfy_output, self.workspace_subdir)

        self.host = self.raw["server_host"]
        self.port = int(self.raw["server_port"])

        self.timeout_img = int(self.raw["render_timeout_img"])
        self.timeout_vid = int(self.raw["render_timeout_vid"])
        self.poll = float(self.raw["poll_seconds"])

        self.img_mult = int(self.raw["image_upscale_mult"])
        self.lf_mult = int(self.raw["lastframe_upscale_mult"])
        self.final_mult = int(self.raw["final_upscale_mult"])
        self.fps = int(self.raw["fps"])
        self.cq = int(self.raw["video_cq"])
        self.nvenc = bool(self.raw["nvenc"])
        self.denoise = self.raw["upscale_denoise"]
        self.sharp = self.raw["upscale_sharp"]
        self.grain = self.raw["upscale_grain"]

        self.poses = self.motion["poses"]
        self.speed_by_pose = self.motion["speed_by_pose"]
        self.classes = self.motion["classes"]

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
