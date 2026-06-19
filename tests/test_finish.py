"""Tests for the native-overlap continuation + ESRGAN/contrast finisher recipe."""
import os
from types import SimpleNamespace

from stillnorth.config import Config
from stillnorth import finish


def test_recipe_config_defaults():
    c = Config()
    assert c.continuation_mode in ("native_overlap", "single_frame")
    assert c.overlap_frames >= 1
    assert c.continuation_length >= c.overlap_frames
    assert isinstance(c.continuation_drop_camera, bool)
    assert c.final_upscaler in ("esrgan", "lanczos")
    assert c.final_height >= 720
    assert 1.0 <= c.contrast_boost <= 1.5
    assert 1.0 <= c.saturation_boost <= 1.5


def test_wan_workflow_has_continuation_nodes():
    wf = Config().workflows["wan"]
    for k in ("node_wci2v", "field_start_image", "field_camera_cond"):
        assert wf.get(k), f"workflows.json wan missing {k}"


def test_codec_switch():
    a = finish._codec(SimpleNamespace(nvenc=True), 17)
    assert "hevc_nvenc" in a and "17" in a
    b = finish._codec(SimpleNamespace(nvenc=False), 14)
    assert "libx264" in b and "14" in b


def test_esrgan_available_false_when_missing(tmp_path):
    c = SimpleNamespace(esrgan_bin=str(tmp_path / "nope.exe"),
                        esrgan_models_dir=str(tmp_path))
    assert finish.esrgan_available(c) is False
    assert finish.esrgan_available(
        SimpleNamespace(esrgan_bin="", esrgan_models_dir="")) is False
