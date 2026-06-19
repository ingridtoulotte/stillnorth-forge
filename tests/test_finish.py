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
    assert isinstance(c.continuation_speed_match, bool)
    assert c.final_upscaler in ("esrgan", "lanczos")
    assert c.final_height >= 720
    assert 1.0 <= c.contrast_boost <= 1.5
    assert 1.0 <= c.saturation_boost <= 1.5


def test_speed_match_and_join_config_loads():
    c = Config()
    assert isinstance(c.continuation_speed_match, bool)
    assert isinstance(c.esrgan_color_match, bool)
    assert c.join_sharpen >= 0.0


def test_join_graph_is_hard_cut_no_xfade():
    # join is a straight concat at the true cut frame, never a cross-dissolve
    for r in (None, 1.0, 1.0005):
        g = finish._join_graph("CORR", cut=18, fps=16, retime=r)
        assert "CORR" in g and "concat=n=2:v=1" in g
        assert "xfade" not in g                       # no ghosting/jump source
        assert "trim=start_frame=18" in g             # drop reproduced overlap
        assert "(PTS-STARTPTS)/" not in g             # no retime when ratio ~1


def test_join_graph_retimes_new_frames_when_ratio_off():
    g = finish._join_graph("CORR", cut=18, fps=16, retime=1.300)
    assert "trim=start_frame=18" in g
    assert "setpts=(PTS-STARTPTS)/1.300" in g
    assert "concat=n=2:v=1" in g and "xfade" not in g


def test_norm_lut_pulls_toward_source():
    # current (neon) std bigger than source -> gain < 1 (pulls contrast DOWN)
    lut = finish._norm_lut([100, 100, 100], [40, 40, 40],
                           [110, 110, 110], [60, 60, 60])
    assert lut.startswith("lutrgb=") and "r=clip(" in lut
    assert "*0.667" in lut          # 40/60
    assert "+100.00" in lut         # mapped back onto source mean


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
