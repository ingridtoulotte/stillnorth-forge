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


def test_speed_match_config_loads():
    c = Config()
    assert isinstance(c.continuation_speed_match, bool)


def test_join_graph_plain_when_no_retime():
    # retime None or ~1.0 -> simple colour-correct + xfade, no split/retime nodes
    for r in (None, 1.0, 1.0005):
        g = finish._join_graph("CORR", ov=17, n1=81, fps=16, retime=r)
        assert "CORR" in g and "xfade=transition=fade" in g
        assert "split" not in g and "trim=" not in g
        assert "offset=4.0000" in g and "duration=1.0625" in g  # (81-17)/16, 17/16


def test_join_graph_retimes_only_new_frames():
    g = finish._join_graph("CORR", ov=17, n1=81, fps=16, retime=1.300)
    # overlap kept at native rate; ONLY the post-overlap frames time-stretched by 1/F
    assert "trim=end_frame=17" in g          # the overlap segment
    assert "trim=start_frame=17" in g        # the new segment
    assert "setpts=(PTS-STARTPTS)/1.300" in g
    assert "settb=AVTB" in g and "concat=n=2:v=1" in g
    assert "xfade=transition=fade:duration=1.0625:offset=4.0000" in g


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
