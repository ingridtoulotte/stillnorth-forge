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
    assert isinstance(c.esrgan_saturation_match, bool)
    assert isinstance(c.clip2_dedrift, bool)
    assert c.join_sharpen >= 0.0
    assert 0.0 <= c.edge_crop < 0.2


def test_join_graph_is_hard_cut_with_edge_crop():
    # straight concat (never a cross-dissolve) + border crop scaled back to full
    g = finish._join_graph(16, 0.8, 768, 448, 32, 16, 832, 480)
    assert "concat=n=2:v=1" in g
    assert "xfade" not in g                       # no ghosting/jump source
    assert "setpts=(PTS-STARTPTS)/" not in g      # no mid-stream retime (seam hiccup)
    assert "crop=768:448:32:16" in g              # drop hallucinated border
    assert "scale=832:480" in g                   # back to full frame
    assert "unsharp=5:5:0.80" in g                # clip2 sharpen


def test_join_graph_no_crop_when_full_frame():
    g = finish._join_graph(16, 0.8, 832, 480, 0, 0, 832, 480)
    assert "crop=" not in g and "concat=n=2:v=1" in g


def test_resample_speed_changes_frame_count():
    frames = list(range(100))
    assert finish._resample_speed(frames, 1.0) == frames        # no-op near 1
    faster = finish._resample_speed(frames, 1.25)               # speed up -> fewer
    assert 70 <= len(faster) <= 90 and faster[0] == 0
    slower = finish._resample_speed(frames, 0.8)                # slow down -> more
    assert len(slower) > 100


def test_even_rounds_down_to_even():
    assert finish._even(831) == 830 and finish._even(799.4) == 798


def test_dedrift_removes_brightness_trend():
    import numpy as np
    # synth clip2 that darkens linearly; de-drift should pin it back to ref
    ref = np.array([120.0, 120.0, 120.0])
    new = [np.full((8, 8, 3), 120.0 - i * 5.0) for i in range(10)]  # 120 -> 75
    out = finish._dedrift_clip2([f.copy() for f in new], ref)
    means = [float(f.mean()) for f in out]
    assert max(means) - min(means) < 6.0          # drift flattened
    assert abs(float(np.mean(means)) - 120.0) < 4.0  # pinned near clip1 ref


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


def test_match_seam_sharpness_blurs_grainy_clip2():
    import numpy as np
    rng = np.random.default_rng(0)
    grainy = [np.full((40, 40, 3), 120.0) + rng.normal(0, 25, (40, 40, 3)) for _ in range(8)]
    box = (0, 0, 40, 40)
    ref_sharp = 50.0   # clip1's tail is much smoother than this noisy clip2
    out = finish._match_seam_sharpness([f.copy() for f in grainy], ref_sharp, box)
    assert finish._sharp_metric(out, box) < finish._sharp_metric(grainy, box)


def test_match_seam_sharpness_noop_when_already_matched():
    import numpy as np
    flat = [np.full((20, 20, 3), 100.0) for _ in range(4)]
    out = finish._match_seam_sharpness([f.copy() for f in flat], 0.0, (0, 0, 20, 20))
    assert len(out) == len(flat)


def test_join_graph_sharpens_both_sides_evenly():
    # old bug: unsharp only on clip2 [b], amplifying its grain mismatch vs clip1
    g = finish._join_graph(16, 0.8, 832, 480, 0, 0, 832, 480)
    assert g.count("unsharp=5:5:0.80") == 2


def test_match_seam_contrast_pulls_punchy_clip2_down():
    import numpy as np
    rng = np.random.default_rng(1)
    base = 110.0 + rng.normal(0, 35, (30, 30, 3))   # punchy: bigger spread than ref
    punchy = [np.clip(base + rng.normal(0, 3, (30, 30, 3)), 0, 255) for _ in range(8)]
    box = (0, 0, 30, 30)
    ref_con, ref_sat = finish._contrast_sat(punchy, box)
    softer = [f * 0.6 + 50 for f in punchy]          # the "clip1 tail" reference
    soft_con, soft_sat = finish._contrast_sat(softer, box)
    out = finish._match_seam_contrast([f.copy() for f in punchy], soft_con, soft_sat, box)
    out_con, out_sat = finish._contrast_sat(out, box)
    assert out_con < ref_con


def test_match_seam_contrast_noop_when_matched():
    import numpy as np
    flat = [np.full((20, 20, 3), 100.0) for _ in range(4)]
    out = finish._match_seam_contrast([f.copy() for f in flat], 0.0, 0.0, (0, 0, 20, 20))
    assert len(out) == len(flat)


def test_ramp_toward_eases_from_orig_to_corrected():
    import numpy as np
    orig = [np.full((4, 4, 3), 100.0) for _ in range(5)]
    corrected = [np.full((4, 4, 3), 200.0) for _ in range(5)]
    out = finish._ramp_toward(orig, corrected)
    means = [float(f.mean()) for f in out]
    assert means[0] == 100.0           # first frame untouched
    assert means[-1] == 200.0          # last frame fully corrected
    assert means[0] < means[2] < means[-1]   # monotonic ease, no jump


def test_seam_blend_alpha_config_default():
    c = Config()
    assert 0.0 <= c.seam_blend_alpha <= 1.0


def test_esrgan_available_false_when_missing(tmp_path):
    c = SimpleNamespace(esrgan_bin=str(tmp_path / "nope.exe"),
                        esrgan_models_dir=str(tmp_path))
    assert finish.esrgan_available(c) is False
    assert finish.esrgan_available(
        SimpleNamespace(esrgan_bin="", esrgan_models_dir="")) is False


def test_new_seam_config_keys():
    c = Config()
    assert c.seam_sharp_window >= 0
    assert isinstance(c.body_sharpen, bool)
    assert c.speed_clamp_hi >= 1.0
    assert c.continuation_drop_camera is False    # camera-kept is the default


def test_affine_match_hits_target_without_double_desat():
    import numpy as np
    rng = np.random.default_rng(7)
    # punchy clip2 head: higher mean and higher spread than the target
    frames = [rng.normal(140, 40, (48, 64, 3)).clip(0, 255) for _ in range(6)]
    box = (0, 0, 64, 48)
    cur_m, cur_sd = finish._box_stats(frames, box)
    T_m, T_sd = np.array([120.0] * 3), np.array([25.0] * 3)
    out = finish._affine_match(frames, cur_m, cur_sd, T_m, T_sd)
    out_m, out_sd = finish._box_stats(out, box)
    assert np.allclose(out_m, T_m, atol=1.5)      # one pass lands ON target
    assert np.allclose(out_sd, T_sd, atol=1.5)    # std matched, not overshot


def test_windowed_sharp_leaves_body_untouched():
    import numpy as np
    rng = np.random.default_rng(3)
    frames = [rng.normal(128, 30, (48, 64, 3)).clip(0, 255) for _ in range(40)]
    body_before = [f.copy() for f in frames[16:]]
    out = finish._windowed_sharp(frames, T_sharp=1.0, box=(0, 0, 64, 48), W=16)
    assert len(out) == 40
    for a, b in zip(out[16:], body_before):       # frames past W: bit-identical
        assert np.array_equal(a, b)


def test_body_sharpen_noop_when_already_sharp():
    import numpy as np
    rng = np.random.default_rng(5)
    sharp = [rng.normal(128, 50, (48, 64, 3)).clip(0, 255) for _ in range(40)]
    out = finish._body_sharpen([f.copy() for f in sharp], sharp, (0, 0, 64, 48))
    for a, b in zip(out, sharp):                  # same texture level -> no-op
        assert np.array_equal(a, b)


def test_retime_mc_config_keys():
    c = Config()
    assert isinstance(c.speed_retime_mc, bool)
    assert isinstance(c.band_dedrift, bool)


def test_dedrift_bands_fixes_divergent_drift():
    import numpy as np
    rng = np.random.default_rng(11)
    H, W = 48, 64
    ref = [rng.normal(120, 20, (H, W, 3)).clip(0, 255) for _ in range(10)]
    new = []
    for i in range(24):
        f = rng.normal(120, 20, (H, W, 3)).clip(0, 255)
        f[:H // 3] += i * 0.5          # sky drifts lighter
        f[2 * H // 3:] -= i * 0.6      # ground drifts darker
        new.append(f.clip(0, 255))
    out = finish._dedrift_bands([f.copy() for f in new], ref, (0, 0, W, H))
    sky_end = out[-1][:H // 3].mean()
    gnd_end = out[-1][2 * H // 3:].mean()
    sky_ref = np.mean([f[:H // 3].mean() for f in ref[-8:]])
    gnd_ref = np.mean([f[2 * H // 3:].mean() for f in ref[-8:]])
    assert abs(sky_end - sky_ref) < 3.0    # divergent drifts both pinned
    assert abs(gnd_end - gnd_ref) < 3.0
