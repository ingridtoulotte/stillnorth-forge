"""Slideshow lane — pure math (no ffmpeg, no GPU, stdlib only).

Guards the seamless-loop construction: the xfade chain-graph offsets, total
duration, single-clip passthrough, and the config block. The actual render is
ffmpeg and is exercised live, not here.

    python tests/test_slideshow.py     # prints PASS/FAIL
    pytest
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stillnorth.config import get_config, STAGE_DIRS  # noqa: E402
from stillnorth import slideshow                        # noqa: E402

CFG = get_config()


def test_dims_are_16_9_and_even():
    assert slideshow._dims(2160) == (3840, 2160)
    assert slideshow._dims(1080) == (1920, 1080)
    w, h = slideshow._dims(721)
    assert w % 2 == 0 and h % 2 == 0            # ffmpeg yuv420p needs even dims


def test_single_clip_is_passthrough():
    g, lbl, tot = slideshow.xfade_chain_graph([8.0], 2.5)
    assert g == "" and lbl == "[0:v]" and tot == 8.0


def test_equal_clips_offsets_and_total():
    durs, xf = [9.5, 9.5, 9.5, 9.5], 2.5
    g, lbl, tot = slideshow.xfade_chain_graph(durs, xf)
    assert lbl == "[vout]"
    # total = sum(durations) - (N-1)*xfade
    assert abs(tot - (sum(durs) - 3 * xf)) < 1e-9
    # offsets are cumulative composite length minus one xfade: 7, 14, 21
    for off in ("offset=7", "offset=14", "offset=21"):
        assert off in g, (off, g)
    assert g.count("xfade") == 3
    assert g.count("transition=fade") == 3


def test_uneven_durations_offsets_track_cumulative():
    # a short wrap clip at the end must not break the offset math
    durs, xf = [9.5, 9.5, 3.3], 2.5
    g, _, tot = slideshow.xfade_chain_graph(durs, xf)
    assert abs(tot - (sum(durs) - 2 * xf)) < 1e-9
    assert "offset=7" in g and "offset=14" in g


def test_empty_raises():
    try:
        slideshow.xfade_chain_graph([], 2.5)
    except ValueError:
        return
    assert False, "empty durations should raise ValueError"


def test_config_slideshow_block_loads():
    c = CFG
    assert c.ss_hold > 0 and c.ss_xfade > 0
    assert 1.0 <= c.ss_zoom <= 1.5
    assert c.ss_height >= 720
    assert isinstance(c.ss_tiers, dict) and c.ss_tiers
    assert c.ss_audio_kind
    assert "slideshow" in STAGE_DIRS
    assert STAGE_DIRS["slideshow"] == "12_slideshows"


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t(); print(f"  PASS {t.__name__}"); passed += 1
        except Exception as e:
            print(f"  FAIL {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
