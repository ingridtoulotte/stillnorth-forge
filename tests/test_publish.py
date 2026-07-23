"""Publish lane — pure math (no ffmpeg, no GPU, stdlib only).

Guards the duration-tier loop count, the procedural audio-bed construction, and
the tier defaults. Actual muxing/looping is ffmpeg and is exercised live.

    python tests/test_publish.py     # prints PASS/FAIL
    pytest
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stillnorth import publish  # noqa: E402


def test_loop_reps_always_covers_target():
    for target, loop in [(60, 55), (1800, 55), (3600, 73.3), (30, 8), (60, 60)]:
        r = publish.loop_reps(target, loop)
        assert r >= 0
        # -stream_loop N plays N+1 times; that must cover the target length
        assert (r + 1) * loop >= target, (target, loop, r)


def test_loop_reps_zero_loop_is_safe():
    assert publish.loop_reps(60, 0) == 0
    assert publish.loop_reps(60, -1) == 0


def test_audio_presets_are_well_formed():
    for kind in ("wind", "rain", "drone", "still"):
        assert kind in publish.AUDIO_PRESETS
        color, amp, chain = publish.AUDIO_PRESETS[kind]
        assert isinstance(color, str) and color
        assert 0.0 < amp <= 1.0
        assert "volume=" in chain


def test_audio_input_builds_lavfi_source():
    inp, chain = publish._audio_input("wind", 60)
    assert inp[:2] == ["-f", "lavfi"]
    src = inp[3]
    assert src.startswith("anoisesrc=")
    assert "duration=60" in src
    assert "volume=" in chain


def test_audio_input_unknown_kind_falls_back_to_wind():
    _, chain = publish._audio_input("does-not-exist", 10)
    assert chain == publish.AUDIO_PRESETS["wind"][2]


def test_default_tiers():
    assert {"1min", "30min", "1h"} <= set(publish.DEFAULT_TIERS)
    assert publish.DEFAULT_TIERS["1min"] == 60
    assert publish.DEFAULT_TIERS["1h"] == 3600


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
