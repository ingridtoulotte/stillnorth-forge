"""Dive lane — pure logic (no ffmpeg, no GPU, no Ollama, no depthflow).

Guards the DepthFlow-subprocess arg construction, the coherency-gate short
circuit, availability check, and the config `dive` block. The actual render is a
subprocess to the external DepthFlow venv and is exercised live, not here.

    python tests/test_dive.py     # prints PASS/FAIL
    pytest
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stillnorth.config import get_config      # noqa: E402
from stillnorth import dive                    # noqa: E402

CFG = get_config()


def _stub(**kw):
    base = dict(dive_venv_python="/no/python", dive_render_script="/no/script.py",
                fps=16, dive_ssaa=2.0, dive_dolly=1.0, dive_parallax=0.6,
                dive_judge_coherency=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_dive_cmd_builds_expected_args():
    cmd = dive.dive_cmd(_stub(), "src.png", "out.mp4", 9.5, 3840, 2160)
    assert cmd[:6] == ["/no/python", "/no/script.py", "--image", "src.png",
                       "--out", "out.mp4"], cmd
    # flag/value pairs present and correctly typed-to-string
    for flag, val in (("--time", "9.500"), ("--width", "3840"),
                      ("--height", "2160"), ("--fps", "16"), ("--ssaa", "2.0"),
                      ("--dolly", "1.0"), ("--parallax", "0.6")):
        assert flag in cmd, (flag, cmd)
        assert cmd[cmd.index(flag) + 1] == val, (flag, cmd[cmd.index(flag) + 1])


def test_coherency_off_short_circuits():
    # judging disabled must return True without importing judge / hitting Ollama
    assert dive.coherency_ok(_stub(dive_judge_coherency=False), "x.png") is True


def test_dive_available_false_when_paths_missing():
    assert dive.dive_available(_stub()) is False


def test_dive_shot_returns_none_on_coherency_reject():
    # a coherency-rejected still must return None (caller SKIPS it) -- not False
    # (which aborts the build) and without touching any render stage.
    orig = dive.coherency_ok
    dive.coherency_ok = lambda cfg, still, log=None: False
    try:
        r = dive.dive_shot(_stub(), "s.png", "d.mp4", 9.5, height=2160)
    finally:
        dive.coherency_ok = orig
    assert r is None, r


def test_config_dive_block_loads():
    c = CFG
    assert isinstance(c.dive_enabled, bool)
    assert c.dive_upscaler == "high-fidelity-4x"      # shootout winner (2026-07-22)
    assert 0.0 < c.dive_dolly <= 5.0
    assert 0.0 < c.dive_parallax <= 1.0
    assert c.dive_ssaa >= 1.0
    assert isinstance(c.dive_judge_coherency, bool)
    # render script defaults into the package next to dive.py
    assert c.dive_render_script.endswith(os.path.join("stillnorth", "dive_render.py"))


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
