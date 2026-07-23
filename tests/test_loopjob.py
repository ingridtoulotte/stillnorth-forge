"""loopjob: hash resolution guard (stdlib only, no ffmpeg/GPU).

The full build_loop_job is ffmpeg orchestration, exercised live. Here we only
guard that a bad still hash fails loudly (FileNotFoundError) BEFORE any render
time is spent -- the cheap protection against a typo in a 10-hash arc.

    python tests/test_loopjob.py
    pytest
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stillnorth.config import get_config  # noqa: E402
from stillnorth import loopjob            # noqa: E402

CFG = get_config()


def test_resolve_stills_raises_on_missing():
    try:
        loopjob.resolve_stills(CFG, ["definitely_not_a_real_hash_zzz"])
    except FileNotFoundError as e:
        assert "definitely_not_a_real_hash_zzz" in str(e)
        return
    assert False, "missing still hash should raise FileNotFoundError"


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
