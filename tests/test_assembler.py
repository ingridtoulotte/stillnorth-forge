"""Assembler + library + retention tests. No GPU, no ffmpeg, no ComfyUI, no deps.

Covers the parts that decide WHAT goes into a long video and HOW usage is
tracked -- the selection maths, the usage-bucket file moves, the one-week
retention selection, and the 'last done' preview pick. The ffmpeg-calling parts
(normalise / concat) are integration-only and not exercised here.

    python tests/test_assembler.py     # prints PASS/FAIL
    pytest
"""
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stillnorth.config import STAGE_DIRS                       # noqa: E402
from stillnorth import library as lib                          # noqa: E402
from stillnorth import assembler as asm                        # noqa: E402
from stillnorth import server                                  # noqa: E402


class _Cfg:
    """Minimal config stand-in pointing the library at a temp workspace."""
    def __init__(self, root):
        self.comfy_output = root
        self.workspace = os.path.join(root, "StillNorthForge")
        os.makedirs(self.workspace, exist_ok=True)

    def stage_dir(self, stage, ensure=True):
        d = os.path.join(self.workspace, STAGE_DIRS[stage])
        if ensure:
            os.makedirs(d, exist_ok=True)
        return d


def _make_master(cfg, level, stem):
    d = lib.bucket_dir(cfg, level, ensure=True)
    p = os.path.join(d, stem + ".mp4")
    with open(p, "w") as fh:
        fh.write("x")
    return p


# ---- pure planning -------------------------------------------------------
def test_clips_needed():
    assert asm.clips_needed(15 * 60, 10) == 90
    assert asm.clips_needed(12 * 3600, 10) == 4320
    assert asm.clips_needed(5, 10) == 1            # always at least one


def test_select_prefers_fresh_and_respects_count():
    pool = {0: [f"A_{i:02d}" for i in range(50)],
            1: [f"B_{i:02d}" for i in range(50)]}
    weights = {"0": 100, "1": 0, "2": 0, "3": 0, "4": 0}
    chosen = asm.select_distinct(pool, weights, 20)
    assert len(chosen) == 20
    assert all(s.startswith("A_") for s in chosen), "all-fresh weight ignored"


def test_select_falls_back_when_bucket_empty():
    # ask for used-once clips but only never-used exist -> must still fill
    pool = {0: [f"A_{i}" for i in range(10)], 1: []}
    weights = {"0": 0, "1": 100, "2": 0, "3": 0, "4": 0}
    chosen = asm.select_distinct(pool, weights, 6)
    assert len(chosen) == 6 and all(s.startswith("A_") for s in chosen)


def test_select_capped_by_availability():
    pool = {0: ["A_1", "A_2", "A_3"]}
    chosen = asm.select_distinct(pool, {"0": 100}, 99)
    assert sorted(chosen) == ["A_1", "A_2", "A_3"]   # can't invent clips


def test_select_mixes_two_buckets():
    pool = {0: [f"A_{i}" for i in range(100)], 1: [f"B_{i}" for i in range(100)]}
    chosen = asm.select_distinct(pool, {"0": 50, "1": 50, "2": 0, "3": 0, "4": 0}, 40)
    a = sum(s.startswith("A_") for s in chosen)
    b = sum(s.startswith("B_") for s in chosen)
    assert len(chosen) == 40 and a > 0 and b > 0


def test_fill_timeline_no_adjacent_repeats():
    distinct = ["x", "y", "z"]
    tl = asm.fill_timeline(distinct, 30)
    assert len(tl) == 30
    assert all(tl[i] != tl[i + 1] for i in range(len(tl) - 1)), "back-to-back repeat"
    assert set(tl) == set(distinct)


def test_fill_timeline_distinct_when_enough():
    tl = asm.fill_timeline([f"c{i}" for i in range(20)], 8)
    assert len(tl) == 8 and len(set(tl)) == 8


# ---- library buckets + usage --------------------------------------------
def test_bucket_naming_and_levels():
    assert lib.level_for_uses(0) == 0
    assert lib.level_for_uses(1) == 1
    assert lib.level_for_uses(3) == 3
    assert lib.level_for_uses(9) == lib.MAX_BUCKET
    assert lib.bucket_folder_name(1) == "11sec_used_1"
    assert lib.bucket_folder_name(lib.MAX_BUCKET) == "11sec_used_4plus"


def test_move_between_buckets_and_scan():
    root = tempfile.mkdtemp(prefix="snf_lib_")
    cfg = _Cfg(root)
    _make_master(cfg, 0, "A_aaa")
    _make_master(cfg, 0, "B_bbb")
    assert lib.master_exists(cfg, "A_aaa")
    counts = lib.bucket_counts(cfg)
    assert counts[0] == 2 and counts["total"] == 2

    # first use -> bucket 1
    newp = lib.move_to_bucket(cfg, "A_aaa", 1)
    assert newp and os.path.isfile(newp)
    assert "11sec_used_1" in newp
    assert lib.master_exists(cfg, "A_aaa")          # still findable after move
    counts = lib.bucket_counts(cfg)
    assert counts[0] == 1 and counts[1] == 1

    sc = lib.scan(cfg)
    assert sc["A_aaa"]["level"] == 1 and sc["B_bbb"]["level"] == 0


def test_state_roundtrip_atomic():
    root = tempfile.mkdtemp(prefix="snf_state_")
    cfg = _Cfg(root)
    st = lib.load_state(cfg)
    st["clips"]["A_x"] = {"uses": 2, "first_used_at": 123.0}
    st["weights"] = {"0": 70, "1": 30}
    lib.save_state(cfg, st)
    again = lib.load_state(cfg)
    assert again["clips"]["A_x"]["uses"] == 2
    assert again["weights"]["0"] == 70


# ---- retention -----------------------------------------------------------
def test_expired_stems():
    now = time.time()
    state = {"clips": {
        "A_old": {"first_used_at": now - 8 * 86400},   # 8 days -> expired
        "B_new": {"first_used_at": now - 2 * 86400},   # 2 days -> safe
        "C_never": {"uses": 0},                          # never used -> safe
    }}
    due = lib.expired_stems(state, 7, now=now)
    assert due == ["A_old"], due


def test_intermediate_paths_never_touch_masters():
    root = tempfile.mkdtemp(prefix="snf_prune_")
    cfg = _Cfg(root)
    a = asm.Assembler.__new__(asm.Assembler)
    a.cfg = cfg
    paths = a._intermediate_paths("A_deadbeef")
    # flux / x2 stages are keyed by the bare key (no class letter)
    assert any(p.endswith(os.path.join("01_flux", "deadbeef.png")) for p in paths)
    assert any(p.endswith(os.path.join("03_classified", "A_deadbeef.png")) for p in paths)
    # the finished-master folders must never appear in the prune list
    assert not any("09_final_up4" in p for p in paths)
    assert not any("11sec_used_" in p for p in paths)
    assert not any("Compilations" in p for p in paths)


# ---- preview -------------------------------------------------------------
def test_last_artifact_picks_newest():
    root = tempfile.mkdtemp(prefix="snf_prev_")
    cfg = _Cfg(root)
    d1 = cfg.stage_dir("flux"); d2 = cfg.stage_dir("vid1")
    older = os.path.join(d1, "old.png"); newer = os.path.join(d2, "new.mp4")
    open(older, "w").close()
    time.sleep(0.02)
    open(newer, "w").close()
    last = server._last_artifact(cfg)
    assert last and last["name"] == "new.mp4" and last["kind"] == "video"
    assert last["stage"] == "vid1"


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
