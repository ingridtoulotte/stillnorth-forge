"""Config + workflow-map validation. No GPU, no ComfyUI, no third-party deps.

Guards the integration that the GPU stages depend on: every node id referenced
in config/workflows.json must actually exist (with the named input field) in the
workflow JSON it points at, and the motion config must be complete and coherent.

    python tests/test_config.py     # prints PASS/FAIL
    pytest
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stillnorth.config import get_config, STAGE_DIRS  # noqa: E402

CFG = get_config()


def _wf(which):
    return json.load(open(CFG.workflow_path(which), encoding="utf-8"))


def _assert_node_field(wf, node, field, where):
    assert node in wf, f"{where}: node {node} missing from workflow"
    assert field in wf[node].get("inputs", {}), \
        f"{where}: node {node} has no input '{field}'"


def test_flux_node_map():
    wf = _wf("flux")
    m = CFG.workflows["flux"]
    _assert_node_field(wf, m["node_text"], m["field_text"], "flux.text")
    _assert_node_field(wf, m["node_seed"], m["field_seed"], "flux.seed")
    _assert_node_field(wf, m["node_save"], m["field_save"], "flux.save")


def test_wan_node_map():
    wf = _wf("wan")
    m = CFG.workflows["wan"]
    _assert_node_field(wf, m["node_image"], m["field_image"], "wan.image")
    _assert_node_field(wf, m["node_text"], m["field_text"], "wan.text")
    _assert_node_field(wf, m["node_camera"], m["field_pose"], "wan.pose")
    _assert_node_field(wf, m["node_camera"], m["field_speed"], "wan.speed")
    _assert_node_field(wf, m["node_seed"], m["field_seed"], "wan.seed")
    _assert_node_field(wf, m["node_save"], m["field_save"], "wan.save")


def test_motion_classes_complete():
    for letter in "ABCD":
        assert letter in CFG.classes, f"missing class {letter}"
        assert CFG.classes[letter]["motion"].strip(), f"empty motion for {letter}"
        assert CFG.motion_text(letter) == CFG.classes[letter]["motion"]


def test_motion_keyword_override():
    """A class may reroute to a content-specific motion prompt when the FLUX
    source prompt matches a keyword (e.g. waterfalls must not get the
    calm-water class-C prompt — at cfg=1 it freezes the falls)."""
    cls = CFG.classes["C"]
    saved = cls.get("overrides")
    cls["overrides"] = [{"keywords": ["waterfall", "cascade"],
                         "motion": "FALLS-MOTION"}]
    try:
        assert CFG.motion_text("C", "cliffs with small waterfalls") == \
            "FALLS-MOTION"
        assert CFG.motion_text("C", "three cascades tumble") == "FALLS-MOTION"
        assert CFG.motion_text("C", "a calm mirror lake") == cls["motion"]
        assert CFG.motion_text("C") == cls["motion"]          # no text given
        assert CFG.motion_text("A", "waterfall") == \
            CFG.classes["A"]["motion"]                        # other classes untouched
    finally:
        if saved is None:
            cls.pop("overrides", None)
        else:
            cls["overrides"] = saved


def test_flux_suffix_augmentation():
    """Falls-keyword sources get fast-shutter language appended to the FLUX
    prompt (a long-exposure silky waterfall still gives Wan nothing to move)."""
    saved = CFG.flux_suffixes
    CFG.flux_suffixes = [{"keywords": ["waterfall"], "suffix": " SHUTTER"}]
    try:
        assert CFG.flux_text("cliffs with waterfalls") == \
            "cliffs with waterfalls SHUTTER"
        assert CFG.flux_text("a calm lake") == "a calm lake"
        assert CFG.flux_text("") == ""
    finally:
        CFG.flux_suffixes = saved


def test_live_falls_config_coherent():
    """The shipped motion_prompts.json falls override + flux suffix must
    stay keyword-aligned so both halves of the fix fire together."""
    ov = CFG.classes["C"].get("overrides", [])
    assert ov, "class C falls override missing"
    assert CFG.flux_suffixes, "flux_suffixes missing"
    assert set(ov[0]["keywords"]) == set(CFG.flux_suffixes[0]["keywords"])
    assert "downward" in ov[0]["motion"]
    assert "shutter" in CFG.flux_suffixes[0]["suffix"].lower()


def test_poses_have_speeds():
    assert CFG.poses, "no poses configured"
    for p in CFG.poses:
        assert p in CFG.speed_by_pose, f"pose '{p}' has no speed"
        assert 0 < CFG.speed_by_pose[p] <= 2, f"odd speed for {p}"


def test_stage_dirs_unique_and_ordered():
    vals = list(STAGE_DIRS.values())
    assert len(vals) == len(set(vals)), "duplicate stage folder names"
    assert vals == sorted(vals), "stage folders not in pipeline order"


# ---- scene-aware camera (2026-07-12) --------------------------------------

def test_pose_groups_cover_all_poses_and_have_speeds():
    grouped = [p for g in CFG.pose_groups.values() for p in g["poses"]]
    for p in CFG.poses:
        assert p in grouped, f"pose '{p}' is in no pose_group"
    for p in grouped:
        assert p in CFG.speed_by_pose, f"grouped pose '{p}' has no speed"


def test_choose_pose_weighted_5050_split():
    import random
    from collections import Counter
    rng = random.Random(7)
    c = Counter(CFG.choose_pose([], rng)[0] for _ in range(4000))
    up_back = sum(c[p] for p in CFG.pose_groups["up_back"]["poses"])
    frac = up_back / sum(c.values())
    assert 0.42 < frac < 0.58, f"up_back group not ~50%: {frac:.2f}"


def test_choose_pose_excludes_blocked_directions():
    import random
    rng = random.Random(7)
    for _ in range(400):
        assert CFG.choose_pose(["FORWARD"], rng)[0] != "Zoom In"
    for _ in range(400):
        assert CFG.choose_pose(["LEFT", "RIGHT"], rng)[0] not in ("Pan Left", "Pan Right")


def test_choose_pose_all_blocked_falls_back_gracefully():
    import random
    rng = random.Random(7)
    pose, speed = CFG.choose_pose(["FORWARD", "LEFT", "RIGHT", "UP"], rng)
    assert pose == "Zoom Out"          # only dolly-back survives
    assert speed > 0


def test_choose_pose_speed_matches_pose():
    import random
    rng = random.Random(1)
    pose, speed = CFG.choose_pose([], rng)
    assert pose in CFG.poses
    assert CFG.speed_by_pose[pose] == speed


def test_config_numbers_sane():
    assert CFG.img_mult == 2 and CFG.lf_mult == 4 and CFG.final_mult == 4
    assert CFG.fps > 0 and CFG.port > 0
    assert CFG.comfy_prefix("flux", "abc").endswith("/01_flux/abc")


def test_native_long_config_and_node_map():
    """The seam fix renders the whole clip in one native Wan pass; that needs a
    length-node map and sane frame count."""
    assert isinstance(CFG.native_long, bool)
    assert CFG.native_frames > 1
    assert CFG.cont_seed in ("native", "upscaled")
    if CFG.native_long:                      # length override must be mappable
        wf = _wf("wan")
        m = CFG.workflows["wan"]
        _assert_node_field(wf, m["node_length"], m["field_length"], "wan.length")


def test_nodes_cli_resolves_workflows():
    """`stillnorth nodes <name>` must accept a bare filename (resolved against
    workflows/) and return None for a bogus name instead of crashing."""
    from stillnorth.__main__ import _resolve_workflow
    for which in ("flux", "wan"):
        bare = os.path.basename(CFG.workflows[which]["file"])
        assert _resolve_workflow(bare), f"bare name '{bare}' did not resolve"
    assert _resolve_workflow("does-not-exist.json") is None


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
