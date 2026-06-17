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


def test_poses_have_speeds():
    assert CFG.poses, "no poses configured"
    for p in CFG.poses:
        assert p in CFG.speed_by_pose, f"pose '{p}' has no speed"
        assert 0 < CFG.speed_by_pose[p] <= 2, f"odd speed for {p}"


def test_stage_dirs_unique_and_ordered():
    vals = list(STAGE_DIRS.values())
    assert len(vals) == len(set(vals)), "duplicate stage folder names"
    assert vals == sorted(vals), "stage folders not in pipeline order"


def test_config_numbers_sane():
    assert CFG.img_mult == 2 and CFG.lf_mult == 4 and CFG.final_mult == 4
    assert CFG.fps > 0 and CFG.port > 0
    assert CFG.comfy_prefix("flux", "abc").endswith("/01_flux/abc")


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
