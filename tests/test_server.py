"""Server + pipeline-robustness tests. No GPU, no ComfyUI, no third-party deps.

Covers the security-sensitive file endpoint (must never read outside the
workspace), the outputs listing shape, error categorisation, and that a submit
aborts immediately when cancelled.

    python tests/test_server.py     # prints PASS/FAIL
    pytest
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stillnorth import server                       # noqa: E402
from stillnorth.config import STAGE_DIRS            # noqa: E402
from stillnorth.pipeline import Pipeline            # noqa: E402


class _Cfg:
    def __init__(self, ws):
        self.workspace = ws


def test_file_sandbox_blocks_escape():
    ws = tempfile.mkdtemp(prefix="snf_ws_")
    inside = os.path.join(ws, "01_flux"); os.makedirs(inside)
    open(os.path.join(inside, "a.png"), "w").close()
    secret = os.path.join(os.path.dirname(ws), "secret.txt"); open(secret, "w").close()
    cfg = _Cfg(ws)
    assert server._safe_workspace_path(cfg, "01_flux/a.png"), "in-bounds file rejected"
    assert server._safe_workspace_path(cfg, "../secret.txt") is None, "traversal allowed!"
    assert server._safe_workspace_path(cfg, "..\\secret.txt") is None, "win traversal allowed!"
    assert server._safe_workspace_path(cfg, os.path.abspath(secret)) is None, "abs escape allowed!"
    assert server._safe_workspace_path(cfg, "01_flux/missing.png") is None
    assert server._safe_workspace_path(cfg, "") is None


def test_list_outputs_shape():
    ws = tempfile.mkdtemp(prefix="snf_ws_")
    d = os.path.join(ws, STAGE_DIRS["final_up"]); os.makedirs(d)
    open(os.path.join(d, "A_x.mp4"), "w").close()
    open(os.path.join(d, "note.txt"), "w").close()      # ignored: not media
    out = server._list_outputs(_Cfg(ws))
    assert set(out.keys()) == set(STAGE_DIRS.keys())
    files = out["final_up"]["files"]
    assert len(files) == 1 and files[0]["kind"] == "video"
    assert files[0]["rel"] == STAGE_DIRS["final_up"] + "/A_x.mp4"


def test_error_categories():
    c = Pipeline._categorize
    assert c("URLError: connection refused") == "comfy offline"
    assert c("timeout") == "render timeout (GPU/slow)"
    assert "workflow" in c("comfy error")
    assert "workflow" in c("no output")


def test_submit_aborts_when_cancelled():
    p = Pipeline()
    p.cancel_flag = True                               # already cancelled
    ok, msg = p._submit({}, 1, "flux")                 # must not touch the network
    assert ok is False and msg == "cancelled"


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
