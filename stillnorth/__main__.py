"""CLI entrypoint.

    python -m stillnorth                 # launch the web UI (default)
    python -m stillnorth --no-browser    # launch UI, don't open a browser
    python -m stillnorth nodes <wf.json> # list text/save nodes of a workflow
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOWS_DIR = os.path.join(ROOT, "workflows")


def _resolve_workflow(arg):
    """Find the workflow JSON the user meant: an exact/relative path, or a bare
    filename that lives in the repo's workflows/ dir. Returns a path or None."""
    candidates = [arg]
    if not os.path.isabs(arg):
        candidates.append(os.path.join(WORKFLOWS_DIR, os.path.basename(arg)))
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _list_nodes(path):
    try:
        with open(path, encoding="utf-8") as fh:
            wf = json.load(fh)
    except json.JSONDecodeError as e:
        sys.exit(f"not valid JSON: {path}\n  {e}")
    if not isinstance(wf, dict):
        sys.exit(f"not an API-format workflow (expected a node dict): {path}")
    print(f"Text / Save / Camera nodes in {path}:")
    for nid, node in wf.items():
        ct = node.get("class_type", "?")
        inp = node.get("inputs", {})
        if any(key in inp for key in ("text", "image", "filename_prefix",
                                      "camera_pose", "noise_seed")) \
                or "Text" in ct or "Save" in ct:
            keys = [k for k in ("text", "image", "filename_prefix",
                                "camera_pose", "speed", "noise_seed") if k in inp]
            preview = str(inp.get("text", ""))[:50].replace("\n", " ")
            preview = preview.encode("ascii", "replace").decode()  # console-safe
            print(f"  node {nid:>6}  {ct:<26} fields={keys} {preview}")
    print('\nMap the ids you want into config/workflows.json')


def main():
    try:  # console-safe logging regardless of Windows code page
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(prog="stillnorth")
    ap.add_argument("cmd", nargs="?", default="serve",
                    choices=["serve", "nodes"], help="serve (default) or nodes")
    ap.add_argument("arg", nargs="?", help="workflow json path for 'nodes'")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    if a.cmd == "nodes":
        if not a.arg:
            sys.exit("usage: python -m stillnorth nodes <workflow.json>")
        path = _resolve_workflow(a.arg)
        if not path:
            avail = sorted(glob.glob(os.path.join(WORKFLOWS_DIR, "*.json")))
            lines = [f"workflow not found: {a.arg}"]
            if avail:
                lines.append("available workflows (pass one of these):")
                lines += [f"  workflows/{os.path.basename(p)}" for p in avail]
            else:
                lines.append(f"(no .json files in {WORKFLOWS_DIR})")
            sys.exit("\n".join(lines))
        return _list_nodes(path)

    from .server import serve
    serve(open_browser=not a.no_browser)


if __name__ == "__main__":
    main()
