"""CLI entrypoint.

    python -m stillnorth                 # launch the web UI (default)
    python -m stillnorth --no-browser    # launch UI, don't open a browser
    python -m stillnorth nodes <wf.json> # list text/save nodes of a workflow
"""
import argparse
import json
import sys


def _list_nodes(path):
    wf = json.load(open(path, encoding="utf-8"))
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
        return _list_nodes(a.arg)

    from .server import serve
    serve(open_browser=not a.no_browser)


if __name__ == "__main__":
    main()
