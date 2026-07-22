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
                    choices=["serve", "nodes", "loop"],
                    help="serve (default), nodes, or loop")
    ap.add_argument("arg", nargs="?", help="workflow json path for 'nodes'")
    ap.add_argument("--no-browser", action="store_true")
    # 'loop' options (§5E slideshow loop-publishing)
    ap.add_argument("--stills", help="loop: comma-separated source-still hashes")
    ap.add_argument("--name", default="loop", help="loop: output basename")
    ap.add_argument("--audio", help="loop: ambient bed (wind/rain/drone/still)")
    ap.add_argument("--no-kenburns", action="store_true",
                    help="loop: static stills instead of Ken Burns")
    ap.add_argument("--motion", choices=["kenburns", "dive"], default="kenburns",
                    help="loop: shot motion -- 'kenburns' (default) or 'dive' "
                         "(DepthFlow depth-parallax; needs the dive venv "
                         "configured in config.json)")
    ap.add_argument("--no-shorts", action="store_true",
                    help="loop: skip the 9:16 Shorts cut")
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

    if a.cmd == "loop":
        if not a.stills:
            sys.exit("usage: python -m stillnorth loop --stills A_x,B_y,... "
                     "[--name N] [--audio wind] [--no-kenburns] [--no-shorts]")
        from .config import get_config
        from . import loopjob
        cfg = get_config()
        if a.motion == "dive":
            from . import dive
            if not cfg.dive_enabled:
                sys.exit("dive motion is disabled -- set dive.enabled=true in "
                         "config/config.json")
            if not dive.dive_available(cfg):
                sys.exit("dive venv not found -- set dive.venv_python (a python with "
                         "depthflow installed) in config.json's dive block")
        hashes = [h.strip() for h in a.stills.split(",") if h.strip()]
        try:
            res = loopjob.build_loop_job(
                cfg, hashes, name=a.name, audio_kind=a.audio,
                make_shorts=not a.no_shorts, kenburns=not a.no_kenburns,
                motion=a.motion, log=lambda m: print(f"  {m}"))
        except FileNotFoundError as e:
            sys.exit(str(e))
        if not res:
            sys.exit("loop build failed (see log above)")
        print(f"\nbase loop : {res['base']}  (~{res['duration']:.0f}s)")
        for k, p in res["tiers"].items():
            print(f"tier {k:<6}: {p}")
        if res["short"]:
            print(f"shorts    : {res['short']}")
        return

    from .server import serve
    serve(open_browser=not a.no_browser)


if __name__ == "__main__":
    main()
