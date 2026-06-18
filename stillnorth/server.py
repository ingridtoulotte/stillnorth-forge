"""HTTP server: REST API + static web UI. Standard library only.

Endpoints
---------
GET  /                 -> web/index.html
GET  /<asset>          -> web/<asset>  (css/js)
GET  /api/status       -> pipeline snapshot + live VRAM + ComfyUI reachability
GET  /api/health       -> environment check (ComfyUI, ffmpeg)
GET  /api/log          -> last N lines of forge.log (live activity feed)
GET  /api/outputs      -> rendered files per stage (name/size/mtime/kind)
GET  /api/file?path=   -> stream a rendered file (sandboxed to the workspace)
POST /api/ingest       -> {name, html}  add prompts from one HTML payload
POST /api/run          -> start/resume the worker
POST /api/cancel       -> pause after current item (resumable)
POST /api/clear        -> forget queued prompts (keeps rendered media)
POST /api/purge        -> DESTRUCTIVE: delete all rendered output + reset state
"""
import json
import os
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .config import get_config, STAGE_DIRS
from .pipeline import get_pipeline
from .assembler import get_assembler
from . import media

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
MIME = {".html": "text/html", ".css": "text/css", ".js": "application/javascript",
        ".json": "application/json", ".svg": "image/svg+xml", ".ico": "image/x-icon"}
MEDIA_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
              ".webp": "image/webp", ".mp4": "video/mp4", ".webm": "video/webm"}
IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")
VID_EXTS = (".mp4", ".webm")


def _tail_log(cfg, n=40):
    """Return the last `n` lines of forge.log (newest last), or []."""
    try:
        with open(cfg.log_path(), "r", encoding="utf-8", errors="replace") as fh:
            return [ln.rstrip("\n") for ln in fh.readlines()[-n:]]
    except Exception:
        return []


def _list_outputs(cfg, limit=300):
    """Per-stage listing of rendered media (newest first)."""
    out = {}
    for key, folder in STAGE_DIRS.items():
        d = os.path.join(cfg.workspace, folder)
        files = []
        if os.path.isdir(d):
            for f in os.listdir(d):
                p = os.path.join(d, f)
                ext = os.path.splitext(f)[1].lower()
                kind = "image" if ext in IMG_EXTS else "video" if ext in VID_EXTS else None
                if kind is None or not os.path.isfile(p):
                    continue
                try:
                    stt = os.stat(p)
                except OSError:
                    continue
                files.append({"name": f, "rel": folder + "/" + f, "size": stt.st_size,
                              "mtime": int(stt.st_mtime), "kind": kind})
            files.sort(key=lambda e: e["mtime"], reverse=True)
        out[key] = {"folder": folder, "files": files[:limit], "total": len(files)}
    return out


def _last_artifact(cfg):
    """The single most-recently written media file across every stage folder
    (newest by mtime) -- powers the minimal 'last done only' preview."""
    newest = None
    for key, folder in STAGE_DIRS.items():
        d = os.path.join(cfg.workspace, folder)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            p = os.path.join(d, f)
            ext = os.path.splitext(f)[1].lower()
            kind = "image" if ext in IMG_EXTS else "video" if ext in VID_EXTS else None
            if kind is None or not os.path.isfile(p):
                continue
            try:
                mt = os.path.getmtime(p)
            except OSError:
                continue
            if newest is None or mt > newest["_mt"]:
                newest = {"rel": folder + "/" + f, "name": f, "kind": kind,
                          "stage": key, "mtime": int(mt), "_mt": mt}
    if newest:
        newest.pop("_mt", None)
    return newest


def _safe_workspace_path(cfg, rel):
    """Resolve `rel` under the workspace and refuse anything that escapes it.

    realpath collapses `..` and symlinks, so a request like `../../secret` or an
    absolute path can never read outside the rendered-output workspace.
    """
    if not rel:
        return None
    base = os.path.realpath(cfg.workspace)
    full = os.path.realpath(os.path.join(base, rel))
    if full != base and not full.startswith(base + os.sep):
        return None
    return full if os.path.isfile(full) else None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # quiet; pipeline logs to forge.log

    # -- helpers ------------------------------------------------------------
    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            return {}

    def _static(self, path):
        if path in ("/", ""):
            path = "/index.html"
        full = os.path.normpath(os.path.join(WEB_DIR, path.lstrip("/")))
        if not full.startswith(WEB_DIR) or not os.path.isfile(full):
            return self._send(404, {"error": "not found"})
        ext = os.path.splitext(full)[1].lower()
        with open(full, "rb") as fh:
            self._send(200, fh.read(), MIME.get(ext, "application/octet-stream"))

    def _serve_media(self, full):
        """Stream a media file with the right MIME and HTTP Range support so
        the in-app gallery can seek inside clips."""
        ext = os.path.splitext(full)[1].lower()
        ctype = MEDIA_MIME.get(ext, "application/octet-stream")
        size = os.path.getsize(full)
        rng = self.headers.get("Range", "")
        if rng.startswith("bytes="):
            try:
                s, e = rng[6:].split("-", 1)
                start = int(s) if s else 0
                end = int(e) if e else size - 1
                end = min(end, size - 1)
                if start > end:
                    raise ValueError
                with open(full, "rb") as fh:
                    fh.seek(start)
                    data = fh.read(end - start + 1)
                self.send_response(206)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            except Exception:
                pass
        with open(full, "rb") as fh:
            data = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self.wfile.write(data)

    # -- routes -------------------------------------------------------------
    def do_GET(self):
        pipe = get_pipeline()
        if self.path == "/api/status":
            snap = pipe.snapshot()
            v = media.vram()
            snap["vram"] = ({"used": v[0], "total": v[1], "name": v[2],
                             "pct": round(v[0] / v[1] * 100)} if v else None)
            snap["comfy"] = pipe.comfy.reachable()
            return self._send(200, snap)
        if self.path == "/api/health":
            cfg = get_config()
            return self._send(200, {
                "comfy": pipe.comfy.reachable(),
                "comfy_server": cfg.comfy_server,
                "ffmpeg": os.path.exists(cfg.ffmpeg),
                "workspace": cfg.workspace,
            })
        if self.path == "/api/log":
            return self._send(200, {"lines": _tail_log(get_config(), 40)})
        if self.path == "/api/outputs":
            return self._send(200, _list_outputs(get_config()))
        if self.path == "/api/preview":
            return self._send(200, {"last": _last_artifact(get_config())})
        if self.path == "/api/library":
            return self._send(200, get_assembler().snapshot())
        if self.path.startswith("/api/file?"):
            rel = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).get("path", [""])[0]
            full = _safe_workspace_path(get_config(), rel)
            if not full:
                return self._send(404, {"error": "not found or outside workspace"})
            return self._serve_media(full)
        return self._static(self.path)

    def do_POST(self):
        pipe = get_pipeline()
        if self.path == "/api/ingest":
            d = self._body()
            name = d.get("name", "pasted.html")
            html = d.get("html", "")
            if not html:
                return self._send(400, {"error": "no html"})
            added, found = pipe.ingest_html(name, html)
            return self._send(200, {"added": added, "found": found,
                                    "total": len(pipe.prompts)})
        if self.path == "/api/run":
            started = pipe.start()
            return self._send(200, {"started": started})
        if self.path == "/api/cancel":
            pipe.cancel()
            return self._send(200, {"ok": True})
        if self.path == "/api/clear":
            pipe.clear_queue()
            return self._send(200, {"ok": True})
        if self.path == "/api/purge":
            removed = pipe.purge_outputs()
            return self._send(200, {"ok": True, "removed": removed})
        if self.path == "/api/assemble":
            d = self._body()
            asm = get_assembler()
            ok, msg = asm.start_build(d.get("duration", ""), d.get("weights"))
            return self._send(200 if ok else 400, {"ok": ok, "msg": msg})
        if self.path == "/api/assemble/cancel":
            get_assembler().cancel()
            return self._send(200, {"ok": True})
        if self.path == "/api/weights":
            d = self._body()
            get_assembler().save_weights(d.get("weights", {}))
            return self._send(200, {"ok": True})
        if self.path == "/api/cleanup":
            d = self._body()
            removed = get_assembler().run_retention(dry_run=bool(d.get("dry_run")))
            return self._send(200, {"ok": True, "removed": removed})
        return self._send(404, {"error": "unknown endpoint"})


def serve(open_browser=True):
    cfg = get_config()
    pipe = get_pipeline()   # init (loads state)
    asm = get_assembler()
    # crash-safe: if a batch / long-video build was running when the terminal
    # was closed or the PC shut down, pick it up automatically on launch.
    try:
        if pipe.maybe_auto_resume():
            print("auto-resumed the generation batch")
        if asm.maybe_resume():
            print("auto-resumed an unfinished long-video build")
        asm.run_retention()    # one-week auto-delete of aged intermediates
    except Exception as e:
        print(f"startup resume/retention skipped: {e}")
    httpd = ThreadingHTTPServer((cfg.host, cfg.port), Handler)
    url = f"http://{cfg.host}:{cfg.port}"
    print(f"StillNorth Forge UI  ->  {url}")
    print(f"workspace: {cfg.workspace}")
    print(f"ComfyUI:   {cfg.comfy_server}   ffmpeg: {cfg.ffmpeg}")
    if open_browser:
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
