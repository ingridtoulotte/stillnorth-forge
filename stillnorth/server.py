"""HTTP server: REST API + static web UI. Standard library only.

Endpoints
---------
GET  /                 -> web/index.html
GET  /<asset>          -> web/<asset>  (css/js)
GET  /api/status       -> pipeline snapshot + live VRAM + ComfyUI reachability
GET  /api/health       -> environment check (ComfyUI, ffmpeg)
POST /api/ingest       -> {name, html}  add prompts from one HTML payload
POST /api/run          -> start/resume the worker
POST /api/cancel       -> pause after current item (resumable)
POST /api/clear        -> forget queued prompts (keeps rendered media)
"""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .config import get_config
from .pipeline import get_pipeline
from . import media

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
MIME = {".html": "text/html", ".css": "text/css", ".js": "application/javascript",
        ".json": "application/json", ".svg": "image/svg+xml", ".ico": "image/x-icon"}


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
        return self._send(404, {"error": "unknown endpoint"})


def serve(open_browser=True):
    cfg = get_config()
    get_pipeline()  # init (loads state)
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
