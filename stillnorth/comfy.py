"""Tiny ComfyUI HTTP client + output-file helpers (standard library only)."""
import glob
import json
import os
import time
import urllib.error
import urllib.request


class Comfy:
    def __init__(self, server, poll=2.0):
        self.server = server
        self.poll = poll

    def _http(self, path, payload=None, timeout=30):
        url = f"http://{self.server}{path}"
        data = json.dumps(payload).encode() if payload is not None else None
        headers = {"Content-Type": "application/json"} if data else {}
        req = urllib.request.Request(url, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}

    def reachable(self):
        try:
            self._http("/system_stats", timeout=5)
            return True
        except Exception:
            return False

    def queue(self, workflow):
        """Submit an API-format workflow dict, return its prompt_id."""
        return self._http("/prompt", {"prompt": workflow})["prompt_id"]

    def wait(self, prompt_id, timeout, cancel=None):
        """Block until the prompt finishes. Returns (ok: bool, msg: str).

        `cancel` is an optional callable returning True to abort early.
        """
        t0 = time.time()
        while time.time() - t0 < timeout:
            if cancel and cancel():
                return False, "cancelled"
            try:
                hist = self._http(f"/history/{prompt_id}")
            except urllib.error.URLError:
                time.sleep(self.poll); continue
            except Exception:
                time.sleep(self.poll); continue
            if prompt_id in hist:
                st = hist[prompt_id].get("status", {})
                if st.get("status_str") == "error":
                    return False, "comfy error"
                if st.get("completed") or st.get("status_str") == "success":
                    return (True, "ok") if hist[prompt_id].get("outputs") \
                        else (False, "no output")
            time.sleep(self.poll)
        return False, "timeout"

    def interrupt(self):
        try:
            self._http("/interrupt", payload={})
        except Exception:
            pass


def rename_out(dest_dir, stem, exts=(".png", ".mp4", ".webm")):
    """ComfyUI Save nodes append ``_00001_`` to filename_prefix. Collapse the
    newest match back to ``<stem><ext>`` so downstream stages have stable names.
    Returns the final path or None.
    """
    hits = []
    for ext in exts:
        hits += glob.glob(os.path.join(dest_dir, f"{stem}_*{ext}"))
    # also accept an already-correct file
    for ext in exts:
        p = os.path.join(dest_dir, f"{stem}{ext}")
        if os.path.exists(p):
            return p
    if not hits:
        return None
    hits.sort(key=os.path.getmtime)
    src = hits[-1]
    ext = os.path.splitext(src)[1]
    final = os.path.join(dest_dir, f"{stem}{ext}")
    if os.path.abspath(src) == os.path.abspath(final):
        return final
    if os.path.exists(final):
        try:
            os.remove(final)
        except OSError:
            pass
    # ComfyUI may briefly hold the file open after saving -> retry on lock
    for _ in range(15):
        try:
            os.replace(src, final)
            return final
        except PermissionError:
            time.sleep(1)
    return None
