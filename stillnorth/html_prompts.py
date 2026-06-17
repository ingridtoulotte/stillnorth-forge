"""Robust prompt extraction from TSN HTML files.

Ported from the project's proven `tsn_comfy_runner.py`. It does not rely on any
particular HTML layout: it scans the raw text for every balanced ``{...}`` block
and keeps the ones that contain a ``"scene"`` key (the shape of every TSN FLUX
prompt). Handles single-quoted JS object literals, HTML-escaped entities,
unquoted keys, comments and trailing commas. Pure standard library.
"""
import hashlib
import html
import json
import re

__all__ = ["extract_prompts_from_text", "extract_prompts_from_file",
           "prompt_key", "prompt_to_text"]


def _js_to_json(s):
    """Best-effort convert a JS object-literal string to strict JSON."""
    out = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == "/" and i + 1 < n and s[i + 1] == "/":
            while i < n and s[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and s[i + 1] == "*":
            i += 2
            while i + 1 < n and not (s[i] == "*" and s[i + 1] == "/"):
                i += 1
            i += 2
            continue
        if c == '"':
            out.append(c)
            i += 1
            while i < n:
                out.append(s[i])
                if s[i] == "\\" and i + 1 < n:
                    out.append(s[i + 1]); i += 2; continue
                if s[i] == '"':
                    i += 1; break
                i += 1
            continue
        if c == "'":
            out.append('"')
            i += 1
            while i < n and s[i] != "'":
                ch = s[i]
                if ch == "\\" and i + 1 < n:
                    out.append(ch); out.append(s[i + 1]); i += 2; continue
                out.append('\\"' if ch == '"' else ch)
                i += 1
            out.append('"'); i += 1
            continue
        out.append(c); i += 1
    js = "".join(out)
    js = re.sub(r'([{\[,]\s*)([A-Za-z_$][\w$]*)(\s*:)', r'\1"\2"\3', js)  # quote keys
    js = re.sub(r",\s*([}\]])", r"\1", js)                                # trailing commas
    return js


def _balanced_json_objects(text):
    """Yield each balanced {...} block parsed as a JSON object."""
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1; continue
        depth = 0; quote = None; esc = False; end = None
        for j in range(i, n):
            c = text[j]
            if quote:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == quote:
                    quote = None
            else:
                if c in "\"'":
                    quote = c
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        end = j + 1; break
        if end is None:
            break
        frag = text[i:end]
        obj = None
        for candidate in (frag, html.unescape(frag)):
            try:
                obj = json.loads(candidate); break
            except Exception:
                obj = None
        if obj is None:
            for candidate in (frag, html.unescape(frag)):
                try:
                    obj = json.loads(_js_to_json(candidate)); break
                except Exception:
                    obj = None
        if obj is not None:
            yield obj
            i = end
        else:
            i += 1


def _find_scene_dicts(obj):
    """Yield every dict carrying a 'scene' key, however deeply nested."""
    if isinstance(obj, dict):
        if "scene" in obj:
            yield obj
        else:
            for v in obj.values():
                yield from _find_scene_dicts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _find_scene_dicts(v)
    elif isinstance(obj, str):
        s = obj.strip()
        if s.startswith("{") and '"scene"' in s:
            try:
                yield from _find_scene_dicts(json.loads(s))
            except Exception:
                pass


def extract_prompts_from_text(text):
    """Return list of prompt dicts (objects containing a 'scene' key)."""
    prompts = []
    for obj in _balanced_json_objects(text):
        prompts.extend(_find_scene_dicts(obj))
    return prompts


def extract_prompts_from_file(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return extract_prompts_from_text(fh.read())


def prompt_key(obj):
    """Stable 16-char id for de-dup / state / filenames."""
    basis = (obj.get("scene", "") + "|" + obj.get("title", "")).strip()
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


def prompt_to_text(obj, flatten=False):
    """Turn the prompt dict into the string fed to the text encoder.

    Default: the full JSON object (matches the FLUX.2 workflow's own example,
    which embeds the whole structured prompt). flatten=True joins the
    descriptive fields into a plain comma-separated sentence instead.
    """
    if not flatten:
        return json.dumps(obj, ensure_ascii=False)
    parts = []
    for k in ("scene", "location", "composition", "lighting",
              "motion_elements", "style", "mood"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    return ", ".join(parts)
