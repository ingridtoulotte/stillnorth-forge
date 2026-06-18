"""Master-clip library + usage tracking for the long-video assembler.

The finished ~11-second masters land in the pipeline's ``09_final_up4`` folder.
A master that has NEVER been used in a compilation lives there (bucket level 0).
The first time the assembler places a master into a long video it is MOVED into
a sibling "used" bucket folder under the ComfyUI output root, and bumped one
bucket higher on every later build it is part of::

    <workspace>/09_final_up4/   uses = 0   (never used)
    <output>/11sec_used_1/      uses = 1
    <output>/11sec_used_2/      uses = 2
    <output>/11sec_used_3/      uses = 3
    <output>/11sec_used_4plus/  uses >= 4  (capped bucket)

The folder a clip sits in is the durable, manually-inspectable source of truth
for its bucket level; ``assembler_state.json`` caches the exact per-clip use
count and the first-used timestamp (used by the one-week auto-delete retention).

Everything here is pure standard library so it is unit-testable without ffmpeg,
a GPU or ComfyUI.
"""
import json
import os
import time

VID_EXTS = (".mp4", ".webm")
BUCKET_PREFIX = "11sec_used_"
MAX_BUCKET = 4                       # level >= 4 collapses into "4plus"
COMPILATIONS_DIRNAME = "Compilations"
CACHE_DIRNAME = "_assembler_cache"
STATE_NAME = "assembler_state.json"


# -- bucket level / folder naming -----------------------------------------
def level_for_uses(uses):
    """Bucket level (0..MAX_BUCKET) for an exact use count."""
    if uses <= 0:
        return 0
    return uses if uses < MAX_BUCKET else MAX_BUCKET


def bucket_folder_name(level):
    """Folder name for a used bucket (level >= 1)."""
    return f"{BUCKET_PREFIX}{MAX_BUCKET}plus" if level >= MAX_BUCKET \
        else f"{BUCKET_PREFIX}{level}"


def bucket_dir(cfg, level, ensure=False):
    """Absolute path of a bucket folder. Level 0 is the pipeline's final_up
    output; every higher level is a sibling folder under the ComfyUI output."""
    if level <= 0:
        d = cfg.stage_dir("final_up", ensure=ensure)
        return d
    d = os.path.join(cfg.comfy_output, bucket_folder_name(level))
    if ensure:
        os.makedirs(d, exist_ok=True)
    return d


def all_bucket_dirs(cfg):
    """Every folder a master might live in, level 0 first."""
    return [bucket_dir(cfg, lvl) for lvl in range(0, MAX_BUCKET + 1)]


def compilations_dir(cfg, ensure=False):
    d = os.path.join(cfg.comfy_output, COMPILATIONS_DIRNAME)
    if ensure:
        os.makedirs(d, exist_ok=True)
    return d


def cache_dir(cfg, ensure=False):
    d = os.path.join(cfg.workspace, CACHE_DIRNAME)
    if ensure:
        os.makedirs(d, exist_ok=True)
    return d


# -- locating masters ------------------------------------------------------
def _master_in_dir(d, stem):
    for ext in VID_EXTS:
        p = os.path.join(d, stem + ext)
        if os.path.isfile(p):
            return p
    return None


def find_master(cfg, stem):
    """Path of a master named `stem` anywhere in the library, or None."""
    for lvl in range(0, MAX_BUCKET + 1):
        p = _master_in_dir(bucket_dir(cfg, lvl), stem)
        if p:
            return p
    return None


def master_exists(cfg, stem):
    """True if a finished master for `stem` exists anywhere in the library.

    Used by the pipeline so a master that has been MOVED into a used bucket is
    never mistaken for missing work and re-rendered."""
    return find_master(cfg, stem) is not None


def scan(cfg):
    """Map stem -> {path, level, mtime} for every master across all buckets.

    If the same stem somehow exists in two buckets the higher one wins (it has
    been used more)."""
    out = {}
    for lvl in range(0, MAX_BUCKET + 1):
        d = bucket_dir(cfg, lvl)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            stem, ext = os.path.splitext(f)
            if ext.lower() not in VID_EXTS:
                continue
            p = os.path.join(d, f)
            if not os.path.isfile(p):
                continue
            prev = out.get(stem)
            if prev and prev["level"] >= lvl:
                continue
            try:
                mtime = int(os.path.getmtime(p))
            except OSError:
                mtime = 0
            out[stem] = {"path": p, "level": lvl, "mtime": mtime}
    return out


def bucket_counts(cfg):
    """{level: count, ...} plus 'total' across every bucket."""
    counts = {lvl: 0 for lvl in range(0, MAX_BUCKET + 1)}
    for info in scan(cfg).values():
        counts[info["level"]] += 1
    counts["total"] = sum(counts[lvl] for lvl in range(0, MAX_BUCKET + 1))
    return counts


def move_to_bucket(cfg, stem, new_level):
    """Move the master `stem` into the folder for `new_level`. Returns the new
    path, or None if the master could not be found. Same-drive os.replace keeps
    the move atomic."""
    src = find_master(cfg, stem)
    if not src:
        return None
    dest_dir = bucket_dir(cfg, new_level, ensure=True)
    dst = os.path.join(dest_dir, os.path.basename(src))
    if os.path.abspath(src) == os.path.abspath(dst):
        return dst
    try:
        if os.path.exists(dst):
            os.remove(dst)
        os.replace(src, dst)            # atomic within the same drive
        return dst
    except OSError:
        return None


# -- usage state (atomic JSON) --------------------------------------------
def state_path(cfg):
    os.makedirs(cfg.workspace, exist_ok=True)
    return os.path.join(cfg.workspace, STATE_NAME)


def load_state(cfg):
    p = state_path(cfg)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            d.setdefault("clips", {})
            d.setdefault("weights", None)
            d.setdefault("job", None)
            return d
        except Exception:
            pass
    return {"clips": {}, "weights": None, "job": None}


def save_state(cfg, state):
    p = state_path(cfg)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=1)
    os.replace(tmp, p)


# -- retention (pure: returns which stems are due, deletes nothing) --------
def expired_stems(state, retention_days, now=None):
    """Stems whose master was first used into a long video at least
    `retention_days` ago -> their intermediate build artifacts may be deleted."""
    if now is None:
        now = time.time()
    cutoff = now - retention_days * 86400
    due = []
    for stem, info in state.get("clips", {}).items():
        fu = info.get("first_used_at")
        if fu and fu <= cutoff:
            due.append(stem)
    return sorted(due)
