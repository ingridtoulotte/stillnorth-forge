"""Long-video assembler: build multi-hour ambient compilations from the
finished ~11-second masters.

Design (scales to a 12-hour / thousands-of-clip timeline without choking ffmpeg
or memory):

  1. Pick a target duration (15 min ... 12 h).
  2. Choose WHICH distinct masters take part, drawn from the usage buckets by a
     user-set percentage mix (fresh, never-used clips preferred by default),
     randomised within each bucket -- never "the first file in the folder".
  3. Lay them out in a globally randomised timeline, repeating the pool when a
     long target needs more slots than there are masters (never back-to-back
     repeats).
  4. Normalise each distinct master ONCE into a faded MPEG-TS segment (cached on
     disk), then stream-copy the timeline into the final mp4 -- so a 12 h video
     is hundreds of cheap copies, not a multi-hour re-encode.
  5. Bump each used master one usage bucket (move the file) and stamp its
     first-used time for the one-week auto-delete of intermediates.

Crash-safe: the plan is persisted, segments live on disk (resume skips finished
ones), and finalize records which clips it already moved -- so a closed terminal
or a shutdown mid-build resumes cleanly on the next launch.

The selection/planning maths are pure functions (no ffmpeg, GPU or ComfyUI) so
they are unit-tested directly.
"""
import math
import os
import random
import threading
import time

from .config import get_config
from . import library as lib
from . import media

# duration menu, ordered, key -> seconds
DURATIONS = [
    ("15min", 15 * 60), ("30min", 30 * 60), ("1h", 3600), ("2h", 2 * 3600),
    ("3h", 3 * 3600), ("4h", 4 * 3600), ("6h", 6 * 3600), ("8h", 8 * 3600),
    ("10h", 10 * 3600), ("12h", 12 * 3600),
]
DUR_SECONDS = dict(DURATIONS)


# ===================== pure planning helpers =============================
def clips_needed(target_seconds, avg_clip_len):
    """How many ~avg_clip_len clips fill `target_seconds` (at least 1)."""
    avg = max(0.1, float(avg_clip_len))
    return max(1, math.ceil(float(target_seconds) / avg))


def _normalize_weights(weights):
    """Coerce a {level: percent} dict into level(0..MAX)->float fractions that
    sum to 1. Empty / all-zero falls back to all-fresh (level 0)."""
    levels = list(range(0, lib.MAX_BUCKET + 1))
    vals = {}
    for lvl in levels:
        try:
            vals[lvl] = max(0.0, float(weights.get(str(lvl), weights.get(lvl, 0))))
        except (TypeError, ValueError):
            vals[lvl] = 0.0
    total = sum(vals.values())
    if total <= 0:
        return {lvl: (1.0 if lvl == 0 else 0.0) for lvl in levels}
    return {lvl: vals[lvl] / total for lvl in levels}


def select_distinct(pool_by_bucket, weights, k, rng=None):
    """Pick up to `k` distinct stems from the buckets per the weight mix.

    `pool_by_bucket` maps level -> list of stems. Allocation is proportional to
    the weights; any shortfall (a bucket lacks enough clips, or rounding) is
    filled from the remaining clips preferring FRESHER buckets (lower level)
    first. Randomised within every bucket."""
    rng = rng or random
    fr = _normalize_weights(weights)
    levels = list(range(0, lib.MAX_BUCKET + 1))
    avail = {lvl: list(pool_by_bucket.get(lvl, [])) for lvl in levels}
    for lvl in levels:
        rng.shuffle(avail[lvl])
    total_avail = sum(len(avail[lvl]) for lvl in levels)
    want = min(k, total_avail)
    if want <= 0:
        return []

    # proportional allocation, then hand out the rounding remainder to the
    # buckets with the largest fractional part (ties: fresher bucket first)
    raw = {lvl: fr[lvl] * want for lvl in levels}
    alloc = {lvl: min(len(avail[lvl]), int(math.floor(raw[lvl]))) for lvl in levels}
    placed = sum(alloc.values())
    order = sorted(levels, key=lambda l: (-(raw[l] - math.floor(raw[l])), l))
    i = 0
    while placed < want and order:
        lvl = order[i % len(order)]
        if alloc[lvl] < len(avail[lvl]):
            alloc[lvl] += 1
            placed += 1
        i += 1
        if i > 4 * want:                      # safety; remaining filled below
            break

    chosen = []
    for lvl in levels:
        chosen += avail[lvl][:alloc[lvl]]

    # fill any remaining shortfall from leftover clips, freshest first
    if len(chosen) < want:
        taken = set(chosen)
        for lvl in levels:                    # level 0 (freshest) first
            for stem in avail[lvl]:
                if stem not in taken:
                    chosen.append(stem)
                    taken.add(stem)
                    if len(chosen) >= want:
                        break
            if len(chosen) >= want:
                break
    rng.shuffle(chosen)
    return chosen[:want]


def fill_timeline(distinct, k, rng=None):
    """Order `distinct` stems into a length-`k` timeline. If k exceeds the pool
    the pool repeats, but never the same clip twice in a row."""
    rng = rng or random
    distinct = list(distinct)
    if not distinct:
        return []
    if k <= len(distinct):
        seq = distinct[:]
        rng.shuffle(seq)
        return seq[:k]
    timeline = []
    while len(timeline) < k:
        block = distinct[:]
        rng.shuffle(block)
        if timeline and block[0] == timeline[-1] and len(block) > 1:
            block[0], block[1] = block[1], block[0]
        timeline += block
    return timeline[:k]


# ===================== the assembler worker ==============================
class Assembler:
    def __init__(self):
        self.cfg = get_config()
        self.lock = threading.Lock()
        self._thread = None
        self.cancel_flag = False
        self.status = {
            "running": False, "phase": "idle", "percent": 0,
            "done": 0, "total": 0, "note": "", "last_error": None,
            "output": None, "duration": None,
        }

    # -- logging (shares the pipeline's forge.log) -------------------------
    def _log(self, msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [assembler] {msg}"
        try:
            with open(self.cfg.log_path(), "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:
            pass
        print(line, flush=True)

    def set_status(self, **kw):
        with self.lock:
            self.status.update(kw)

    # -- public snapshot for the UI ----------------------------------------
    def snapshot(self):
        st = lib.load_state(self.cfg)
        counts = lib.bucket_counts(self.cfg)
        weights = st.get("weights") or self.cfg.asm_default_weights
        with self.lock:
            status = dict(self.status)
        return {
            "status": status,
            "buckets": counts,
            "weights": weights,
            "durations": [{"key": k, "seconds": s} for k, s in DURATIONS],
            "retention_days": self.cfg.retention_days,
            "autodelete": self.cfg.autodelete,
            "compilations": self._list_compilations(),
        }

    def _list_compilations(self):
        d = lib.compilations_dir(self.cfg)
        out = []
        if os.path.isdir(d):
            for f in sorted(os.listdir(d), reverse=True):
                p = os.path.join(d, f)
                if os.path.isfile(p) and f.lower().endswith(".mp4"):
                    try:
                        stt = os.stat(p)
                    except OSError:
                        continue
                    out.append({"name": f, "path": p, "size": stt.st_size,
                                "mtime": int(stt.st_mtime)})
        return out[:50]

    def save_weights(self, weights):
        st = lib.load_state(self.cfg)
        st["weights"] = weights
        lib.save_state(self.cfg, st)

    # -- start / cancel ----------------------------------------------------
    def start_build(self, duration_key, weights=None):
        if duration_key not in DUR_SECONDS:
            return False, "unknown duration"
        with self.lock:
            if self._thread and self._thread.is_alive():
                return False, "a build is already running"
            self.cancel_flag = False
        st = lib.load_state(self.cfg)
        if weights:
            st["weights"] = weights
        st["job"] = {"duration": duration_key, "phase": "planned",
                     "timeline": None, "distinct": None, "finalized": [],
                     "output": None, "started_at": time.time(),
                     "weights": st.get("weights") or self.cfg.asm_default_weights}
        lib.save_state(self.cfg, st)
        self._spawn()
        return True, "build started"

    def cancel(self):
        self.cancel_flag = True
        self.set_status(note="stopping after current step…")
        self._log("cancel requested")

    def _spawn(self):
        self._thread = threading.Thread(target=self._run_build, daemon=True)
        self._thread.start()

    def maybe_resume(self):
        """On launch, resume a build that a crash/shutdown left unfinished."""
        st = lib.load_state(self.cfg)
        job = st.get("job")
        if job and job.get("phase") not in (None, "done", "cancelled", "error"):
            self.cancel_flag = False
            self._log(f"resuming unfinished build ({job.get('duration')})")
            self._spawn()
            return True
        return False

    # -- the build ---------------------------------------------------------
    def _run_build(self):
        self.set_status(running=True, last_error=None, phase="planning",
                        percent=0, note="planning timeline…")
        try:
            st = lib.load_state(self.cfg)
            job = st.get("job") or {}
            dur_key = job.get("duration")
            target = DUR_SECONDS.get(dur_key, 0)
            self.set_status(duration=dur_key)

            timeline, distinct = self._plan(st, job, target)
            if not timeline:
                self.set_status(running=False, phase="error",
                                last_error="no masters available to assemble",
                                note="render some masters first")
                self._finish_job(st, "error")
                return
            if self.cancel_flag:
                return self._paused(st)

            cache = lib.cache_dir(self.cfg, ensure=True)
            params = self._param_tag()
            seg_for = self._normalize_all(st, distinct, cache, params)
            if self.cancel_flag:
                return self._paused(st)

            out = self._concat(job, timeline, seg_for)
            if self.cancel_flag:
                return self._paused(st)
            if not out:
                self.set_status(running=False, phase="error",
                                last_error="concat failed", note="")
                self._finish_job(st, "error")
                return

            self._finalize_usage(st, distinct)
            self.set_status(running=False, phase="done", percent=100,
                            output=out, note=f"done — {os.path.basename(out)}")
            self._finish_job(st, "done", output=out)
            self._log(f"build complete: {out}")
            # opportunistic retention sweep now that masters were just used
            try:
                self.run_retention()
            except Exception as e:
                self._log(f"retention skipped: {e}")
        except Exception as e:
            import traceback
            self._log("FATAL " + repr(e) + "\n" + traceback.format_exc())
            self.set_status(running=False, phase="error", last_error=str(e), note="")
            try:
                self._finish_job(lib.load_state(self.cfg), "error")
            except Exception:
                pass

    def _paused(self, st):
        self.set_status(running=False, phase="paused",
                        note="paused — press Build to resume")
        self._log("build paused (resumable)")

    def _plan(self, st, job, target):
        """Build (or reuse a persisted) timeline + distinct pool."""
        if job.get("timeline") and job.get("distinct"):
            return job["timeline"], job["distinct"]
        masters = lib.scan(self.cfg)
        pool = {}
        for stem, info in masters.items():
            pool.setdefault(info["level"], []).append(stem)
        if not masters:
            return [], []
        weights = job.get("weights") or st.get("weights") or self.cfg.asm_default_weights
        k = clips_needed(target, self.cfg.asm_clip_len)
        distinct = select_distinct(pool, weights, k)
        timeline = fill_timeline(distinct, k)
        job.update(timeline=timeline, distinct=distinct, phase="normalizing")
        st["job"] = job
        lib.save_state(self.cfg, st)
        self._log(f"plan: target {target}s -> {len(timeline)} slots, "
                  f"{len(distinct)} distinct masters")
        return timeline, distinct

    def _param_tag(self):
        codec = "nvenc" if self.cfg.nvenc else "x264"
        return f"h{self.cfg.asm_height}_f{self.cfg.asm_fade:g}_fps{self.cfg.fps}_{codec}"

    def _seg_path(self, cache, stem, params):
        return os.path.join(cache, f"{stem}__{params}.ts")

    def _normalize_all(self, st, distinct, cache, params):
        """Render each distinct master into a cached faded segment (skip ones
        already on disk so a resumed build doesn't redo finished work)."""
        seg_for = {}
        total = len(distinct)
        self.set_status(phase="normalizing", total=total, done=0, percent=0,
                        note=f"0/{total} clips prepared")
        for i, stem in enumerate(distinct, 1):
            if self.cancel_flag:
                break
            seg = self._seg_path(cache, stem, params)
            seg_for[stem] = seg
            if os.path.exists(seg) and os.path.getsize(seg) > 0:
                self._tick("normalizing", i, total)
                continue
            src = lib.find_master(self.cfg, stem)
            if not src:
                self._log(f"normalize skip (master gone): {stem}")
                continue
            dur = media.probe_duration(self.cfg, src)
            self._remember_duration(st, stem, dur)
            ok = media.normalize_segment(
                self.cfg, src, seg, self.cfg.asm_height, self.cfg.fps,
                self.cfg.asm_fade, dur, cancel=lambda: self.cancel_flag)
            if not ok:
                self._log(f"normalize FAIL {stem}")
            self._tick("normalizing", i, total)
        return seg_for

    def _remember_duration(self, st, stem, dur):
        if dur:
            c = st.setdefault("clips", {}).setdefault(stem, {})
            c["duration"] = dur
            lib.save_state(self.cfg, st)

    def _concat(self, job, timeline, seg_for):
        self.set_status(phase="concatenating", percent=0,
                        note="stitching the long video…", done=0, total=len(timeline))
        cache = lib.cache_dir(self.cfg, ensure=True)
        list_file = os.path.join(cache, "_concat_list.txt")
        lines = []
        for stem in timeline:
            seg = seg_for.get(stem)
            if seg and os.path.exists(seg):
                safe = seg.replace("\\", "/").replace("'", "'\\''")
                lines.append(f"file '{safe}'")
        if not lines:
            return None
        with open(list_file, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        ts = time.strftime("%Y%m%d-%H%M%S")
        out_dir = lib.compilations_dir(self.cfg, ensure=True)
        out = os.path.join(out_dir, f"North-{job.get('duration')}-{ts}.mp4")
        job["output"] = out
        ok = media.concat_copy(self.cfg, list_file, out, cancel=lambda: self.cancel_flag)
        return out if ok else None

    def _finalize_usage(self, st, distinct):
        """Bump every distinct master used into this build one usage bucket
        (move the file) and stamp first-used. Records progress per-clip so a
        crash mid-finalize resumes without double-bumping."""
        self.set_status(phase="finalizing", note="updating usage buckets…")
        job = st.get("job") or {}
        done = set(job.get("finalized", []))
        now = time.time()
        for stem in distinct:
            if stem in done:
                continue
            clip = st.setdefault("clips", {}).setdefault(stem, {})
            uses = int(clip.get("uses", 0)) + 1
            clip["uses"] = uses
            clip.setdefault("first_used_at", now)
            clip["last_used_at"] = now
            lib.move_to_bucket(self.cfg, stem, lib.level_for_uses(uses))
            done.add(stem)
            job["finalized"] = sorted(done)
            st["job"] = job
            lib.save_state(self.cfg, st)

    def _finish_job(self, st, phase, output=None):
        job = st.get("job") or {}
        job["phase"] = phase
        if output:
            job["output"] = output
        st["job"] = job
        lib.save_state(self.cfg, st)

    def _tick(self, phase, done, total):
        pct = round(done / total * 100) if total else 0
        self.set_status(phase=phase, done=done, total=total, percent=pct,
                        note=f"{done}/{total}")

    # -- retention / auto-delete ------------------------------------------
    def run_retention(self, dry_run=False):
        """Delete intermediate build artifacts (FLUX images, first clips, last
        frames, upscales, concats) for masters first used in a long video at
        least `retention_days` ago. The finished masters themselves -- in
        09_final_up4 and the used buckets -- and the Compilations are NEVER
        touched. Returns the number of files removed (or that would be)."""
        if not self.cfg.autodelete and not dry_run:
            return 0
        st = lib.load_state(self.cfg)
        due = lib.expired_stems(st, self.cfg.retention_days)
        removed = 0
        for stem in due:
            for path in self._intermediate_paths(stem):
                if os.path.isfile(path):
                    if dry_run:
                        removed += 1
                        continue
                    try:
                        os.remove(path)
                        removed += 1
                    except OSError as e:
                        self._log(f"retention: could not delete {path}: {e}")
            if not dry_run:
                st.setdefault("clips", {}).setdefault(stem, {})["pruned"] = True
        if removed and not dry_run:
            lib.save_state(self.cfg, st)
            self._log(f"auto-delete: removed {removed} intermediate files "
                      f"for {len(due)} aged masters")
        return removed

    # Intermediate stages whose files are safe to prune. The finished-master
    # folders (final_up + used buckets) and Compilations are deliberately
    # absent from this list so a master can never be deleted.
    _PRUNE_STAGES = ["flux", "img_up", "classified", "vid1",
                     "lastframe", "lf_up", "vid2", "concat"]

    def _intermediate_paths(self, stem):
        """Every intermediate file that belongs to a master stem `<letter>_<key>`.
        The FLUX / x2 stages are named by the bare key (no class letter)."""
        key = stem.split("_", 1)[1] if "_" in stem else stem
        paths = []
        for stage in self._PRUNE_STAGES:
            d = self.cfg.stage_dir(stage, ensure=False)
            base = key if stage in ("flux", "img_up") else stem
            for ext in (".png", ".mp4", ".webm"):
                paths.append(os.path.join(d, base + ext))
        return paths


_ASM = None


def get_assembler():
    global _ASM
    if _ASM is None:
        _ASM = Assembler()
    return _ASM
