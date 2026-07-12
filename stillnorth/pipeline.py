"""StillNorth Forge orchestrator.

Runs the whole HTML -> images -> clips -> long clips pipeline as STAGED GLOBAL
BATCHES: every item clears a stage before the next stage begins, so heavy models
(FLUX.2, Wan 2.2) load once per stage instead of once per prompt.

    extract -> flux -> img_up(x2) -> classify -> vid1 -> lastframe
            -> lf_up(x4) -> vid2(continuation) -> concat -> final_up(x4)

Resumability is driven by the filesystem: each stage skips any item whose output
already exists, exactly like the project's standalone tsn_* scripts. The only
state we must persist is the prompt set, the class letters and the per-clip
pose map (so the continuation clip pushes the camera the SAME direction).

New HTML can be ingested at any time -- the worker re-reads the prompt set on
every pass, so late arrivals flow through from the FLUX stage onward.
"""
import glob
import json
import os
import random
import shutil
import subprocess
import threading
import time
import traceback

from .config import get_config, STAGE_DIRS
from . import comfy as comfymod
from . import judge as judgemod
from . import media
from . import finish
from . import html_prompts
from . import library as lib

# process stage -> output folder key in config.STAGE_DIRS
STAGE_OUT = {
    "flux": "flux", "img_up": "img_up", "classify": "classified",
    "vid1": "vid1", "lastframe": "lastframe", "lf_up": "lf_up",
    "vid2": "vid2", "concat": "concat", "final_up": "final_up",
}
PROCESS = ["flux", "img_up", "classify", "vid1", "lastframe",
           "lf_up", "vid2", "concat", "final_up"]

STAGE_LABELS = {
    "idle": "idle", "done": "done", "extract": "extracting prompts",
    "flux": "image batch", "img_up": "upscaling the flux images",
    "classify": "classifying images", "vid1": "1st vid batch",
    "lastframe": "extracting last frames",
    "lf_up": "upscaling the lastframes (1st vid)",
    "vid2": "2nd vid batch", "concat": "concatenating clips",
    "final_up": "final upscale",
    "judge_flux": "AI-judging flux images",
    "judge_final": "AI-judging finished vids",
}

# chain-completion waves: in target/time-budget mode only this many keys are
# in flight at once, most-advanced first, so chains reach the master stage
# instead of the whole prompt set camping in the flux stage.
WAVE_SIZE = 4


class Pipeline:
    def __init__(self):
        self.cfg = get_config()
        self.comfy = comfymod.Comfy(self.cfg.comfy_server, self.cfg.poll)
        self.lock = threading.Lock()
        self._thread = None
        self.cancel_flag = False
        self.prompts = {}    # key -> {text, src, title}
        self.letters = {}    # key -> "A".."D"
        self.posemap = {}    # stem(<letter>_<key>) -> {letter, pose, speed}
        self.posehints = {}  # key -> [blocked glide dirs from image judge]
        self.motionhints = {}  # key -> bespoke Wan motion clause from image judge
        self.metrics = {}    # stage -> {n, fail, t, min, max}  (per-stage timing)
        self.desired_running = False  # crash-safe intent: auto-resume on launch
        # AI-judge bookkeeping (all persisted)
        self.judged = {}       # key  -> True   (flux still accepted)
        self.judged_gate = {}  # key  -> CV_GATE_VERSION that accepted it
        self.img_rejects = {}  # key  -> reject count so far
        self.vid_judged = {}   # stem -> True/False (master verdict)
        self.vid_rejects = {}  # key  -> master remake count so far
        self.abandoned = []    # keys given up after too many master rejects
        self.html_seen = {}    # basename -> mtime of auto-ingested HTML files
        # run mode: None = classic run-everything; {"target": N} = stop after
        # N accepted masters; {"deadline": epoch} = stop when time is up.
        self.mode = None
        self._active = None    # per-pass active key set (None = no limit)
        self._judge_on = False # judge reachable this pass (checked once)
        self.status = {
            "running": False, "stage": "idle", "label": "idle",
            "percent": 0, "stage_done": 0, "stage_total": 0,
            "last_error": None, "note": "", "cancelled": False,
        }
        self._load_state()

    # -- persistence --------------------------------------------------------
    def _load_state(self):
        p = self.cfg.state_path()
        if os.path.exists(p):
            try:
                d = json.load(open(p, "r", encoding="utf-8"))
                self.prompts = d.get("prompts", {})
                self.letters = d.get("letters", {})
                self.posemap = d.get("posemap", {})
                self.posehints = d.get("posehints", {})
                self.motionhints = d.get("motionhints", {})
                self.metrics = d.get("metrics", {})
                self.desired_running = bool(d.get("desired_running", False))
                self.judged = d.get("judged", {})
                self.judged_gate = d.get("judged_gate", {})
                self.img_rejects = d.get("img_rejects", {})
                self.vid_judged = d.get("vid_judged", {})
                self.vid_rejects = d.get("vid_rejects", {})
                self.abandoned = d.get("abandoned", [])
                self.html_seen = d.get("html_seen", {})
                self.mode = d.get("mode")
            except Exception:
                pass

    def _save_state(self):
        """Atomic write: serialise to a temp file then os.replace() so a crash
        mid-write can never leave a half-written (corrupt) state file."""
        p = self.cfg.state_path()
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"prompts": self.prompts, "letters": self.letters,
                       "posemap": self.posemap, "posehints": self.posehints,
                       "motionhints": self.motionhints,
                       "metrics": self.metrics,
                       "desired_running": self.desired_running,
                       "judged": self.judged, "judged_gate": self.judged_gate,
                       "img_rejects": self.img_rejects,
                       "vid_judged": self.vid_judged,
                       "vid_rejects": self.vid_rejects,
                       "abandoned": self.abandoned,
                       "html_seen": self.html_seen,
                       "mode": self.mode}, fh, indent=1)
        os.replace(tmp, p)

    def _log(self, msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        try:
            with open(self.cfg.log_path(), "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:
            pass
        try:
            print(line, flush=True)
        except UnicodeEncodeError:
            # Windows cp1252 console can't print ✓/em-dash etc — degrade,
            # never let a log line kill the worker thread.
            print(line.encode("ascii", "replace").decode(), flush=True)

    # -- public API ---------------------------------------------------------
    def ingest_html(self, name, text):
        """Parse one HTML payload, add its prompts to the queue. Returns count."""
        objs = html_prompts.extract_prompts_from_text(text)
        added = 0
        with self.lock:
            for obj in objs:
                k = html_prompts.prompt_key(obj)
                if k in self.prompts:
                    continue
                self.prompts[k] = {
                    "text": html_prompts.prompt_to_text(obj),
                    "src": name,
                    "title": obj.get("title") or obj.get("scene", "")[:50],
                }
                added += 1
            self._save_state()
        self._log(f"ingest {name}: +{added} prompts (total {len(self.prompts)})")
        return added, len(objs)

    def set_status(self, **kw):
        with self.lock:
            self.status.update(kw)

    def snapshot(self):
        with self.lock:
            st = dict(self.status)
            st["totals"] = {"prompts": len(self.prompts)}
            st["counts"] = {s: self._count(s) for s in PROCESS}
            st["queue"] = self._queue_view()
            st["cancel"] = self.cancel_flag
            st["metrics"] = self._metrics_view()
            st["desired_running"] = self.desired_running
            m = self.mode or {}
            st["mode"] = {
                "kind": ("target" if m.get("target")
                         else "time" if m.get("deadline") else "all"),
                "target": m.get("target"),
                "minutes": m.get("minutes"),
                "seconds_left": (max(0, int(m["deadline"] - time.time()))
                                 if m.get("deadline") else None),
                "accepted": self.accepted_count(),
                # accepted_new = masters this run has added past its baseline;
                # the target counts THESE, not the raw catalog total.
                "accepted_new": (self.accepted_count() - m.get("baseline", 0)
                                 if m.get("target") else None),
                "review": self._count("review"),
                "abandoned": len(self.abandoned),
                "judge": bool(self.cfg.judge_enabled),
            }
        try:
            st["library"] = lib.bucket_counts(self.cfg)   # masters incl. used buckets
        except Exception:
            st["library"] = None
        return st

    def _metrics_view(self):
        out = {}
        for stage, m in self.metrics.items():
            n = m.get("n", 0)
            out[stage] = {
                "n": n, "fail": m.get("fail", 0),
                "avg": round(m["t"] / n, 1) if n else None,
                "min": round(m["min"], 1) if m.get("min") is not None else None,
                "max": round(m["max"], 1) if m.get("max") is not None else None,
            }
        return out

    def _record(self, stage, dt, ok):
        """Accumulate per-stage timing + failure counts (for the metrics UI)."""
        m = self.metrics.setdefault(
            stage, {"n": 0, "fail": 0, "t": 0.0, "min": None, "max": None})
        if ok:
            m["n"] += 1
            m["t"] += dt
            m["min"] = dt if m["min"] is None else min(m["min"], dt)
            m["max"] = dt if m["max"] is None else max(m["max"], dt)
        else:
            m["fail"] += 1

    def _queue_view(self):
        by_src = {}
        for v in self.prompts.values():
            by_src[v["src"]] = by_src.get(v["src"], 0) + 1
        return [{"src": k, "prompts": n} for k, n in sorted(by_src.items())]

    def start(self, target=None, minutes=None, keep_mode=False):
        """Start the worker if not already running (idempotent).

        target  — stop once this many masters have been produced AND accepted
                  by the AI judge (rejected chains are remade with new prompts
                  or new seeds until the count is reached or prompts run out).
        minutes — time budget: produce as many accepted masters as possible,
                  then stop (the item being rendered is finished, not killed).
        Neither — classic behaviour: process every queued prompt.
        keep_mode — auto-resume path: reuse the persisted mode instead of
                  clobbering it to classic full-set (a target=3 run that got
                  restarted mid-flight once resumed as "render everything"
                  and burned the GPU on 48 unwanted flux renders).
        """
        with self.lock:
            if self._thread and self._thread.is_alive():
                return False
            if keep_mode:
                pass                     # self.mode already holds the run's mode
            elif target:
                # target = how many NEW masters this run should add, not the
                # raw catalog total. Snapshot the already-accepted count as a
                # baseline so "5" means "5 more" even when 28 already exist
                # (user complaint: "the number isn't the total, it's how many
                # MORE I want"). Baseline lives in self.mode so it persists and
                # survives a keep_mode auto-resume for free.
                self.mode = {"target": int(target),
                             "baseline": self.accepted_count()}
            elif minutes:
                self.mode = {"deadline": time.time() + float(minutes) * 60,
                             "minutes": float(minutes)}
            else:
                self.mode = None
            self.cancel_flag = False
            self.desired_running = True       # remember intent across restarts
            self._save_state()
            self.status.update(cancelled=False, last_error=None, note="starting…")
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            return True

    # -- mode helpers --------------------------------------------------------
    def _time_up(self):
        m = self.mode
        return bool(m and m.get("deadline") and time.time() >= m["deadline"])

    def _should_stop(self):
        """Checked between items inside every stage loop: an explicit pause
        and an expired time budget both stop AFTER the current item."""
        return self.cancel_flag or self._time_up()

    def accepted_count(self):
        """Masters produced AND accepted. With the video judge on, only
        judged-OK masters count — and only while the master still exists
        (in 09_final_up4 or moved into a library bucket). A stale accept
        for a manually archived/deleted master once made target mode
        declare victory without rendering a replacement."""
        if self.cfg.judge_enabled and self.cfg.judge_video_enabled:
            return sum(1 for stem, v in self.vid_judged.items()
                       if v and (self._exists("final_up", stem, video=True)
                                 or lib.master_exists(self.cfg, stem)))
        return self._count("final_up")

    def _target_progress(self):
        """(new_accepted_this_run, target) for target mode, else (0, 0).
        new_accepted subtracts the baseline captured at start() so a run's
        progress is counted from where it began, not from an empty catalog."""
        m = self.mode
        if not (m and m.get("target")):
            return 0, 0
        return self.accepted_count() - m.get("baseline", 0), m["target"]

    def _target_reached(self):
        done, target = self._target_progress()
        return bool(target and done >= target)

    def _key_progress(self, key):
        """How far key's chain has got (higher = closer to a master)."""
        stem = f"{self.letters.get(key, '')}_{key}" if key in self.letters else None
        if stem:
            for rank, stage in ((6, "final_up"), (5, "concat"), (4, "vid2"),
                                (3, "vid1")):
                if self._exists(stage, stem, video=True):
                    return rank
            if self._classified_exists(key):
                return 2
        if self._exists("img_up", key):
            return 2
        if self._exists("flux", key):
            return 1
        return 0

    def _key_finished(self, key):
        stem = f"{self.letters.get(key, '')}_{key}"
        if key in self.letters and self.vid_judged.get(stem):
            return True
        if not (self.cfg.judge_enabled and self.cfg.judge_video_enabled):
            return (key in self.letters and
                    (self._exists("final_up", stem, video=True) or
                     lib.master_exists(self.cfg, stem)))
        return False

    def _pick_active(self):
        """Per-pass active key set. None = no limit (classic mode). In target
        mode the set holds exactly the still-needed number of chains; in
        time-budget mode a small wave, so chains finish instead of the whole
        prompt set camping in the flux stage. Most-advanced chains first."""
        if not self.mode:
            return None
        if self.mode.get("target"):
            done, target = self._target_progress()
            n = max(0, target - done)
        else:
            n = WAVE_SIZE
        cand = [k for k in self.prompts
                if k not in self.abandoned and not self._key_finished(k)]
        cand.sort(key=self._key_progress, reverse=True)
        return set(cand[:n])

    def _allowed(self, key):
        return self._active is None or key in self._active

    def maybe_auto_resume(self):
        """Called once on launch. If the batch was running when the terminal
        was closed or the PC shut down (intent persisted, work still pending),
        resume it automatically -- no need to press Run again."""
        if self.desired_running and self._pending_work():
            m = self.mode or {}
            if m.get("deadline") and time.time() >= m["deadline"]:
                self.desired_running = False   # budget already spent
                with self.lock:
                    self._save_state()
                return False
            self._log("auto-resume: batch was running before shutdown — resuming")
            return self.start(keep_mode=True)
        return False

    def _active_stages(self):
        """Producing stages that actually run in the current mode, in order.
        native_overlap + native_long both skip the still-seed stages."""
        if self.cfg.native_long or self.cfg.single_clip:
            return ["img_up", "vid1", "concat", "final_up"]
        if self.cfg.continuation_mode == "native_overlap":
            return ["img_up", "vid1", "vid2", "concat", "final_up"]
        return ["img_up", "vid1", "lastframe", "lf_up", "vid2", "concat", "final_up"]

    def _pending_work(self):
        """True if any stage still has an item to render (drives auto-resume).
        Walks only the ACTIVE stages so skipped ones never read as 'pending'."""
        if any(not self._exists("flux", k) for k in self.prompts):
            return True
        prev = self._count("flux")
        for stage in self._active_stages():
            if self._count(stage) < prev:
                return True
            prev = self._count(stage)
        return self._count("final_up") < len(self.prompts)

    def cancel(self):
        """Pause: stop after the current item; rendered files are kept so a
        later Run resumes exactly where it stopped. An explicit pause clears the
        auto-resume intent (only an unexpected shutdown should auto-resume)."""
        self.cancel_flag = True
        self.desired_running = False
        with self.lock:
            self._save_state()
        self.set_status(cancelled=True, note="finishing current item, then pausing…")
        self.comfy.interrupt()
        self._log("cancel requested")

    def clear_queue(self):
        """Forget queued prompts (does NOT delete already-rendered media)."""
        self.cancel_flag = True
        self.desired_running = False
        self.comfy.interrupt()
        with self.lock:
            self.prompts = {}
            self.letters = {}
            self.posemap = {}
            self.posehints = {}
            self.motionhints = {}
            self.judged = {}
            self.judged_gate = {}
            self.img_rejects = {}
            self.vid_judged = {}
            self.vid_rejects = {}
            self.abandoned = []
            self.html_seen = {}   # files still in the folder re-ingest on Run
            self.mode = None
            self._save_state()
        self._log("queue cleared")

    def purge_outputs(self):
        """DESTRUCTIVE. Stop the worker, delete every rendered stage folder and
        the persisted state, and reset to a clean slate so the pipeline counts
        match a fresh start. forge.log is kept. Rendered media is NOT
        recoverable afterwards. Returns the number of stage folders removed."""
        self.cancel_flag = True
        self.desired_running = False
        self.comfy.interrupt()
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=15)            # let the worker release any open files
        removed = 0
        for key in STAGE_DIRS:
            d = self.cfg.stage_dir(key, ensure=False)
            if os.path.isdir(d):
                try:
                    shutil.rmtree(d)
                    removed += 1
                except Exception as e:
                    self._log(f"purge: could not remove {d}: {e}")
        with self.lock:
            self.prompts = {}
            self.letters = {}
            self.posemap = {}
            self.posehints = {}
            self.motionhints = {}
            self.metrics = {}
            self.judged = {}
            self.judged_gate = {}
            self.img_rejects = {}
            self.vid_judged = {}
            self.vid_rejects = {}
            self.abandoned = []
            self.html_seen = {}
            self.mode = None
            try:
                sp = self.cfg.state_path()
                if os.path.exists(sp):
                    os.remove(sp)
            except OSError:
                pass
            self.status.update(stage="idle", label="idle", percent=0,
                               stage_done=0, stage_total=0, note="",
                               cancelled=False, last_error=None)
        self._log(f"purge: removed {removed} stage folders + state (clean slate)")
        return removed

    # -- helpers ------------------------------------------------------------
    def _count(self, stage):
        d = self.cfg.stage_dir(STAGE_OUT.get(stage, stage), ensure=False)
        if not os.path.isdir(d):
            return 0
        n = len(glob.glob(os.path.join(d, "*.png")))
        n += len(glob.glob(os.path.join(d, "*.mp4")))
        n += len(glob.glob(os.path.join(d, "*.webm")))
        return n

    def _exists(self, stage, stem, video=False):
        d = self.cfg.stage_dir(STAGE_OUT[stage], ensure=False)
        if video:
            return (os.path.exists(os.path.join(d, stem + ".mp4")) or
                    os.path.exists(os.path.join(d, stem + ".webm")))
        return os.path.exists(os.path.join(d, stem + ".png"))

    def _clip_path(self, stage, stem):
        d = self.cfg.stage_dir(STAGE_OUT[stage], ensure=False)
        for ext in (".mp4", ".webm"):
            p = os.path.join(d, stem + ext)
            if os.path.exists(p):
                return p
        return None

    def _load_wf(self, which):
        return json.load(open(self.cfg.workflow_path(which), encoding="utf-8"))

    # -- the run loop -------------------------------------------------------
    def _run(self):
        self.set_status(running=True, last_error=None, note="scanning for work…")
        self._log("pipeline run started" +
                  (f" (mode {self.mode})" if self.mode else ""))
        try:
            note = "all stages complete ✓"
            while not self.cancel_flag:
                if self._target_reached():
                    done, target = self._target_progress()
                    note = (f"target reached ✓ — {done} new master(s) this run "
                            f"({self.accepted_count()} in catalog)")
                    break
                if self._time_up():
                    note = (f"time budget spent — {self.accepted_count()} "
                            "accepted masters produced")
                    break
                work = self._one_pass()
                if work == 0:
                    if self.mode and self.mode.get("target") and \
                            not self._target_reached():
                        done, target = self._target_progress()
                        note = (f"prompt set exhausted at "
                                f"{done}/{target} new masters this run — "
                                "drop more HTML prompt files in")
                    break
            if self.cancel_flag:
                self.set_status(running=False, stage="idle", label="paused",
                                percent=0, stage_done=0, stage_total=0,
                                cancelled=True,
                                note="paused — press Run to resume where it stopped")
            else:
                self.desired_running = False  # finished cleanly; nothing to resume
                with self.lock:
                    self._save_state()
                self._log("run complete: " + note)
                self.set_status(running=False, stage="done", label="done",
                                percent=100, stage_done=0, stage_total=0,
                                cancelled=False, note=note)
        except Exception as e:
            self._log("FATAL " + repr(e) + "\n" + traceback.format_exc())
            self.set_status(running=False, stage="idle", label="error",
                            percent=0, stage_done=0, stage_total=0,
                            last_error=str(e), note="")
        self._log("pipeline run finished")

    def _one_pass(self):
        self._scan_html_dir()
        self._active = self._pick_active()
        self._judge_on = (self.cfg.judge_enabled and
                          judgemod.available(self.cfg))
        if self.cfg.judge_enabled and not self._judge_on:
            self.set_status(note="AI judge unreachable — rendering unjudged")
        done = 0
        done += self._stage_flux()
        done += self._stage_judge_flux()
        done += self._stage_img_up()
        done += self._stage_classify()
        done += self._stage_vid1()
        done += self._stage_lastframe()
        done += self._stage_lf_up()
        done += self._stage_vid2()
        done += self._stage_concat()
        done += self._stage_final_up()
        done += self._stage_judge_final()
        return done

    #: prompt-file extensions auto-ingested from the drop folder. The parser
    #: (html_prompts.extract_prompts_from_text) is format-agnostic — it scans
    #: raw text for any balanced {...} block with a "scene" key — so a plain
    #: .txt or .json file holding {"scene": "..."} works exactly like an HTML
    #: export. .html/.htm kept for backward compatibility with FLUX exports.
    PROMPT_GLOBS = ("*.html", "*.htm", "*.txt", "*.json")

    def _scan_html_dir(self):
        """Auto-ingest prompt files dropped into the folder (new or edited
        files only, tracked by mtime) — no UI upload needed. Accepts .html/
        .htm exports AND hand-written .txt/.json holding a {"scene": ...}
        object; see PROMPT_GLOBS."""
        d = self.cfg.html_dir
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:
            return
        paths = {p for pat in self.PROMPT_GLOBS
                 for p in glob.glob(os.path.join(d, pat))}
        for p in sorted(paths):
            name = os.path.basename(p)
            try:
                mt = int(os.path.getmtime(p))
            except OSError:
                continue
            if self.html_seen.get(name) == mt:
                continue
            try:
                with open(p, "r", encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            self.ingest_html(name, text)
            with self.lock:
                self.html_seen[name] = mt
                self._save_state()

    def _begin(self, stage, total):
        self.set_status(stage=stage, label=STAGE_LABELS[stage],
                        stage_done=0, stage_total=total, percent=0,
                        note=f"0/{total} done — starting…")

    def _working(self, stage, idx, total):
        """Mark item `idx` (1-based) as in-flight so the UI shows life while a
        slow FLUX/Wan render is actually running (the bar would otherwise sit
        frozen at the last completed count for ~45s+ per image)."""
        pct = round((idx - 1) / total * 100) if total else 0
        self.set_status(stage=stage, label=STAGE_LABELS[stage],
                        stage_done=idx - 1, stage_total=total, percent=pct,
                        note=f"rendering {idx}/{total}…")

    def _tick(self, stage, done, total):
        pct = round(done / total * 100) if total else 100
        self.set_status(stage=stage, label=STAGE_LABELS[stage],
                        stage_done=done, stage_total=total, percent=pct,
                        note=f"{done}/{total} done")

    # -- ComfyUI submit (one item, with retry + backoff + metrics) ----------
    @staticmethod
    def _categorize(msg):
        """Map a raw failure into a coarse category so the UI/log can tell an
        offline-ComfyUI apart from a workflow/GPU/codec problem."""
        m = (msg or "").lower()
        if "unreach" in m or "urlerror" in m or "refused" in m or "connection" in m:
            return "comfy offline"
        if "timeout" in m:
            return "render timeout (GPU/slow)"
        if "comfy error" in m or "workflow" in m or "node" in m:
            return "workflow error"
        if "no output" in m:
            return "no output (workflow)"
        return msg or "unknown"

    def _backoff(self, attempt):
        """Sleep retry_backoff * 2**attempt (capped 60s), but wake early on cancel."""
        delay = min(self.cfg.retry_backoff * (2 ** attempt), 60.0)
        end = time.time() + delay
        while time.time() < end and not self.cancel_flag:
            time.sleep(0.25)

    def _submit(self, wf, timeout, stage, reseed=None):
        attempts = self.cfg.submit_retries + 1
        t0 = time.time()
        last = "queue failed"
        for a in range(attempts):
            if self.cancel_flag:
                return False, "cancelled"
            if a and reseed:
                # A re-queued graph identical to a finished one is
                # cache-collapsed by ComfyUI into a completed history with
                # no outputs -> every retry instantly fails "no output".
                reseed(wf)
            try:
                pid = self.comfy.queue(wf)
            except Exception as e:
                last = self._categorize(repr(e))
                self.set_status(last_error=f"ComfyUI: {last}")
                self._log(f"{stage} submit error (try {a + 1}/{attempts}): {last}")
                self._backoff(a)
                continue
            ok, msg = self.comfy.wait(pid, timeout, cancel=lambda: self.cancel_flag)
            if ok:
                self._record(stage, time.time() - t0, True)
                return True, msg
            if msg == "cancelled":
                return False, "cancelled"
            if msg == "timeout":
                # The job is still running server-side after a wait timeout:
                # left alone it keeps the GPU for up to ~40 more minutes and
                # the retry just queues behind it. Free the GPU first.
                self.comfy.interrupt()
            last = self._categorize(msg)
            self._log(f"{stage} render failed (try {a + 1}/{attempts}): {last}")
            self._backoff(a)
        self._record(stage, time.time() - t0, False)
        return False, last

    def _salvage_render(self, stage, stem, expect_frames=0):
        """A completed render may still land on disk after _submit failed:
        a timed-out job can finish server-side after we stopped waiting
        (its output keeps ComfyUI's _00001_ suffix, never renamed), and a
        cache-collapsed retry reports "no output" while the original file
        exists. Rescue it instead of re-rendering the whole clip."""
        p = comfymod.rename_out(self.cfg.stage_dir(stage), stem,
                                (".mp4", ".webm"))
        if not p:
            return False
        if expect_frames:
            n = self._clip_nframes(p)
            if n < expect_frames - 1:
                try:
                    os.remove(p)
                except OSError:
                    pass
                self._log(f"{stage} salvage {stem}: incomplete "
                          f"({n}/{expect_frames} frames) — discarded")
                return False
        self._log(f"{stage} salvaged completed render {stem}")
        return True

    # ---- STAGE 1: FLUX images --------------------------------------------
    def _stage_flux(self):
        with self.lock:
            items = [(k, v["text"]) for k, v in self.prompts.items()]
        todo = [(k, t) for k, t in items
                if not self._exists("flux", k) and self._allowed(k)
                and k not in self.abandoned]
        if not todo:
            return 0
        wf = self.cfg.workflows["flux"]
        base = self._load_wf("flux")
        out = self.cfg.stage_dir("flux")
        self._begin("flux", len(todo))
        done = 0
        for k, text in todo:
            if self._should_stop():
                break
            self._working("flux", done + 1, len(todo))
            g = json.loads(json.dumps(base))
            g[wf["node_text"]]["inputs"][wf["field_text"]] = self.cfg.flux_text(text)
            g[wf["node_seed"]]["inputs"][wf["field_seed"]] = random.randint(0, 2**31 - 1)
            g[wf["node_save"]]["inputs"][wf["field_save"]] = self.cfg.comfy_prefix("flux", k)
            ok, msg = self._submit(g, self.cfg.timeout_img, "flux")
            if ok and comfymod.rename_out(out, k, (".png",)):
                done += 1
            else:
                self._log(f"flux FAIL {k}: {msg}")
            self._tick("flux", done, len(todo))
        return done

    # ---- STAGE 1b: AI-judge the new FLUX stills ---------------------------
    def _stage_judge_flux(self):
        """Strict local-AI verdict on every not-yet-judged FLUX still. A
        reject deletes the image (plus any downstream copies) so the next
        pass regenerates it with a fresh seed; after max_image_retries the
        last render is accepted so one stubborn prompt cannot stall the run."""
        if not self._judge_on:
            return 0
        src_dir = self.cfg.stage_dir("flux", ensure=False)
        self._regate_stale_stills(src_dir)
        todo = [k for k in self._keys_with_png(src_dir)
                if k not in self.judged and self._allowed(k)]
        if not todo:
            return 0
        self._begin("judge_flux", len(todo))
        done = 0
        for k in todo:
            if self._should_stop():
                break
            self._working("judge_flux", done + 1, len(todo))
            path = os.path.join(src_dir, k + ".png")
            ok, reason, blocked, motion = judgemod.judge_image_ex(self.cfg, path)
            with self.lock:
                if ok:
                    self.judged[k] = True
                    self.judged_gate[k] = judgemod.CV_GATE_VERSION
                    # remember which glide directions the judge ruled out so
                    # the camera stage never dollies/pans into a near mass,
                    # plus the bespoke per-image motion clause it wrote (''
                    # when denylisted/absent -> class default is used)
                    self.posehints[k] = blocked
                    self.motionhints[k] = motion
                else:
                    n = self.img_rejects.get(k, 0) + 1
                    self.img_rejects[k] = n
                    if n > self.cfg.judge_image_retries:
                        if self.cfg.judge_giveup == "accept":
                            self.judged[k] = True
                            self.judged_gate[k] = judgemod.CV_GATE_VERSION
                            self._log(f"judge_flux {k}: accepted after {n - 1} "
                                      f"rejects (giving up) — last: {reason}")
                        else:
                            # quality-first: a source that failed every retry
                            # would cost ~14 min of doomed Wan renders — drop
                            # the key and let the run pull the next prompt.
                            if k not in self.abandoned:
                                self.abandoned.append(k)
                            self._delete_key_files(k, flux_too=True)
                            self._log(f"judge_flux {k}: abandoned after "
                                      f"{n - 1} rejects — last: {reason}")
                    else:
                        self._delete_key_files(k, flux_too=True)
                        self._log(f"judge_flux REJECT {k} "
                                  f"({n}/{self.cfg.judge_image_retries}): "
                                  f"{reason} — regenerating")
                self._save_state()
            done += 1
            self._tick("judge_flux", done, len(todo))
        return done

    def _delete_key_files(self, key, flux_too=False):
        """Remove a key's rendered artefacts so the filesystem-resume logic
        regenerates them. flux_too also drops the source still."""
        stem = f"{self.letters.get(key, '')}_{key}"
        for stage, name, exts in (
                ("final_up", stem, (".mp4", ".webm")),
                ("concat", stem, (".mp4", ".webm")),
                ("vid2", stem, (".mp4", ".webm")),
                ("vid1", stem, (".mp4", ".webm")),
                ("classified", stem, (".png",)),
                ("img_up", key, (".png",)),
                ("flux", key, (".png",)) if flux_too else (None, None, ())):
            if not stage:
                continue
            d = self.cfg.stage_dir(stage, ensure=False)
            for ext in exts:
                p = os.path.join(d, name + ext)
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except OSError as e:
                    self._log(f"judge cleanup: could not remove {p}: {e}")
        # bookkeeping tied to the dead chain (caller holds the lock)
        self.judged.pop(key, None)
        self.judged_gate.pop(key, None)
        self.vid_judged.pop(stem, None)
        self.posemap.pop(stem, None)
        self.posehints.pop(key, None)
        self.motionhints.pop(key, None)
        if flux_too:
            self.letters.pop(key, None)

    def _regate_stale_stills(self, src_dir):
        """Re-run the deterministic CV gate on stills accepted under an OLDER
        CV_GATE_VERSION (e.g. sources judged before struct_ratio existed). A
        persisted judged=True would otherwise grandfather such a source past a
        gate added after it was first judged — exactly how dense-micro-texture
        "240p mush" masters shipped after the struct_ratio gate went live but
        their pre-gate accepts were never re-checked. CV-only: no VLM re-call,
        so it is cheap and repeatable (a re-VLM could spuriously flip a good
        still). A stale still that now fails the gate has its whole chain
        dropped so a fresh, gate-passing source regenerates in its place."""
        stale = [k for k in list(self.judged)
                 if self.judged.get(k)
                 and self.judged_gate.get(k, 0) < judgemod.CV_GATE_VERSION
                 and os.path.exists(os.path.join(src_dir, k + ".png"))]
        for k in stale:
            ok, reason = judgemod.cv_gate(
                self.cfg, os.path.join(src_dir, k + ".png"))
            with self.lock:
                if ok:
                    self.judged_gate[k] = judgemod.CV_GATE_VERSION
                else:
                    self._log(f"regate DROP {k}: pre-gate accept fails current "
                              f"CV gate (v{judgemod.CV_GATE_VERSION}) — {reason}")
                    self._delete_key_files(k, flux_too=True)
                self._save_state()

    # ---- STAGE 9b: AI-judge the finished masters --------------------------
    def _stage_judge_final(self):
        """3 frames at a random point of each new master -> coherence verdict.
        A reject moves the master to 10_review (kept for human eyes) and
        deletes the whole chain incl. the FLUX still, so a completely fresh
        chain regenerates; after max_video_retries the key is abandoned."""
        if not (self._judge_on and self.cfg.judge_video_enabled):
            return 0
        d = self.cfg.stage_dir("final_up", ensure=False)
        todo = [s for s in self._stems_with_clip(d)
                if s not in self.vid_judged
                and self._allowed(s.split("_", 1)[1])]
        if not todo:
            return 0
        review = self.cfg.stage_dir("review")
        self._begin("judge_final", len(todo))
        done = 0
        for s in todo:
            if self._should_stop():
                break
            self._working("judge_final", done + 1, len(todo))
            path = self._clip_path("final_up", s)
            ok, reason = judgemod.judge_video(self.cfg, path)
            with self.lock:
                if ok:
                    self.vid_judged[s] = True
                    self._log(f"judge_final ACCEPT {s}: {reason or 'clean'}")
                else:
                    key = s.split("_", 1)[1]
                    n = self.vid_rejects.get(key, 0) + 1
                    self.vid_rejects[key] = n
                    dst = os.path.join(
                        review, f"{s}_rej{n}" + os.path.splitext(path)[1])
                    try:
                        shutil.move(path, dst)
                    except OSError as e:
                        self._log(f"judge_final: review move failed {s}: {e}")
                    self._delete_key_files(key, flux_too=True)
                    if n > self.cfg.judge_video_retries:
                        if key not in self.abandoned:
                            self.abandoned.append(key)
                        self._log(f"judge_final REJECT {s}: {reason} — "
                                  f"{n} strikes, key abandoned (in 10_review)")
                    else:
                        self._log(f"judge_final REJECT {s} "
                                  f"({n}/{self.cfg.judge_video_retries}): "
                                  f"{reason} — moved to review, chain remade")
                self._save_state()
            done += 1
            self._tick("judge_final", done, len(todo))
        return done

    # ---- STAGE 2: image upscale x2 ---------------------------------------
    def _stage_img_up(self):
        src_dir = self.cfg.stage_dir("flux", ensure=False)
        todo = [k for k in self._keys_with_png(src_dir)
                if not self._exists("img_up", k) and self._allowed(k)
                and (not self._judge_on or k in self.judged)]
        if not todo:
            return 0
        out = self.cfg.stage_dir("img_up")
        self._begin("img_up", len(todo))
        done = 0
        for k in todo:
            if self._should_stop():
                break
            self._working("img_up", done + 1, len(todo))
            src = os.path.join(src_dir, k + ".png")
            dst = os.path.join(out, k + ".png")
            if media.upscale_image(self.cfg, src, dst, self.cfg.img_mult):
                done += 1
            else:
                self._log(f"img_up FAIL {k}")
            self._tick("img_up", done, len(todo))
        return done

    # ---- STAGE 3: classify -> letter-prefixed copies ---------------------
    def _stage_classify(self):
        src_dir = self.cfg.stage_dir("img_up", ensure=False)
        keys = [k for k in self._keys_with_png(src_dir)
                if (k not in self.letters or not self._classified_exists(k))
                and self._allowed(k)]
        if not keys:
            return 0
        from . import classify
        out = self.cfg.stage_dir("classified")
        files = [os.path.join(src_dir, k + ".png") for k in keys]
        self._begin("classify", len(files))
        try:
            results = classify.classify_all(
                files, progress=lambda d, t: self._tick("classify", d, t))
        except Exception as e:
            self.set_status(last_error=str(e))
            self._log("classify FAIL " + repr(e))
            return 0
        done = 0
        for path, cls, _ in results:
            k = os.path.splitext(os.path.basename(path))[0]
            self.letters[k] = cls
            dst = os.path.join(out, f"{cls}_{k}.png")
            try:
                if not os.path.exists(dst):
                    shutil.copy2(path, dst)
                done += 1
            except Exception as e:
                self._log(f"classify copy FAIL {k}: {e}")
        with self.lock:
            self._save_state()
        self._tick("classify", done, len(files))
        return done

    def _classified_exists(self, k):
        cls = self.letters.get(k)
        if not cls:
            return False
        return os.path.exists(os.path.join(
            self.cfg.stage_dir("classified", ensure=False), f"{cls}_{k}.png"))

    # ---- STAGE 4: first Wan clips ----------------------------------------
    def _stage_vid1(self):
        src_dir = self.cfg.stage_dir("classified", ensure=False)
        stems = [s for s in self._stems_with_png(src_dir)
                 if not self._exists("vid1", s, video=True)
                 and self._allowed(s.split("_", 1)[1])]
        if not stems:
            return 0
        wf = self.cfg.workflows["wan"]
        base = self._load_wf("wan")
        out = self.cfg.stage_dir("vid1")
        self._begin("vid1", len(stems))
        done = 0
        for s in stems:
            if self._should_stop():
                break
            self._working("vid1", done + 1, len(stems))
            letter, key = s.split("_", 1)[0], s.split("_", 1)[1]
            # weighted, obstacle-aware direction (not uniform random): ~50%
            # {up, dolly-back} vs 50% {dolly-fwd, left, right}, minus any
            # direction the image judge flagged as blocked by a near mass.
            pose, speed = self.cfg.choose_pose(self.posehints.get(key, []))
            self.posemap[s] = {"letter": letter, "pose": pose, "speed": speed}
            # native_long renders the whole ~10s clip here in one pass
            length = self.cfg.native_frames if self.cfg.native_long else None
            if self._wan_clip(base, wf, src_dir, s, letter, pose, speed, "vid1",
                              length=length):
                done += 1
                with self.lock:
                    self._save_state()
            self._tick("vid1", done, len(stems))
        return done

    # ---- STAGE 5: last frame of each first clip --------------------------
    def _stage_lastframe(self):
        if self.cfg.native_long or self.cfg.single_clip:
            return 0                  # one clip -> no continuation chain
        if self.cfg.continuation_mode == "native_overlap":
            return 0                  # continuation reads frames from clip1 directly
        src_dir = self.cfg.stage_dir("vid1", ensure=False)
        stems = [s for s in self._stems_with_clip(src_dir)
                 if not self._exists("lastframe", s)]
        if not stems:
            return 0
        out = self.cfg.stage_dir("lastframe")
        self._begin("lastframe", len(stems))
        done = 0
        for s in stems:
            if self._should_stop():
                break
            self._working("lastframe", done + 1, len(stems))
            clip = self._clip_path("vid1", s)
            dst = os.path.join(out, s + ".png")
            if clip and media.last_frame(self.cfg, clip, dst):
                done += 1
            else:
                self._log(f"lastframe FAIL {s}")
            self._tick("lastframe", done, len(stems))
        return done

    # ---- STAGE 6: last-frame upscale x4 ----------------------------------
    def _stage_lf_up(self):
        # In "native" continuation mode the upscaled last frame is not used as
        # the clip-2 seed (Wan resizes the seed back to 832x480 anyway, so the
        # x4 lanczos only round-trips a blur). Skip the stage entirely.
        if self.cfg.native_long or self.cfg.cont_seed == "native":
            return 0
        src_dir = self.cfg.stage_dir("lastframe", ensure=False)
        stems = [s for s in self._stems_with_png(src_dir)
                 if not self._exists("lf_up", s)]
        if not stems:
            return 0
        out = self.cfg.stage_dir("lf_up")
        self._begin("lf_up", len(stems))
        done = 0
        for s in stems:
            if self._should_stop():
                break
            self._working("lf_up", done + 1, len(stems))
            src = os.path.join(src_dir, s + ".png")
            dst = os.path.join(out, s + ".png")
            if media.upscale_image(self.cfg, src, dst, self.cfg.lf_mult):
                done += 1
            else:
                self._log(f"lf_up FAIL {s}")
            self._tick("lf_up", done, len(stems))
        return done

    # ---- STAGE 7: continuation Wan clips (forced same direction) ---------
    def _stage_vid2(self):
        if self.cfg.native_long:      # whole clip already rendered in stage vid1
            return 0
        if self.cfg.single_clip:      # vid1 IS the master -- no continuation
            return 0
        if self.cfg.continuation_mode == "native_overlap":
            return self._stage_vid2_native()
        # Seed clip 2 from the crisp NATIVE last frame (832x480 = clip 1's real
        # last frame) so the continuation matches clip 1's sharpness. Seeding
        # from the x4-upscaled frame made clip 2 visibly softer than clip 1.
        seed_stage = "lastframe" if self.cfg.cont_seed == "native" else "lf_up"
        src_dir = self.cfg.stage_dir(seed_stage, ensure=False)
        stems = [s for s in self._stems_with_png(src_dir)
                 if s in self.posemap and not self._exists("vid2", s, video=True)
                 and self._allowed(s.split("_", 1)[1])]
        if not stems:
            return 0
        wf = self.cfg.workflows["wan"]
        base = self._load_wf("wan")
        self._begin("vid2", len(stems))
        done = 0
        for s in stems:
            if self._should_stop():
                break
            self._working("vid2", done + 1, len(stems))
            pm = self.posemap[s]
            if self._wan_clip(base, wf, src_dir, s, pm["letter"],
                              pm["pose"], pm["speed"], "vid2"):
                done += 1
            self._tick("vid2", done, len(stems))
        return done

    # ---- STAGE 7 (native-overlap): motion-conditioned continuation -------
    def _stage_vid2_native(self):
        """Seed the continuation with clip1's last `overlap` real frames (camera
        embedding dropped) so Wan continues the actual motion in one pass -> no
        surge/colour/speed/blur seam by construction. Output = raw 97f clip."""
        d1 = self.cfg.stage_dir("vid1", ensure=False)
        stems = [s for s in self._stems_with_clip(d1)
                 if s in self.posemap and not self._exists("vid2", s, video=True)
                 and self._allowed(s.split("_", 1)[1])]
        if not stems:
            return 0
        wf = self.cfg.workflows["wan"]
        base = self._load_wf("wan")
        self._begin("vid2", len(stems))
        done = 0
        for s in stems:
            if self._should_stop():
                break
            self._working("vid2", done + 1, len(stems))
            clip1 = self._clip_path("vid1", s)
            pm = self.posemap[s]
            letter = pm.get("letter", s.split("_", 1)[0])
            if clip1 and self._wan_continuation(base, wf, clip1, s, letter,
                                                pm.get("pose"), pm.get("speed")):
                done += 1
                with self.lock:
                    self._save_state()
            self._tick("vid2", done, len(stems))
        return done

    def _clip_nframes(self, path):
        try:
            out = subprocess.run(
                [self.cfg.ffprobe, "-v", "error", "-select_streams", "v:0",
                 "-count_frames", "-show_entries", "stream=nb_read_frames",
                 "-of", "csv=p=0", path],
                capture_output=True, text=True, timeout=120).stdout.strip()
            return int(out.splitlines()[0])
        except Exception:
            return 0

    def _wan_continuation(self, base, wf, clip1_path, stem, letter,
                          pose=None, speed=None):
        n1 = self._clip_nframes(clip1_path) or 81
        ov = self.cfg.overlap_frames
        g = json.loads(json.dumps(base))
        self._apply_render_res(g, wf)
        seed_path, skip = clip1_path, max(0, n1 - ov)
        restored = None
        if self.cfg.seed_restore:
            # Wan holds the detail level of its seed frames for the whole
            # continuation: seed with a detail-restored tail instead of the
            # raw (already melted) one. Measured +60% sharpness across the
            # full continuation on matched-seed A/B, no extra render time.
            seed_dir = os.path.join(self.cfg.workspace, "_seed_cache")
            os.makedirs(seed_dir, exist_ok=True)
            restored = os.path.join(seed_dir, f"{stem}_tail.mp4")
            if finish.restore_tail(self.cfg, clip1_path, restored, ov):
                seed_path, skip = restored, 0
            else:
                self._log(f"vid2 seed-restore failed {stem} — raw tail used")
                restored = None
        ckey = stem.split("_", 1)[-1]
        src_text = self.prompts.get(ckey, {}).get("text", "")
        g["200"] = {"class_type": "VHS_LoadVideoPath",
                    "_meta": {"title": "clip1 tail"},
                    "inputs": {"video": seed_path, "force_rate": 0.0,
                               "custom_width": 0, "custom_height": 0,
                               "frame_load_cap": ov,
                               "skip_first_frames": skip,
                               "select_every_nth": 1}}
        g[wf["node_wci2v"]]["inputs"][wf["field_start_image"]] = ["200", 0]
        g[wf["node_text"]]["inputs"][wf["field_text"]] = \
            self.cfg.motion_text(letter, src_text,
                                 custom=self.motionhints.get(ckey, ""))
        if wf.get("node_length"):
            g[wf["node_length"]]["inputs"][wf["field_length"]] = \
                int(self.cfg.continuation_length)
        if self.cfg.continuation_drop_camera:
            # legacy mode: inherit motion from the seeded frames only. Proven
            # to go near-static on pans/zooms (flow ~0.1-0.5x of clip1) ->
            # visible slow-down at the seam. Default is now camera-kept.
            g[wf["node_wci2v"]]["inputs"].pop(wf["field_camera_cond"], None)
        elif pose is not None:
            # camera-kept continuation: same pose/speed as clip1 so the move
            # carries through the cut at full rate.
            g[wf["node_camera"]]["inputs"][wf["field_pose"]] = pose
            g[wf["node_camera"]]["inputs"][wf["field_speed"]] = speed
        g[wf["node_seed"]]["inputs"][wf["field_seed"]] = random.randint(0, 2**31 - 1)
        g[wf["node_save"]]["inputs"][wf["field_save"]] = \
            self.cfg.comfy_prefix("vid2", stem)
        ok, msg = self._submit(
            g, self.cfg.timeout_vid, "vid2",
            reseed=lambda gg: gg[wf["node_seed"]]["inputs"].update(
                {wf["field_seed"]: random.randint(0, 2**31 - 1)}))
        if restored:
            try:
                os.remove(restored)
            except OSError:
                pass
        if ok and comfymod.rename_out(self.cfg.stage_dir("vid2"), stem,
                                      (".mp4", ".webm")):
            return True
        if not ok and self._salvage_render(
                "vid2", stem, int(self.cfg.continuation_length)):
            return True
        self._log(f"vid2 continuation FAIL {stem}: {msg}")
        return False

    # ---- STAGE 8: concat clip1 + clip2 -> ~10-11s ------------------------
    def _stage_concat(self):
        d1 = self.cfg.stage_dir("vid1", ensure=False)
        # native_long / single_clip: there is no clip2 to join, so just
        # promote vid1 to the concat output unchanged (a copy, no re-encode)
        # and final_up upscales it like any other.
        native = self.cfg.native_long or self.cfg.single_clip
        stems = [s for s in self._stems_with_clip(d1)
                 if (native or self._clip_path("vid2", s))
                 and not self._exists("concat", s, video=True)
                 and self._allowed(s.split("_", 1)[1])]
        if not stems:
            return 0
        out = self.cfg.stage_dir("concat")
        self._begin("concat", len(stems))
        done = 0
        for s in stems:
            if self._should_stop():
                break
            self._working("concat", done + 1, len(stems))
            a = self._clip_path("vid1", s)
            dst = os.path.join(out, s + ".mp4")
            if native:
                ok = bool(a) and self._promote_clip(a, dst)
            elif self.cfg.continuation_mode == "native_overlap":
                raw = self._clip_path("vid2", s)
                ok = bool(a) and bool(raw) and finish.gold_join(self.cfg, a, raw, dst)
            else:
                ok = media.concat_pair(self.cfg, a, self._clip_path("vid2", s), dst)
            if ok:
                done += 1
            else:
                self._log(f"concat FAIL {s}")
            self._tick("concat", done, len(stems))
        return done

    @staticmethod
    def _promote_clip(src, dst):
        """Copy a single native clip to the concat slot (no re-encode)."""
        try:
            shutil.copy2(src, dst)
            return os.path.exists(dst)
        except OSError:
            return False

    # ---- STAGE 9: final video upscale x4 ---------------------------------
    def _stage_final_up(self):
        d = self.cfg.stage_dir("concat", ensure=False)
        # skip a master that already exists ANYWHERE in the library: the
        # assembler may have moved it out of 09_final_up4 into a used bucket,
        # and a moved master is finished work, not missing work.
        stems = [s for s in self._stems_with_clip(d)
                 if not self._exists("final_up", s, video=True)
                 and not lib.master_exists(self.cfg, s)
                 and self._allowed(s.split("_", 1)[1])]
        if not stems:
            return 0
        out = self.cfg.stage_dir("final_up")
        self._begin("final_up", len(stems))
        done = 0
        for s in stems:
            if self._should_stop():
                break
            self._working("final_up", done + 1, len(stems))
            src = self._clip_path("concat", s)
            dst = os.path.join(out, s + ".mp4")
            if self.cfg.final_upscaler == "esrgan" and finish.esrgan_available(self.cfg):
                ok = finish.esrgan_finish(self.cfg, src, dst)
            else:
                ok = media.upscale_video(self.cfg, src, dst, self.cfg.final_mult)
            if ok:
                done += 1
            else:
                self._log(f"final_up FAIL {s}")
            self._tick("final_up", done, len(stems))
        return done

    def _apply_render_res(self, g, wf):
        """Inject the configured Wan render resolution into the camera
        embedding node (which feeds width/height to the sampler). 0 = keep
        whatever the workflow export carries."""
        if self.cfg.wan_width and self.cfg.wan_height:
            g[wf["node_camera"]]["inputs"]["width"] = self.cfg.wan_width
            g[wf["node_camera"]]["inputs"]["height"] = self.cfg.wan_height
        self._apply_sampler_steps(g, wf)

    def _apply_sampler_steps(self, g, wf):
        """Inject the configured sampler step count into the two-expert
        KSamplerAdvanced pair (high-noise then low-noise model). The split
        point stays at half: matched-seed A/B on a worst-case busy source
        measured end-of-clip texture melt 15% @4 steps, 10% @6, 3% @8 —
        with render time scaling roughly linearly (311s/498s/632s vid1).
        0 = keep the workflow's exported step count (4, Seko distill)."""
        steps = int(getattr(self.cfg, "wan_steps", 0) or 0)
        n1, n2 = wf.get("node_sampler1"), wf.get("node_sampler2")
        if not steps or not n1 or not n2:
            return
        split = steps // 2
        g[n1]["inputs"].update({"steps": steps, "start_at_step": 0,
                                "end_at_step": split})
        g[n2]["inputs"].update({"steps": steps, "start_at_step": split,
                                "end_at_step": steps})

    # -- Wan clip submit (shared by vid1 + vid2) ---------------------------
    def _wan_clip(self, base, wf, src_dir, stem, letter, pose, speed, out_stage,
                  length=None):
        src = os.path.join(src_dir, stem + ".png")
        staged = os.path.join(self.cfg.comfy_input, stem + ".png")
        try:
            os.makedirs(self.cfg.comfy_input, exist_ok=True)
            shutil.copy2(src, staged)
        except Exception as e:
            self._log(f"{out_stage} stage FAIL {stem}: {e}")
            return False
        g = json.loads(json.dumps(base))
        self._apply_render_res(g, wf)
        key = stem.split("_", 1)[-1]
        g[wf["node_image"]]["inputs"][wf["field_image"]] = stem + ".png"
        g[wf["node_text"]]["inputs"][wf["field_text"]] = self.cfg.motion_text(
            letter, self.prompts.get(key, {}).get("text", ""),
            custom=self.motionhints.get(key, ""))
        g[wf["node_camera"]]["inputs"][wf["field_pose"]] = pose
        g[wf["node_camera"]]["inputs"][wf["field_speed"]] = speed
        if length and wf.get("node_length"):
            g[wf["node_length"]]["inputs"][wf["field_length"]] = int(length)
        g[wf["node_seed"]]["inputs"][wf["field_seed"]] = random.randint(0, 2**31 - 1)
        g[wf["node_save"]]["inputs"][wf["field_save"]] = self.cfg.comfy_prefix(out_stage, stem)
        ok, msg = self._submit(
            g, self.cfg.timeout_vid, out_stage,
            reseed=lambda gg: gg[wf["node_seed"]]["inputs"].update(
                {wf["field_seed"]: random.randint(0, 2**31 - 1)}))
        try:
            os.remove(staged)
        except OSError:
            pass
        if ok and comfymod.rename_out(self.cfg.stage_dir(out_stage), stem, (".mp4", ".webm")):
            return True
        if not ok and self._salvage_render(out_stage, stem,
                                           int(length) if length else 81):
            return True
        self._log(f"{out_stage} FAIL {stem}: {msg}")
        return False

    # -- filename scanners --------------------------------------------------
    def _keys_with_png(self, d):
        if not os.path.isdir(d):
            return []
        return sorted(os.path.splitext(os.path.basename(p))[0]
                      for p in glob.glob(os.path.join(d, "*.png")))

    def _stems_with_png(self, d):
        return self._keys_with_png(d)

    def _stems_with_clip(self, d):
        if not os.path.isdir(d):
            return []
        stems = set()
        for ext in ("*.mp4", "*.webm"):
            for p in glob.glob(os.path.join(d, ext)):
                stems.add(os.path.splitext(os.path.basename(p))[0])
        return sorted(stems)


_PIPE = None


def get_pipeline():
    global _PIPE
    if _PIPE is None:
        _PIPE = Pipeline()
    return _PIPE
