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
import threading
import time
import traceback

from .config import get_config, STAGE_DIRS
from . import comfy as comfymod
from . import media
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
}


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
        self.metrics = {}    # stage -> {n, fail, t, min, max}  (per-stage timing)
        self.desired_running = False  # crash-safe intent: auto-resume on launch
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
                self.metrics = d.get("metrics", {})
                self.desired_running = bool(d.get("desired_running", False))
            except Exception:
                pass

    def _save_state(self):
        """Atomic write: serialise to a temp file then os.replace() so a crash
        mid-write can never leave a half-written (corrupt) state file."""
        p = self.cfg.state_path()
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"prompts": self.prompts, "letters": self.letters,
                       "posemap": self.posemap, "metrics": self.metrics,
                       "desired_running": self.desired_running}, fh, indent=1)
        os.replace(tmp, p)

    def _log(self, msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        try:
            with open(self.cfg.log_path(), "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:
            pass
        print(line, flush=True)

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

    def start(self):
        """Start the worker if not already running (idempotent)."""
        with self.lock:
            if self._thread and self._thread.is_alive():
                return False
            self.cancel_flag = False
            self.desired_running = True       # remember intent across restarts
            self._save_state()
            self.status.update(cancelled=False, last_error=None, note="starting…")
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            return True

    def maybe_auto_resume(self):
        """Called once on launch. If the batch was running when the terminal
        was closed or the PC shut down (intent persisted, work still pending),
        resume it automatically -- no need to press Run again."""
        if self.desired_running and self._pending_work():
            self._log("auto-resume: batch was running before shutdown — resuming")
            return self.start()
        return False

    def _pending_work(self):
        """True if any stage still has an item to render (drives auto-resume)."""
        if any(not self._exists("flux", k) for k in self.prompts):
            return True
        stages = ["img_up", "vid1", "concat", "final_up"] if self.cfg.native_long \
            else ["img_up", "vid1", "lastframe", "lf_up", "vid2", "concat", "final_up"]
        for stage in stages:
            if self._count(stage) < self._count_prev(stage):
                return True
        return self._count("final_up") < len(self.prompts)

    def _count_prev(self, stage):
        order = ["flux", "img_up", "classify", "vid1", "lastframe",
                 "lf_up", "vid2", "concat", "final_up"]
        i = order.index(stage)
        return self._count(order[i - 1]) if i > 0 else len(self.prompts)

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
            self.metrics = {}
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
        d = self.cfg.stage_dir(STAGE_OUT[stage], ensure=False)
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
        self._log("pipeline run started")
        try:
            while not self.cancel_flag:
                work = self._one_pass()
                if work == 0:
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
                self.set_status(running=False, stage="done", label="done",
                                percent=100, stage_done=0, stage_total=0,
                                cancelled=False, note="all stages complete ✓")
        except Exception as e:
            self._log("FATAL " + repr(e) + "\n" + traceback.format_exc())
            self.set_status(running=False, stage="idle", label="error",
                            percent=0, stage_done=0, stage_total=0,
                            last_error=str(e), note="")
        self._log("pipeline run finished")

    def _one_pass(self):
        done = 0
        done += self._stage_flux()
        done += self._stage_img_up()
        done += self._stage_classify()
        done += self._stage_vid1()
        done += self._stage_lastframe()
        done += self._stage_lf_up()
        done += self._stage_vid2()
        done += self._stage_concat()
        done += self._stage_final_up()
        return done

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

    def _submit(self, wf, timeout, stage):
        attempts = self.cfg.submit_retries + 1
        t0 = time.time()
        last = "queue failed"
        for a in range(attempts):
            if self.cancel_flag:
                return False, "cancelled"
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
            last = self._categorize(msg)
            self._log(f"{stage} render failed (try {a + 1}/{attempts}): {last}")
            self._backoff(a)
        self._record(stage, time.time() - t0, False)
        return False, last

    # ---- STAGE 1: FLUX images --------------------------------------------
    def _stage_flux(self):
        with self.lock:
            items = [(k, v["text"]) for k, v in self.prompts.items()]
        todo = [(k, t) for k, t in items if not self._exists("flux", k)]
        if not todo:
            return 0
        wf = self.cfg.workflows["flux"]
        base = self._load_wf("flux")
        out = self.cfg.stage_dir("flux")
        self._begin("flux", len(todo))
        done = 0
        for k, text in todo:
            if self.cancel_flag:
                break
            self._working("flux", done + 1, len(todo))
            g = json.loads(json.dumps(base))
            g[wf["node_text"]]["inputs"][wf["field_text"]] = text
            g[wf["node_seed"]]["inputs"][wf["field_seed"]] = random.randint(0, 2**31 - 1)
            g[wf["node_save"]]["inputs"][wf["field_save"]] = self.cfg.comfy_prefix("flux", k)
            ok, msg = self._submit(g, self.cfg.timeout_img, "flux")
            if ok and comfymod.rename_out(out, k, (".png",)):
                done += 1
            else:
                self._log(f"flux FAIL {k}: {msg}")
            self._tick("flux", done, len(todo))
        return done

    # ---- STAGE 2: image upscale x2 ---------------------------------------
    def _stage_img_up(self):
        src_dir = self.cfg.stage_dir("flux", ensure=False)
        todo = [k for k in self._keys_with_png(src_dir) if not self._exists("img_up", k)]
        if not todo:
            return 0
        out = self.cfg.stage_dir("img_up")
        self._begin("img_up", len(todo))
        done = 0
        for k in todo:
            if self.cancel_flag:
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
        keys = [k for k in self._keys_with_png(src_dir) if k not in self.letters
                or not self._classified_exists(k)]
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
                 if not self._exists("vid1", s, video=True)]
        if not stems:
            return 0
        wf = self.cfg.workflows["wan"]
        base = self._load_wf("wan")
        out = self.cfg.stage_dir("vid1")
        self._begin("vid1", len(stems))
        done = 0
        for s in stems:
            if self.cancel_flag:
                break
            self._working("vid1", done + 1, len(stems))
            letter = s.split("_", 1)[0]
            pose = random.choice(self.cfg.poses)          # randomized direction
            speed = self.cfg.speed_by_pose[pose]
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
        if self.cfg.native_long:      # one native clip -> no continuation chain
            return 0
        src_dir = self.cfg.stage_dir("vid1", ensure=False)
        stems = [s for s in self._stems_with_clip(src_dir)
                 if not self._exists("lastframe", s)]
        if not stems:
            return 0
        out = self.cfg.stage_dir("lastframe")
        self._begin("lastframe", len(stems))
        done = 0
        for s in stems:
            if self.cancel_flag:
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
            if self.cancel_flag:
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
        # Seed clip 2 from the crisp NATIVE last frame (832x480 = clip 1's real
        # last frame) so the continuation matches clip 1's sharpness. Seeding
        # from the x4-upscaled frame made clip 2 visibly softer than clip 1.
        seed_stage = "lastframe" if self.cfg.cont_seed == "native" else "lf_up"
        src_dir = self.cfg.stage_dir(seed_stage, ensure=False)
        stems = [s for s in self._stems_with_png(src_dir)
                 if s in self.posemap and not self._exists("vid2", s, video=True)]
        if not stems:
            return 0
        wf = self.cfg.workflows["wan"]
        base = self._load_wf("wan")
        self._begin("vid2", len(stems))
        done = 0
        for s in stems:
            if self.cancel_flag:
                break
            self._working("vid2", done + 1, len(stems))
            pm = self.posemap[s]
            if self._wan_clip(base, wf, src_dir, s, pm["letter"],
                              pm["pose"], pm["speed"], "vid2"):
                done += 1
            self._tick("vid2", done, len(stems))
        return done

    # ---- STAGE 8: concat clip1 + clip2 -> ~10-11s ------------------------
    def _stage_concat(self):
        d1 = self.cfg.stage_dir("vid1", ensure=False)
        # native_long: the ~10s clip is the single vid1 render -> there is no
        # clip2 to join, so just promote vid1 to the concat output unchanged
        # (a copy, no re-encode) and final_up upscales it like any other.
        native = self.cfg.native_long
        stems = [s for s in self._stems_with_clip(d1)
                 if (native or self._clip_path("vid2", s))
                 and not self._exists("concat", s, video=True)]
        if not stems:
            return 0
        out = self.cfg.stage_dir("concat")
        self._begin("concat", len(stems))
        done = 0
        for s in stems:
            if self.cancel_flag:
                break
            self._working("concat", done + 1, len(stems))
            a = self._clip_path("vid1", s)
            dst = os.path.join(out, s + ".mp4")
            if native:
                ok = bool(a) and self._promote_clip(a, dst)
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
                 and not lib.master_exists(self.cfg, s)]
        if not stems:
            return 0
        out = self.cfg.stage_dir("final_up")
        self._begin("final_up", len(stems))
        done = 0
        for s in stems:
            if self.cancel_flag:
                break
            self._working("final_up", done + 1, len(stems))
            src = self._clip_path("concat", s)
            dst = os.path.join(out, s + ".mp4")
            if media.upscale_video(self.cfg, src, dst, self.cfg.final_mult):
                done += 1
            else:
                self._log(f"final_up FAIL {s}")
            self._tick("final_up", done, len(stems))
        return done

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
        g[wf["node_image"]]["inputs"][wf["field_image"]] = stem + ".png"
        g[wf["node_text"]]["inputs"][wf["field_text"]] = self.cfg.motion_text(letter)
        g[wf["node_camera"]]["inputs"][wf["field_pose"]] = pose
        g[wf["node_camera"]]["inputs"][wf["field_speed"]] = speed
        if length and wf.get("node_length"):
            g[wf["node_length"]]["inputs"][wf["field_length"]] = int(length)
        g[wf["node_seed"]]["inputs"][wf["field_seed"]] = random.randint(0, 2**31 - 1)
        g[wf["node_save"]]["inputs"][wf["field_save"]] = self.cfg.comfy_prefix(out_stage, stem)
        ok, msg = self._submit(g, self.cfg.timeout_vid, out_stage)
        try:
            os.remove(staged)
        except OSError:
            pass
        if ok and comfymod.rename_out(self.cfg.stage_dir(out_stage), stem, (".mp4", ".webm")):
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
