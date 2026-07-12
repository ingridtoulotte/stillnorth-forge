"""Tests for target/time-budget scheduling, judge bookkeeping and chain
deletion in the pipeline (temp workspace, no ComfyUI/Ollama)."""
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stillnorth.config import STAGE_DIRS  # noqa: E402


def make_cfg(ws):
    cfg = mock.Mock()
    cfg.workspace = ws
    cfg.comfy_server = "127.0.0.1:1"
    cfg.poll = 0.1
    cfg.judge_enabled = True
    cfg.judge_video_enabled = True
    cfg.judge_image_retries = 2
    cfg.judge_video_retries = 1
    cfg.html_dir = os.path.join(ws, "00_html_prompts")
    cfg.stage_dir = lambda k, ensure=True: _stage_dir(ws, k, ensure)
    cfg.state_path = lambda: os.path.join(ws, "forge_state.json")
    cfg.log_path = lambda: os.path.join(ws, "forge.log")
    return cfg


def _stage_dir(ws, k, ensure):
    d = os.path.join(ws, STAGE_DIRS[k])
    if ensure:
        os.makedirs(d, exist_ok=True)
    return d


def touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(b"x")


class PipelineFixture(unittest.TestCase):
    def setUp(self):
        self.ws = tempfile.mkdtemp(prefix="snf_test_")
        self.cfg = make_cfg(self.ws)
        with mock.patch("stillnorth.pipeline.get_config",
                        return_value=self.cfg), \
             mock.patch("stillnorth.pipeline.comfymod.Comfy"):
            from stillnorth.pipeline import Pipeline
            self.pipe = Pipeline()

    def tearDown(self):
        shutil.rmtree(self.ws, ignore_errors=True)

    def add_prompts(self, keys):
        for k in keys:
            self.pipe.prompts[k] = {"text": "t", "src": "s", "title": k}


class TestActiveSet(PipelineFixture):
    def test_no_mode_no_limit(self):
        self.add_prompts(["k1", "k2"])
        self.pipe.mode = None
        self.assertIsNone(self.pipe._pick_active())
        self.assertTrue(self.pipe._allowed("anything"))

    def test_target_mode_limits_to_remaining(self):
        self.add_prompts(["k1", "k2", "k3", "k4"])
        self.pipe.mode = {"target": 2}
        active = self.pipe._pick_active()
        self.assertEqual(len(active), 2)

    def test_advanced_chains_first(self):
        self.add_prompts(["k1", "k2", "k3"])
        self.pipe.mode = {"target": 1}
        # k2 already has a vid1 clip -> must be preferred over fresh keys
        self.pipe.letters["k2"] = "A"
        touch(os.path.join(self.ws, STAGE_DIRS["vid1"], "A_k2.mp4"))
        active = self.pipe._pick_active()
        self.assertEqual(active, {"k2"})

    def test_accepted_master_leaves_active_set(self):
        self.add_prompts(["k1", "k2"])
        self.pipe.mode = {"target": 1}
        self.pipe.letters["k1"] = "A"
        self.pipe.vid_judged["A_k1"] = True
        touch(os.path.join(self.ws, STAGE_DIRS["final_up"], "A_k1.mp4"))
        active = self.pipe._pick_active()
        self.assertEqual(len(active), 0)   # target already satisfied

    def test_abandoned_key_excluded(self):
        self.add_prompts(["k1", "k2"])
        self.pipe.mode = {"target": 2}
        self.pipe.abandoned = ["k1"]
        active = self.pipe._pick_active()
        self.assertEqual(active, {"k2"})

    def test_time_mode_wave(self):
        from stillnorth.pipeline import WAVE_SIZE
        self.add_prompts([f"k{i}" for i in range(10)])
        self.pipe.mode = {"deadline": time.time() + 600}
        active = self.pipe._pick_active()
        self.assertEqual(len(active), WAVE_SIZE)


class TestSingleClipMode(PipelineFixture):
    def test_active_stages_skip_continuation(self):
        self.pipe.cfg.native_long = False
        self.pipe.cfg.single_clip = True
        self.pipe.cfg.continuation_mode = "native_overlap"
        self.assertEqual(self.pipe._active_stages(),
                         ["img_up", "vid1", "concat", "final_up"])

    def test_vid2_stage_skipped(self):
        self.pipe.cfg.native_long = False
        self.pipe.cfg.single_clip = True
        self.assertEqual(self.pipe._stage_vid2(), 0)

    def test_concat_promotes_vid1(self):
        self.pipe.cfg.native_long = False
        self.pipe.cfg.single_clip = True
        self.pipe.cfg.continuation_mode = "native_overlap"
        self.add_prompts(["k1"])
        self.pipe.letters["k1"] = "A"
        touch(os.path.join(self.ws, STAGE_DIRS["vid1"], "A_k1.mp4"))
        self.assertEqual(self.pipe._stage_concat(), 1)
        self.assertTrue(os.path.exists(
            os.path.join(self.ws, STAGE_DIRS["concat"], "A_k1.mp4")))

    def test_config_default_off(self):
        from stillnorth.config import Config
        c = Config.__new__(Config)
        c.raw = {"comfy_server": "x", "ffmpeg": "ffmpeg",
                 "comfy_input_dir": ".", "comfy_output_dir": ".",
                 "workspace_subdir": "W", "server_host": "127.0.0.1",
                 "server_port": 1, "render_timeout_img": 1,
                 "render_timeout_vid": 1, "poll_seconds": 1,
                 "image_upscale_mult": 2, "lastframe_upscale_mult": 4,
                 "final_upscale_mult": 4, "fps": 16, "video_cq": 19,
                 "nvenc": True, "upscale_denoise": "", "upscale_sharp": "",
                 "upscale_grain": ""}
        with mock.patch("stillnorth.config._load",
                        side_effect=[c.raw, {}, {"poses": [], "speed_by_pose": {},
                                                 "classes": {}}]):
            c.reload()
        self.assertFalse(c.single_clip)


class TestAutoResumeKeepsMode(PipelineFixture):
    def test_resume_preserves_target_mode(self):
        """Auto-resume after a crash/restart must reuse the persisted mode —
        resuming a target=3 run as classic full-set once rendered 48
        unwanted flux stills."""
        self.pipe.mode = {"target": 3}
        self.pipe.desired_running = True
        with mock.patch.object(self.pipe, "_pending_work", return_value=True), \
             mock.patch.object(self.pipe, "_run"):
            self.assertTrue(self.pipe.maybe_auto_resume())
        self.assertEqual(self.pipe.mode, {"target": 3})

    def test_resume_spent_time_budget_stays_stopped(self):
        self.pipe.mode = {"deadline": time.time() - 10, "minutes": 5.0}
        self.pipe.desired_running = True
        with mock.patch.object(self.pipe, "_pending_work", return_value=True):
            self.assertFalse(self.pipe.maybe_auto_resume())
        self.assertFalse(self.pipe.desired_running)

    def test_explicit_start_still_sets_mode(self):
        with mock.patch.object(self.pipe, "_run"):
            self.pipe.start(target=5)
        # start() now also snapshots a baseline (0 with an empty catalog) so
        # target counts NEW masters, not the raw total.
        self.assertEqual(self.pipe.mode, {"target": 5, "baseline": 0})


class TestAcceptedCountsOnlyExistingMasters(PipelineFixture):
    def test_stale_accept_for_archived_master_not_counted(self):
        """An accept whose master was manually archived/deleted must not
        count toward the target — it once made target mode declare victory
        without rendering a replacement."""
        self.pipe.cfg.judge_enabled = True
        self.pipe.cfg.judge_video_enabled = True
        touch(os.path.join(self.ws, STAGE_DIRS["final_up"], "A_k1.mp4"))
        self.pipe.vid_judged = {"A_k1": True, "A_gone": True}
        with mock.patch("stillnorth.pipeline.lib.master_exists",
                        return_value=False):
            self.assertEqual(self.pipe.accepted_count(), 1)


class TestModeStops(PipelineFixture):
    def test_target_reached(self):
        self.pipe.mode = {"target": 2}
        self.pipe.vid_judged = {"A_k1": True, "B_k2": True}
        touch(os.path.join(self.ws, STAGE_DIRS["final_up"], "A_k1.mp4"))
        touch(os.path.join(self.ws, STAGE_DIRS["final_up"], "B_k2.mp4"))
        self.assertTrue(self.pipe._target_reached())

    def test_target_counts_only_accepted(self):
        self.pipe.mode = {"target": 2}
        self.pipe.vid_judged = {"A_k1": True, "B_k2": False}
        touch(os.path.join(self.ws, STAGE_DIRS["final_up"], "A_k1.mp4"))
        touch(os.path.join(self.ws, STAGE_DIRS["final_up"], "B_k2.mp4"))
        self.assertFalse(self.pipe._target_reached())

    def test_time_up(self):
        self.pipe.mode = {"deadline": time.time() - 1}
        self.assertTrue(self.pipe._time_up())
        self.assertTrue(self.pipe._should_stop())

    def test_time_not_up(self):
        self.pipe.mode = {"deadline": time.time() + 60}
        self.assertFalse(self.pipe._time_up())


class TestTargetIsIncremental(PipelineFixture):
    """target=N means 'N MORE masters', not the raw catalog total. With 3
    already accepted, target=2 must render exactly 2 new ones, not declare
    instant victory (user: 'the number isn't the total, it's how many MORE
    I want')."""
    def setUp(self):
        super().setUp()
        self.pipe.cfg.judge_enabled = True
        self.pipe.cfg.judge_video_enabled = True

    def _accept(self, stem):
        self.pipe.vid_judged[stem] = True
        touch(os.path.join(self.ws, STAGE_DIRS["final_up"], stem + ".mp4"))

    def test_baseline_captured_at_start(self):
        for s in ("A_a", "A_b", "A_c"):
            self._accept(s)
        with mock.patch.object(self.pipe, "_run"):
            self.pipe.start(target=2)
        self.assertEqual(self.pipe.mode["baseline"], 3)

    def test_pre_existing_masters_do_not_satisfy_target(self):
        for s in ("A_a", "A_b", "A_c"):
            self._accept(s)
        self.pipe.mode = {"target": 2, "baseline": self.pipe.accepted_count()}
        self.assertFalse(self.pipe._target_reached())        # 0 new yet
        self.add_prompts(["k4", "k5", "k6"])
        self.assertEqual(len(self.pipe._pick_active()), 2)   # wants exactly 2 new

    def test_reaches_target_after_n_new(self):
        for s in ("A_a", "A_b", "A_c"):
            self._accept(s)
        self.pipe.mode = {"target": 2, "baseline": self.pipe.accepted_count()}
        self._accept("A_new1")
        self.assertFalse(self.pipe._target_reached())        # 1 new < 2
        self._accept("A_new2")
        self.assertTrue(self.pipe._target_reached())         # 2 new -> done


class TestSubmitRecovery(PipelineFixture):
    """Timeout-storm fixes: a wait timeout must interrupt the job still
    running server-side, retries must not resubmit a byte-identical graph
    (ComfyUI cache-collapses it into a completed-but-empty history), and a
    completed-but-orphaned output must be salvaged, not re-rendered."""

    def _prep(self):
        self.pipe.cfg.submit_retries = 1
        self.pipe.cfg.retry_backoff = 0.0
        self.pipe.cancel_flag = False

    def test_timeout_interrupts_running_job(self):
        self._prep()
        self.pipe.comfy.queue = mock.Mock(return_value="pid1")
        self.pipe.comfy.wait = mock.Mock(return_value=(False, "timeout"))
        self.pipe.comfy.interrupt = mock.Mock()
        ok, msg = self.pipe._submit({}, 1, "vid2")
        self.assertFalse(ok)
        self.assertEqual(self.pipe.comfy.interrupt.call_count, 2)

    def test_retry_reseeds_graph(self):
        self._prep()
        g = {"71": {"inputs": {"noise_seed": 111}}}
        seeds = []
        self.pipe.comfy.queue = mock.Mock(
            side_effect=lambda wf: seeds.append(
                wf["71"]["inputs"]["noise_seed"]) or "pid")
        self.pipe.comfy.wait = mock.Mock(return_value=(False, "no output"))
        self.pipe.comfy.interrupt = mock.Mock()

        def reseed(wf):
            wf["71"]["inputs"]["noise_seed"] = 222

        ok, _ = self.pipe._submit(g, 1, "vid2", reseed=reseed)
        self.assertFalse(ok)
        self.assertEqual(seeds, [111, 222])

    def test_salvage_orphaned_completed_render(self):
        d = os.path.join(self.ws, STAGE_DIRS["vid2"])
        touch(os.path.join(d, "A_k1_00001_.mp4"))
        with mock.patch.object(self.pipe, "_clip_nframes", return_value=97):
            self.assertTrue(self.pipe._salvage_render("vid2", "A_k1", 97))
        self.assertTrue(os.path.exists(os.path.join(d, "A_k1.mp4")))

    def test_salvage_discards_short_clip(self):
        d = os.path.join(self.ws, STAGE_DIRS["vid2"])
        touch(os.path.join(d, "A_k2_00001_.mp4"))
        with mock.patch.object(self.pipe, "_clip_nframes", return_value=40):
            self.assertFalse(self.pipe._salvage_render("vid2", "A_k2", 97))
        self.assertFalse(os.path.exists(os.path.join(d, "A_k2.mp4")))

    def test_salvage_nothing_on_disk(self):
        self.assertFalse(self.pipe._salvage_render("vid2", "A_k3", 97))


class TestDeleteChain(PipelineFixture):
    def test_delete_key_files_full_chain(self):
        self.pipe.letters["k1"] = "A"
        self.pipe.posemap["A_k1"] = {"letter": "A"}
        self.pipe.judged["k1"] = True
        self.pipe.vid_judged["A_k1"] = False
        files = [
            os.path.join(self.ws, STAGE_DIRS["flux"], "k1.png"),
            os.path.join(self.ws, STAGE_DIRS["img_up"], "k1.png"),
            os.path.join(self.ws, STAGE_DIRS["classified"], "A_k1.png"),
            os.path.join(self.ws, STAGE_DIRS["vid1"], "A_k1.mp4"),
            os.path.join(self.ws, STAGE_DIRS["vid2"], "A_k1.mp4"),
            os.path.join(self.ws, STAGE_DIRS["concat"], "A_k1.mp4"),
            os.path.join(self.ws, STAGE_DIRS["final_up"], "A_k1.mp4"),
        ]
        for f in files:
            touch(f)
        self.pipe._delete_key_files("k1", flux_too=True)
        for f in files:
            self.assertFalse(os.path.exists(f), f)
        self.assertNotIn("k1", self.pipe.judged)
        self.assertNotIn("k1", self.pipe.letters)
        self.assertNotIn("A_k1", self.pipe.posemap)
        self.assertNotIn("A_k1", self.pipe.vid_judged)

    def test_image_reject_keeps_nothing_downstream(self):
        self.pipe.letters["k1"] = "A"
        flux = os.path.join(self.ws, STAGE_DIRS["flux"], "k1.png")
        up = os.path.join(self.ws, STAGE_DIRS["img_up"], "k1.png")
        touch(flux)
        touch(up)
        self.pipe._delete_key_files("k1", flux_too=True)
        self.assertFalse(os.path.exists(flux))
        self.assertFalse(os.path.exists(up))


class TestStatePersistence(PipelineFixture):
    def test_judge_state_roundtrip(self):
        self.pipe.judged = {"k1": True}
        self.pipe.judged_gate = {"k1": 2}
        self.pipe.img_rejects = {"k2": 1}
        self.pipe.vid_judged = {"A_k1": True}
        self.pipe.vid_rejects = {"k3": 2}
        self.pipe.abandoned = ["k3"]
        self.pipe.html_seen = {"a.html": 123}
        self.pipe.mode = {"target": 5}
        self.pipe.posehints = {"k1": ["FORWARD"]}
        self.pipe.motionhints = {"k1": "gliding over the ridge"}
        self.pipe._save_state()
        with mock.patch("stillnorth.pipeline.get_config",
                        return_value=self.cfg), \
             mock.patch("stillnorth.pipeline.comfymod.Comfy"):
            from stillnorth.pipeline import Pipeline
            p2 = Pipeline()
        self.assertEqual(p2.judged, {"k1": True})
        self.assertEqual(p2.judged_gate, {"k1": 2})
        self.assertEqual(p2.img_rejects, {"k2": 1})
        self.assertEqual(p2.vid_judged, {"A_k1": True})
        self.assertEqual(p2.abandoned, ["k3"])
        self.assertEqual(p2.html_seen, {"a.html": 123})
        self.assertEqual(p2.mode, {"target": 5})
        self.assertEqual(p2.posehints, {"k1": ["FORWARD"]})
        self.assertEqual(p2.motionhints, {"k1": "gliding over the ridge"})


class TestRegateStaleStills(PipelineFixture):
    """A still accepted under an OLDER CV gate version must be re-checked on
    resume — never grandfathered past a gate added after it was first judged
    (the bug that shipped dense-micro-texture "240p mush" masters whose
    pre-struct_ratio-gate accepts were never re-evaluated)."""

    def _flux(self, k):
        p = os.path.join(self.ws, STAGE_DIRS["flux"], k + ".png")
        touch(p)
        return p

    def _src_dir(self):
        return self.cfg.stage_dir("flux", ensure=False)

    def test_stale_fail_drops_chain(self):
        self.pipe.letters["k1"] = "A"
        self.pipe.judged["k1"] = True          # no judged_gate -> version 0
        p = self._flux("k1")
        with mock.patch("stillnorth.pipeline.judgemod.cv_gate",
                        return_value=(False, "mush")):
            self.pipe._regate_stale_stills(self._src_dir())
        self.assertNotIn("k1", self.pipe.judged)
        self.assertFalse(os.path.exists(p))

    def test_stale_pass_stamps_version(self):
        from stillnorth.judge import CV_GATE_VERSION
        self.pipe.judged["k1"] = True
        self._flux("k1")
        with mock.patch("stillnorth.pipeline.judgemod.cv_gate",
                        return_value=(True, "")):
            self.pipe._regate_stale_stills(self._src_dir())
        self.assertTrue(self.pipe.judged.get("k1"))
        self.assertEqual(self.pipe.judged_gate.get("k1"), CV_GATE_VERSION)

    def test_current_version_not_rechecked(self):
        from stillnorth.judge import CV_GATE_VERSION
        self.pipe.judged["k1"] = True
        self.pipe.judged_gate["k1"] = CV_GATE_VERSION
        self._flux("k1")
        with mock.patch("stillnorth.pipeline.judgemod.cv_gate") as g:
            self.pipe._regate_stale_stills(self._src_dir())
            g.assert_not_called()

    def test_missing_png_skipped(self):
        self.pipe.judged["k1"] = True          # stale, but no still on disk
        with mock.patch("stillnorth.pipeline.judgemod.cv_gate") as g:
            self.pipe._regate_stale_stills(self._src_dir())
            g.assert_not_called()
        self.assertTrue(self.pipe.judged.get("k1"))


class TestHtmlAutoload(PipelineFixture):
    def test_scan_ingests_new_files(self):
        d = self.cfg.html_dir
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "batch.html"), "w", encoding="utf-8") as fh:
            fh.write("<html><body>no prompts here</body></html>")
        with mock.patch.object(self.pipe, "ingest_html",
                               return_value=(0, 0)) as ing:
            self.pipe._scan_html_dir()
            ing.assert_called_once()
            # second scan: same mtime -> not re-ingested
            self.pipe._scan_html_dir()
            ing.assert_called_once()

    def test_scan_ingests_txt_and_json(self):
        """Plain .txt / .json holding a {"scene": ...} object is ingested too,
        not only .html — the drop folder is format-agnostic (real parser)."""
        d = self.cfg.html_dir
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "hand.txt"), "w", encoding="utf-8") as fh:
            fh.write('{"scene": "a lone pine on a snowy ridge", "title": "pine"}')
        with open(os.path.join(d, "batch.json"), "w", encoding="utf-8") as fh:
            fh.write('[{"scene": "a frozen lake at dawn", "title": "lake"}]')
        self.pipe._scan_html_dir()
        titles = {v.get("title") for v in self.pipe.prompts.values()}
        self.assertIn("pine", titles)
        self.assertIn("lake", titles)
        self.assertEqual(len(self.pipe.prompts), 2)


if __name__ == "__main__":
    unittest.main()
