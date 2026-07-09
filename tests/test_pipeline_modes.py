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


class TestModeStops(PipelineFixture):
    def test_target_reached(self):
        self.pipe.mode = {"target": 2}
        self.pipe.vid_judged = {"A_k1": True, "B_k2": True}
        self.assertTrue(self.pipe._target_reached())

    def test_target_counts_only_accepted(self):
        self.pipe.mode = {"target": 2}
        self.pipe.vid_judged = {"A_k1": True, "B_k2": False}
        self.assertFalse(self.pipe._target_reached())

    def test_time_up(self):
        self.pipe.mode = {"deadline": time.time() - 1}
        self.assertTrue(self.pipe._time_up())
        self.assertTrue(self.pipe._should_stop())

    def test_time_not_up(self):
        self.pipe.mode = {"deadline": time.time() + 60}
        self.assertFalse(self.pipe._time_up())


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
        self.pipe.img_rejects = {"k2": 1}
        self.pipe.vid_judged = {"A_k1": True}
        self.pipe.vid_rejects = {"k3": 2}
        self.pipe.abandoned = ["k3"]
        self.pipe.html_seen = {"a.html": 123}
        self.pipe.mode = {"target": 5}
        self.pipe._save_state()
        with mock.patch("stillnorth.pipeline.get_config",
                        return_value=self.cfg), \
             mock.patch("stillnorth.pipeline.comfymod.Comfy"):
            from stillnorth.pipeline import Pipeline
            p2 = Pipeline()
        self.assertEqual(p2.judged, {"k1": True})
        self.assertEqual(p2.img_rejects, {"k2": 1})
        self.assertEqual(p2.vid_judged, {"A_k1": True})
        self.assertEqual(p2.abandoned, ["k3"])
        self.assertEqual(p2.html_seen, {"a.html": 123})
        self.assertEqual(p2.mode, {"target": 5})


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


if __name__ == "__main__":
    unittest.main()
