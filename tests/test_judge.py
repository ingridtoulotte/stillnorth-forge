"""Unit tests for the AI judge module and its pipeline wiring (no Ollama,
no ComfyUI — network and subprocess calls are stubbed)."""
import json
import os
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stillnorth import judge  # noqa: E402


class FakeCfg:
    judge_model = "huihui_ai/Qwen3.6-abliterated:35b"
    ollama_url = "http://127.0.0.1:11434"
    judge_timeout = 5
    judge_keep_alive = "0"
    judge_shimmer_max = 3.5
    judge_instab_max = 0.15
    ffmpeg = "ffmpeg"
    ffprobe = "ffprobe"


class TestVerdictParsing(unittest.TestCase):
    def test_none_passes(self):
        ok, _ = judge._parse_verdict("NONE")
        self.assertTrue(ok)

    def test_minor_passes(self):
        ok, _ = judge._parse_verdict("MINOR: slightly uniform ripples")
        self.assertTrue(ok)

    def test_impossible_rejects(self):
        ok, r = judge._parse_verdict(
            "IMPOSSIBLE: duplicated rectangular patch lower left")
        self.assertFalse(ok)
        self.assertIn("duplicated", r)

    def test_impossible_lowercase_rejects(self):
        ok, _ = judge._parse_verdict("impossible: melted trees at center")
        self.assertFalse(ok)

    def test_impossible_on_later_line_rejects(self):
        ok, _ = judge._parse_verdict(
            "The frames look mostly fine.\nIMPOSSIBLE: shadow teleports")
        self.assertFalse(ok)

    def test_empty_is_reject(self):
        ok, _ = judge._parse_verdict("")
        self.assertFalse(ok)

    def test_freeform_prose_passes(self):
        """Wordy-but-not-IMPOSSIBLE replies pass — measured failure mode of
        local VLMs is false rejects, not false passes."""
        ok, _ = judge._parse_verdict("Everything looks plausible to me.")
        self.assertTrue(ok)


class TestJudgeImage(unittest.TestCase):
    def test_ollama_down_passes_unjudged(self):
        """A stopped Ollama must NOT reject work (batch keeps flowing)."""
        with mock.patch.object(judge, "_chat", side_effect=OSError("boom")), \
             mock.patch.object(judge, "_b64_image", return_value="x"):
            ok, reason = judge.judge_image(FakeCfg(), "img.png")
        self.assertTrue(ok)
        self.assertIn("unjudged", reason)

    def test_reject_flows_through(self):
        with mock.patch.object(judge, "_chat",
                               return_value="IMPOSSIBLE: melted ridge center"), \
             mock.patch.object(judge, "_b64_image", return_value="x"):
            ok, reason = judge.judge_image(FakeCfg(), "img.png")
        self.assertFalse(ok)
        self.assertIn("melted", reason)


class TestJudgeVideo(unittest.TestCase):
    def test_sampling_failure_passes_unjudged(self):
        with mock.patch.object(judge, "sample_frames", return_value=[]):
            ok, reason = judge.judge_video(FakeCfg(), "vid.mp4")
        self.assertTrue(ok)
        self.assertIn("unjudged", reason)

    def test_cv_gate_rejects_flicker(self):
        """Both CV metrics over threshold -> reject before the VLM runs."""
        with mock.patch.object(judge, "flicker_metrics",
                               return_value={"shimmer": 4.2,
                                             "tex_instab": 0.21,
                                             "frames": 200}):
            ok, reason = judge.judge_video(FakeCfg(), "vid.mp4")
        self.assertFalse(ok)
        self.assertIn("flicker", reason)

    def test_cv_gate_needs_both_signals(self):
        """One high metric alone (fast pan over fine texture) must NOT
        reject — it falls through to the VLM check."""
        with mock.patch.object(judge, "flicker_metrics",
                               return_value={"shimmer": 4.2,
                                             "tex_instab": 0.05,
                                             "frames": 200}), \
             mock.patch.object(judge, "sample_frames",
                               return_value=["a", "b", "c"]), \
             mock.patch.object(judge, "_b64_image", return_value="x"), \
             mock.patch.object(judge, "_chat", return_value="NONE"):
            ok, _ = judge.judge_video(FakeCfg(), "vid.mp4")
        self.assertTrue(ok)

    def test_reject(self):
        with mock.patch.object(judge, "sample_frames",
                               return_value=["a", "b", "c"]), \
             mock.patch.object(judge, "_b64_image", return_value="x"), \
             mock.patch.object(judge, "_chat",
                               return_value="IMPOSSIBLE: shadows jump"):
            ok, reason = judge.judge_video(FakeCfg(), "vid.mp4")
        self.assertFalse(ok)


class TestConfigKeys(unittest.TestCase):
    def test_defaults_present(self):
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
        self.assertTrue(c.judge_enabled)
        self.assertEqual(c.judge_model, "huihui_ai/Qwen3.6-abliterated:35b")
        self.assertEqual(c.judge_image_retries, 3)
        self.assertEqual(c.judge_video_retries, 2)
        self.assertTrue(c.html_dir.endswith("00_html_prompts"))

    def test_review_stage_dir(self):
        from stillnorth.config import STAGE_DIRS
        self.assertEqual(STAGE_DIRS["review"], "10_review")


if __name__ == "__main__":
    unittest.main()
