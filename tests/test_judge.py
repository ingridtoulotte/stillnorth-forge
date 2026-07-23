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

    def test_notdrone_rejects(self):
        """Composition off-brief (ground-level macro instead of an aerial
        shot) rejects so the source regenerates."""
        ok, r = judge._parse_verdict("NOTDRONE: ground-level tulip close-up")
        self.assertFalse(ok)
        self.assertIn("tulip", r)

    def test_notdrone_on_later_line_rejects(self):
        ok, _ = judge._parse_verdict(
            "Looks sharp overall.\nNOTDRONE: macro shot of moss")
        self.assertFalse(ok)

    def test_image_prompt_targets_composition(self):
        self.assertIn("VANTAGE: GROUND", judge.IMAGE_PROMPT)
        self.assertIn("close-up", judge.IMAGE_PROMPT)

    def test_vantage_ground_rejects(self):
        ok, r = judge._parse_verdict(
            "VANTAGE: GROUND — eye-level flower bed\nNONE")
        self.assertFalse(ok)
        self.assertIn("off-brief", r)

    def test_vantage_air_passes(self):
        ok, _ = judge._parse_verdict(
            "VANTAGE: AIR — high above a valley\nNONE")
        self.assertTrue(ok)

    def test_vantage_air_with_impossible_still_rejects(self):
        ok, _ = judge._parse_verdict(
            "VANTAGE: AIR — high above a valley\nIMPOSSIBLE: melted trees")
        self.assertFalse(ok)


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


class TestObstacleParsing(unittest.TestCase):
    def test_move_block_none(self):
        self.assertEqual(
            judge._parse_obstacles("VANTAGE: AIR\nNONE\nMOVE-BLOCK: NONE"), [])

    def test_move_block_subset(self):
        self.assertEqual(
            judge._parse_obstacles("a\nb\nMOVE-BLOCK: FORWARD, LEFT"),
            ["FORWARD", "LEFT"])

    def test_move_block_lowercase(self):
        self.assertEqual(judge._parse_obstacles("x\ny\nmove-block: right"),
                         ["RIGHT"])

    def test_move_block_missing_is_empty(self):
        self.assertEqual(judge._parse_obstacles("VANTAGE: AIR\nNONE"), [])

    def test_move_block_ignores_unknown_tokens(self):
        self.assertEqual(
            judge._parse_obstacles("a\nb\nMOVE-BLOCK: DOWN, BACKWARD"), [])


class TestMotionParsing(unittest.TestCase):
    def test_extracts_clean_clause(self):
        reply = ("VANTAGE: AIR\nNONE\nMOVE-BLOCK: NONE\n"
                 "MOTION: gliding forward over the spruce ridge toward the "
                 "frozen river bend, treeline staying razor sharp")
        self.assertIn("spruce ridge", judge._parse_motion(reply))

    def test_none_is_empty(self):
        self.assertEqual(
            judge._parse_motion("a\nb\nc\nMOTION: NONE"), "")

    def test_missing_line_is_empty(self):
        self.assertEqual(
            judge._parse_motion("VANTAGE: AIR\nNONE\nMOVE-BLOCK: NONE"), "")

    def test_volumetric_clause_rejected(self):
        # the load-bearing guardrail: a clause requesting volumetrics must be
        # dropped (-> class default) so cfg=1 never gets blob-spawning language
        for bad in ("drifting fog rolls over the valley",
                    "clouds swirl above the peak",
                    "snowfall streaks past the lens",
                    "mist thickens over the lake"):
            self.assertEqual(
                judge._parse_motion("a\nb\nc\nMOTION: " + bad), "",
                f"should reject: {bad}")

    def test_morphological_variants_rejected(self):
        # -y / -s / -ing forms of volumetrics must also be caught (the VLM
        # slipped "misty horizon" past the bare-noun denylist)
        for bad in ("misty horizon line", "over a foggy valley",
                    "snowy squalls sweep in", "cloudy ridge tops",
                    "hazy far shore"):
            self.assertEqual(
                judge._parse_motion("a\nb\nc\nMOTION: " + bad), "",
                f"should reject: {bad}")

    def test_solid_feature_words_allowed(self):
        # terrain/landform/winding/formation must NOT trip the denylist
        clause = ("panning left across winding terrain and rock formations "
                  "along the shoreline")
        self.assertEqual(judge._parse_motion("a\nb\nc\nMOTION: " + clause),
                         clause)

    def test_overlong_clause_rejected(self):
        self.assertEqual(
            judge._parse_motion("a\nb\nc\nMOTION: " + "ridge " * 60), "")


class TestJudgeImageEx(unittest.TestCase):
    def test_returns_blocked_and_motion(self):
        reply = ("VANTAGE: AIR\nNONE\nMOVE-BLOCK: FORWARD\n"
                 "MOTION: panning left along the granite ridge")
        with mock.patch.object(judge, "_chat", return_value=reply), \
             mock.patch.object(judge, "_b64_image", return_value="x"):
            ok, reason, blocked, motion = judge.judge_image_ex(FakeCfg(), "img.png")
        self.assertTrue(ok)
        self.assertEqual(blocked, ["FORWARD"])
        self.assertIn("granite ridge", motion)

    def test_ollama_error_blocks_nothing(self):
        with mock.patch.object(judge, "_chat", side_effect=OSError("boom")), \
             mock.patch.object(judge, "_b64_image", return_value="x"):
            ok, reason, blocked, motion = judge.judge_image_ex(FakeCfg(), "img.png")
        self.assertTrue(ok)
        self.assertEqual(blocked, [])
        self.assertEqual(motion, "")

    def test_judge_image_still_two_tuple(self):
        reply = "VANTAGE: AIR\nNONE\nMOVE-BLOCK: NONE\nMOTION: NONE"
        with mock.patch.object(judge, "_chat", return_value=reply), \
             mock.patch.object(judge, "_b64_image", return_value="x"):
            result = judge.judge_image(FakeCfg(), "img.png")
        self.assertEqual(len(result), 2)


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


class RiskCfg(FakeCfg):
    judge_fog_cover_max = 0.28
    judge_image_min_sharp = 60.0
    judge_min_struct_ratio = 1.0


class TestImageRiskGate(unittest.TestCase):
    def test_fog_bank_rejects_before_vlm(self):
        """A big fog mass must reject at the image stage — deterministically,
        without spending a VLM call (chat must not be reached)."""
        with mock.patch.object(judge, "image_risk_metrics",
                               return_value={"fog_cover": 0.31, "sharp": 400}), \
             mock.patch.object(judge, "_chat") as chat:
            ok, reason = judge.judge_image(RiskCfg(), "img.png")
        self.assertFalse(ok)
        self.assertIn("fog", reason)
        chat.assert_not_called()

    def test_blurry_still_rejects(self):
        with mock.patch.object(judge, "image_risk_metrics",
                               return_value={"fog_cover": 0.0, "sharp": 12}), \
             mock.patch.object(judge, "_chat") as chat:
            ok, reason = judge.judge_image(RiskCfg(), "img.png")
        self.assertFalse(ok)
        self.assertIn("soft", reason)
        chat.assert_not_called()

    def test_clean_image_falls_through_to_vlm(self):
        with mock.patch.object(judge, "image_risk_metrics",
                               return_value={"fog_cover": 0.18, "sharp": 640,
                                             "struct_ratio": 2.5}), \
             mock.patch.object(judge, "_b64_image", return_value="x"), \
             mock.patch.object(judge, "_chat", return_value="NONE"):
            ok, _ = judge.judge_image(RiskCfg(), "img.png")
        self.assertTrue(ok)

    def test_uniform_micro_texture_rejects(self):
        """A flower-meadow style frame (dense identical micro-elements, no
        coarse structure) rejects deterministically — Wan mushes it from
        frame 1 (the reported '240p blurry layer' master)."""
        with mock.patch.object(judge, "image_risk_metrics",
                               return_value={"fog_cover": 0.0, "sharp": 800,
                                             "struct_ratio": 0.66}), \
             mock.patch.object(judge, "_chat") as chat:
            ok, reason = judge.judge_image(RiskCfg(), "img.png")
        self.assertFalse(ok)
        self.assertIn("micro-texture", reason)
        chat.assert_not_called()

    def test_risk_metrics_failure_never_blocks(self):
        """CV risk gate is best-effort: an exception inside it must fall
        through to the VLM, not reject."""
        with mock.patch.object(judge, "image_risk_metrics",
                               side_effect=RuntimeError("cv2 boom")), \
             mock.patch.object(judge, "_b64_image", return_value="x"), \
             mock.patch.object(judge, "_chat", return_value="NONE"):
            ok, _ = judge.judge_image(RiskCfg(), "img.png")
        self.assertTrue(ok)


class CvOnlyCfg(FakeCfg):
    judge_video_mode = "cv"


class TestVideoCvOnlyMode(unittest.TestCase):
    def test_cv_mode_skips_vlm(self):
        """video_check='cv': CV gate runs, VLM is never called on masters —
        quality steering lives at the image stage."""
        with mock.patch.object(judge, "flicker_metrics",
                               return_value={"shimmer": 0.5,
                                             "tex_instab": 0.02,
                                             "frames": 200}), \
             mock.patch.object(judge, "_chat") as chat, \
             mock.patch.object(judge, "sample_frames") as sf:
            ok, reason = judge.judge_video(CvOnlyCfg(), "vid.mp4")
        self.assertTrue(ok)
        chat.assert_not_called()
        sf.assert_not_called()

    def test_cv_mode_still_rejects_flicker(self):
        with mock.patch.object(judge, "flicker_metrics",
                               return_value={"shimmer": 4.2,
                                             "tex_instab": 0.21,
                                             "frames": 200}):
            ok, reason = judge.judge_video(CvOnlyCfg(), "vid.mp4")
        self.assertFalse(ok)
        self.assertIn("flicker", reason)


class TestSmoothCurveAndDetailHold(unittest.TestCase):
    def test_smooth_curve_tracks_nonlinear_drift(self):
        """The de-drift target curve must follow a rise-then-fall shape (a
        linear fit does not — that bug left +10% saturation drift in real
        masters)."""
        from stillnorth.finish import _smooth_curve
        curve = [80.0] * 30 + [80 + i for i in range(15)] + [95.0] * 30
        sm = _smooth_curve(curve)
        self.assertLess(abs(sm[0] - 80), 2.0)
        self.assertLess(abs(sm[-1] - 95), 2.0)

    def test_detail_hold_config_defaults(self):
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
        self.assertTrue(c.detail_hold)
        self.assertEqual(c.detail_hold_max, 1.5)
        self.assertTrue(c.seed_restore)
        self.assertEqual(c.seed_restore_amount, 0.55)
        # judge risk-gate defaults
        self.assertEqual(c.judge_fog_cover_max, 0.28)
        self.assertEqual(c.judge_image_min_sharp, 60.0)
        # a source that fails every judge retry is dropped, not shipped
        self.assertEqual(c.judge_giveup, "abandon")
        self.assertEqual(c.judge_min_struct_ratio, 1.0)
        # video judge defaults to CV-only mode (VLM steering at image stage)
        self.assertEqual(c.judge_video_mode, "cv")
        self.assertTrue(c.judge_video_enabled)

    def test_detail_hold_skips_fabricated_periodic_frames(self):
        """A soft frame carrying a strong periodic micro-pattern (the SR
        chain-mail/knit fabrication) must NOT be amplified by detail_hold —
        unsharp on fabricated texture makes the artifact worse."""
        import numpy as np
        import cv2
        import tempfile
        from stillnorth.finish import _detail_hold

        d = tempfile.mkdtemp(prefix="snf_dh_")
        rng = np.random.RandomState(7)
        files = []
        # sharp organic frames set a high p80 baseline
        for i in range(4):
            f = rng.randint(0, 255, (256, 256, 3)).astype("uint8")
            p = os.path.join(d, f"a{i}.png")
            cv2.imwrite(p, f)
            files.append(p)
        # soft frame with a strong fine checkerboard = fabricated pattern
        yy, xx = np.mgrid[0:256, 0:256]
        checker = (((xx // 2 + yy // 2) % 2) * 40 + 100).astype("uint8")
        soft = cv2.merge([checker] * 3)
        p_bad = os.path.join(d, "bad.png")
        cv2.imwrite(p_bad, soft)
        files.append(p_bad)
        before = cv2.imread(p_bad).copy()

        class C:
            detail_hold_max = 1.5
        _detail_hold(files, C(), sigma=1.2)
        after = cv2.imread(p_bad)
        self.assertTrue(np.array_equal(before, after),
                        "periodic frame was amplified")

    def test_video_check_parses_bool_and_string(self):
        from stillnorth.config import Config
        base = {"comfy_server": "x", "ffmpeg": "ffmpeg",
                "comfy_input_dir": ".", "comfy_output_dir": ".",
                "workspace_subdir": "W", "server_host": "127.0.0.1",
                "server_port": 1, "render_timeout_img": 1,
                "render_timeout_vid": 1, "poll_seconds": 1,
                "image_upscale_mult": 2, "lastframe_upscale_mult": 4,
                "final_upscale_mult": 4, "fps": 16, "video_cq": 19,
                "nvenc": True, "upscale_denoise": "", "upscale_sharp": "",
                "upscale_grain": ""}
        for vc, mode, enabled in ((True, "full", True), (False, "off", False),
                                  ("cv", "cv", True), ("full", "full", True)):
            c = Config.__new__(Config)
            c.raw = dict(base, judge={"video_check": vc})
            with mock.patch("stillnorth.config._load",
                            side_effect=[c.raw, {},
                                         {"poses": [], "speed_by_pose": {},
                                          "classes": {}}]):
                c.reload()
            self.assertEqual(c.judge_video_mode, mode, vc)
            self.assertEqual(c.judge_video_enabled, enabled, vc)


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
        self.assertIn("contrast", c.final_grade)   # gentle grade on by default

    def test_judge_prompts_target_known_defects(self):
        """Prompt regressions: steam-column reject (stills) and blur/blob
        reject (video) must stay in the calibrated judge prompts."""
        self.assertIn("COLUMNS of steam", judge.IMAGE_PROMPT)
        self.assertIn("fog banks and valley mist are fine", judge.IMAGE_PROMPT)
        self.assertIn("blobs materialising", judge.VIDEO_PROMPT)
        self.assertIn("blurry", judge.VIDEO_PROMPT)

    def test_review_stage_dir(self):
        from stillnorth.config import STAGE_DIRS
        self.assertEqual(STAGE_DIRS["review"], "10_review")


class TestJudgeNumCtx(unittest.TestCase):
    """Regression coverage for the 2026-07-22 context-size bug: Ollama's
    default served context (4096) doesn't fit ONE image + IMAGE_PROMPT
    (measured 4223 tokens needed), so every image-judge call was silently
    HTTP-400ing and fail-opening through judge_image_ex's except-Exception,
    with no visible error anywhere. See
    docs/spikes/2026-07-22-batch-judge-benchmark.md for the full repro."""

    def test_covers_measured_single_image_need(self):
        self.assertGreater(judge._judge_num_ctx(1), 4223)
        self.assertEqual(judge._judge_num_ctx(1), 6144)   # proven-working, pinned

    def test_covers_measured_batch_needs(self):
        self.assertGreater(judge._judge_num_ctx(5), 10947)
        self.assertGreater(judge._judge_num_ctx(10), 18154)

    def test_monotonic_in_image_count(self):
        vals = [judge._judge_num_ctx(n) for n in (1, 3, 5, 10, 20)]
        self.assertEqual(vals, sorted(vals))

    def test_floor_never_below_6144(self):
        self.assertGreaterEqual(judge._judge_num_ctx(0), 6144)
        self.assertGreaterEqual(judge._judge_num_ctx(1), 6144)

    def test_capped_at_32768(self):
        # the spike benchmark hit a hard HTTP 413 around 20 images regardless
        # of num_ctx -- nothing should ever request more than this ceiling.
        self.assertEqual(judge._judge_num_ctx(1000), 32768)

    def test_chat_passes_num_ctx_option(self):
        captured = {}

        class FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return b'{"message": {"content": "NONE"}}'

        def fake_urlopen(req, timeout=None):
            captured["body"] = json.loads(req.data)
            return FakeResp()

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            judge._chat(FakeCfg(), "prompt", ["img1"])
        self.assertEqual(captured["body"]["options"]["num_ctx"],
                         judge._judge_num_ctx(1))


class TestBatchCoherencyParsing(unittest.TestCase):
    def test_matches_by_index(self):
        text = "1: NONE\n2: IMPOSSIBLE\n3: MINOR"
        self.assertEqual(judge.parse_batch_coherency(text, 3),
                         ["NONE", "IMPOSSIBLE", "MINOR"])

    def test_out_of_order_lines(self):
        # the model doesn't always answer in strict order -- the index prefix
        # must still map each severity to the right image.
        text = "2: MINOR\n1: NONE\n3: IMPOSSIBLE"
        self.assertEqual(judge.parse_batch_coherency(text, 3),
                         ["NONE", "MINOR", "IMPOSSIBLE"])

    def test_dropped_line_is_none_not_crash(self):
        text = "1: NONE\n3: IMPOSSIBLE"       # model skipped line 2
        self.assertEqual(judge.parse_batch_coherency(text, 3),
                         ["NONE", None, "IMPOSSIBLE"])

    def test_case_insensitive_and_punctuation_variants(self):
        text = "1) none\n2. impossible"
        self.assertEqual(judge.parse_batch_coherency(text, 2),
                         ["NONE", "IMPOSSIBLE"])


class TestJudgeImagesBatch(unittest.TestCase):
    def test_maps_severity_to_accept_reject(self):
        with mock.patch.object(judge, "_chat",
                               return_value="1: NONE\n2: IMPOSSIBLE\n3: MINOR"), \
             mock.patch.object(judge, "_b64_image", side_effect=lambda p: f"b64:{p}"):
            out = judge.judge_images_batch(FakeCfg(), ["a.png", "b.png", "c.png"])
        self.assertEqual(out["a.png"], (True, "NONE"))
        self.assertEqual(out["b.png"], (False, "IMPOSSIBLE"))   # only IMPOSSIBLE rejects
        self.assertEqual(out["c.png"], (True, "MINOR"))

    def test_unparsed_line_fails_open(self):
        with mock.patch.object(judge, "_chat", return_value="1: NONE"), \
             mock.patch.object(judge, "_b64_image", return_value="x"):
            out = judge.judge_images_batch(FakeCfg(), ["a.png", "b.png"])
        self.assertEqual(out["b.png"], (True, None))


class TestJudgeStillsPrefilter(unittest.TestCase):
    def test_fails_open_on_chat_error_never_drops_silently(self):
        logs = []
        with mock.patch.object(judge, "judge_images_batch",
                               side_effect=RuntimeError("ollama down")):
            accepted, rejected = judge.judge_stills_prefilter(
                FakeCfg(), ["a.png", "b.png"], batch_size=12, log=logs.append)
        self.assertEqual(accepted, ["a.png", "b.png"])
        self.assertEqual(rejected, [])
        self.assertTrue(any("unavailable" in m for m in logs))

    def test_preserves_order_and_collects_rejections(self):
        def fake_batch(cfg, paths):
            return {p: ((False, "IMPOSSIBLE") if i == 1 else (True, "NONE"))
                    for i, p in enumerate(paths)}

        with mock.patch.object(judge, "judge_images_batch", side_effect=fake_batch):
            accepted, rejected = judge.judge_stills_prefilter(
                FakeCfg(), ["a.png", "b.png", "c.png"], batch_size=12)
        self.assertEqual(accepted, ["a.png", "c.png"])
        self.assertEqual(rejected, [{"path": "b.png", "severity": "IMPOSSIBLE"}])

    def test_zero_batch_size_does_not_silently_drop_everything(self):
        # regression: range() step and the chunk slice must clamp the SAME
        # value, or batch_size<=0 makes every chunk empty and every still
        # vanishes with zero exception, zero log line.
        def fake_batch(cfg, paths):
            return {p: (True, "NONE") for p in paths}

        with mock.patch.object(judge, "judge_images_batch", side_effect=fake_batch):
            accepted, rejected = judge.judge_stills_prefilter(
                FakeCfg(), ["a.png", "b.png"], batch_size=0)
        self.assertEqual(accepted, ["a.png", "b.png"])
        self.assertEqual(rejected, [])


if __name__ == "__main__":
    unittest.main()
