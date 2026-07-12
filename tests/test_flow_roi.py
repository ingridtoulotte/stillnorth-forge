"""flow_roi: the frozen-falls accept/reject signals (roi_flow + activity ratio).

Skips cleanly when opencv/numpy are absent (keeps the stdlib-only suite green
on a box without them). roi_flow: a textured patch translating DOWNWARD reads
positive mean_dy. roi_activity_ratio: a churning falls ROI over a static
reference reads ratio > 1 (flowing); a static falls ROI over a moving reference
reads ratio < 1 (frozen) -- the parallax-robust discrimination it exists for.
"""
import os
import tempfile
import unittest

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from stillnorth import flow_roi


def _writer(path, h=240, w=320):
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 16.0, (w, h))


def _make_moving_patch(path, n=32, h=240, w=320, dy=3):
    rng = np.random.RandomState(0)
    bg = rng.randint(40, 90, (h, w), dtype=np.uint8)
    patch = rng.randint(150, 255, (60, 90), dtype=np.uint8)
    vw = _writer(path, h, w)
    for i in range(n):
        frame = bg.copy()
        y = 30 + dy * i
        frame[y:y + 60, 120:210] = patch
        vw.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    vw.release()


def _make_activity_clip(path, falls_churns, n=32, h=240, w=320):
    """falls region (x120-210) either churns (fresh noise each frame) or is
    static; the reference corner (top-left) does the opposite via a moving
    block, so the two cases invert the activity ratio."""
    rng = np.random.RandomState(1)
    bg = rng.randint(60, 120, (h, w), dtype=np.uint8)
    static_falls = rng.randint(120, 220, (200, 90), dtype=np.uint8)
    vw = _writer(path, h, w)
    for i in range(n):
        frame = bg.copy()
        if falls_churns:                       # flowing: falls churn, ref static
            frame[24:216, 120:210] = rng.randint(120, 255, (192, 90), dtype=np.uint8)
        else:                                  # frozen: falls static, ref moves
            frame[24:216, 120:210] = static_falls[:192]
            x = 4 + (i % 5) * 8
            frame[8:40, x:x + 32] = 255
        vw.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    vw.release()


class TestRoiFlow(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="snf_flow_")
        self.clip = os.path.join(self.d, "c.mp4")
        _make_moving_patch(self.clip)

    def test_moving_roi_reads_downward(self):
        r = flow_roi.roi_flow(self.clip, 0.35, 0.10, 0.35, 0.80)
        self.assertGreater(r["mean_dy"], 0.3, f"expected downward flow, got {r}")
        self.assertGreater(r["frames"], 2)

    def test_static_roi_reads_near_zero(self):
        r = flow_roi.roi_flow(self.clip, 0.0, 0.0, 0.20, 0.20)
        self.assertLess(abs(r["mean_dy"]), 0.3, f"expected ~0 flow, got {r}")

    def test_too_short_raises(self):
        one = os.path.join(self.d, "one.mp4")
        _make_moving_patch(one, n=1)
        with self.assertRaises(ValueError):
            flow_roi.roi_flow(one, 0.0, 0.0, 1.0, 1.0)


class TestActivityRatio(unittest.TestCase):
    FALLS = (0.375, 0.10, 0.28, 0.80)
    REF = (0.0, 0.0, 0.20, 0.20)

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="snf_act_")

    def test_churning_falls_reads_flowing(self):
        p = os.path.join(self.d, "flow.mp4")
        _make_activity_clip(p, falls_churns=True)
        r = flow_roi.roi_activity_ratio(p, self.FALLS, self.REF)
        self.assertGreater(r["ratio"], 1.0, f"expected flowing, got {r}")
        self.assertTrue(r["flowing"])

    def test_static_falls_reads_frozen(self):
        p = os.path.join(self.d, "frozen.mp4")
        _make_activity_clip(p, falls_churns=False)
        r = flow_roi.roi_activity_ratio(p, self.FALLS, self.REF)
        self.assertLess(r["ratio"], 0.7, f"expected frozen, got {r}")
        self.assertFalse(r["flowing"])


if __name__ == "__main__":
    unittest.main()
