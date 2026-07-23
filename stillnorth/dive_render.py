"""DepthFlow 'dive' render — executed by the EXTERNAL DepthFlow venv, never by the
project's own interpreter (DepthFlow is AGPL; keeping it in its own venv and
invoking this as a subprocess keeps the repo clear of AGPL). Kept dependency-free
of the stillnorth package on purpose so the dive venv needs only `depthflow`.

Motion = the docs 'Dolly' recipe: state.dolly = amp*(1-cos(cycle)) translates the
camera ray-origins THROUGH the estimated depth (near parallax > far = a real dive
into the scene), and 1-cos returns to 0 at start AND end -> the clip loops to its
own first frame by construction. See docs/spikes/2026-07-22-dive-motion-spike.md.

Usage (called by stillnorth.dive):
  python dive_render.py --image X.png --out Y.mp4 [--depth-out D.png]
     [--time 9.5] [--width 3840] [--height 2160] [--fps 16] [--ssaa 2.0]
     [--quality 100] [--dolly 1.0] [--parallax 0.6] [--focus 0.32] [--steady 0.30]
Prints: DEPTH_SEC=.. RENDER_SEC=.. TOTAL_SEC=.. OUT=..
"""
import argparse
import math
import time

import imageio.v3 as iio
import numpy as np
from attrs import define
from depthflow.scene import DepthScene
from depthflow.estimators.anything import DepthAnythingV2


@define
class Dive(DepthScene):
    """Dolly-in ('dive into the scene'): translate ray origins through depth."""
    dolly_amp: float = 1.0
    parallax: float = 0.6
    focus_v: float = 0.32
    steady_v: float = 0.30

    def update(self):
        self.state.height = self.parallax
        self.state.focus = self.focus_v
        self.state.steady = self.steady_v
        self.state.dolly = self.dolly_amp * (1.0 - math.cos(self.cycle))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth-out", default=None)
    ap.add_argument("--time", type=float, default=9.5)
    ap.add_argument("--width", type=int, default=3840)
    ap.add_argument("--height", type=int, default=2160)
    ap.add_argument("--fps", type=float, default=16.0)
    ap.add_argument("--ssaa", type=float, default=2.0)
    ap.add_argument("--quality", type=float, default=100.0)
    ap.add_argument("--dolly", type=float, default=1.0)
    ap.add_argument("--parallax", type=float, default=0.6)
    ap.add_argument("--focus", type=float, default=0.32)
    ap.add_argument("--steady", type=float, default=0.30)
    args = ap.parse_args()

    img = iio.imread(args.image)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    t0 = time.time()
    est = DepthAnythingV2()
    depth = est.estimate(img)
    t_depth = time.time() - t0

    if args.depth_out:
        d = np.asarray(depth, dtype=np.float64)
        d = (255.0 * (d - d.min()) / max(1e-6, (d.max() - d.min()))).astype("uint8")
        iio.imwrite(args.depth_out, d)

    scene = Dive(backend="headless", dolly_amp=args.dolly, parallax=args.parallax,
                 focus_v=args.focus, steady_v=args.steady)
    scene.input(image=img, depth=depth)
    try:
        scene.ffmpeg.h264(preset="medium", crf=16, tune="film")
    except Exception as e:
        print(f"WARN encoder-config failed, using default: {e}")

    kw = dict(output=args.out, time=args.time, width=args.width,
              height=args.height, quality=args.quality, ssaa=args.ssaa)
    t1 = time.time()
    try:
        scene.main(fps=args.fps, **kw)
    except TypeError:
        scene.main(**kw)
    t_render = time.time() - t1
    print(f"DEPTH_SEC={t_depth:.2f} RENDER_SEC={t_render:.2f} "
          f"TOTAL_SEC={time.time() - t0:.2f} OUT={args.out}")


if __name__ == "__main__":
    main()
