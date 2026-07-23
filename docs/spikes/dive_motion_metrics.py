"""Dive-motion spike metrics -- mirrors stillnorth's own measurement conventions.

Conventions borrowed verbatim from the production code so the numbers are
comparable to the rest of the project:
  * sharpness  = variance of the Laplacian (finish._sharp_metric, judge._sharp)
  * motion     = mean Farneback optical-flow magnitude (finish._flow, flow_roi)
  * ROI flow   = camera-median-subtracted per-ROI mean flow (flow_roi.roi_flow),
                 so a global pan/zoom is removed and only *relative* motion shows
  * depth diff = foreground-ROI flow / background-ROI flow. A real parallax/dive
                 moves the near patch faster than the far patch (ratio >> 1); a
                 flat Ken-Burns pan/zoom moves both patches at ~the same rate
                 (ratio ~ 1). This is THE number that separates a dive from a
                 zoom-in disguise (brief section 4).

Resolution note: Laplacian variance is resolution-dependent, so every frame and
the source still are resized to a common width (default 1280, the same width the
production struct_ratio gate is calibrated at) BEFORE the Laplacian, making the
source-still-vs-output sharpness ratio a fair like-for-like comparison.

Not wired into the package or CI -- a self-contained spike tool. Pure helpers
carry a PASS/FAIL _run() self-test in the project's stdlib-test spirit (the cv2
parts need opencv+numpy, so the self-test is guarded to skip cleanly if absent).

CLI:
  python dive_metrics.py sharp <img|video>
  python dive_metrics.py flow  <video> fx fy fw fh [--cam]
  python dive_metrics.py diff  <video> fgx fgy fgw fgh  bgx bgy bgw bgh
  python dive_metrics.py retention <video> <source_img>
  python dive_metrics.py selftest
(all ROI coords are fractions of frame size, 0..1)
"""
import json
import sys

LAP_W = 1280      # resolution-normalise width for Laplacian comparisons
FLOW_W = 960      # flow_roi.SCALE_W -- match the production ROI-flow convention
# Farneback params: identical to flow_roi.roi_flow (0.5,3,21,3,5,1.2,0)
FB = dict(pyr_scale=0.5, levels=3, winsize=21, iterations=3,
          poly_n=5, poly_sigma=1.2, flags=0)


def _cv():
    import cv2
    import numpy as np
    return cv2, np


# ---------------------------------------------------------------- sharpness
def _lap_var(gray):
    cv2, _ = _cv()
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _to_gray_w(bgr, w=LAP_W):
    cv2, _ = _cv()
    h, ww = bgr.shape[:2]
    g = cv2.resize(bgr, (w, int(h * w / ww)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)


def sharp_image(path, w=LAP_W):
    cv2, _ = _cv()
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise IOError(f"cannot read image {path}")
    return _lap_var(_to_gray_w(im, w))


def sharp_video_mid(path, w=LAP_W, lo=0.3, hi=0.7):
    """Mean Laplacian variance over the middle [lo,hi] fraction of frames."""
    cv2, np = _cv()
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    vals, i = [], 0
    a, b = int(n * lo), int(n * hi) if n else (0, 10 ** 9)
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if a <= i <= b:
            vals.append(_lap_var(_to_gray_w(fr, w)))
        i += 1
    cap.release()
    return float(np.mean(vals)) if vals else 0.0


# ---------------------------------------------------------------- ROI flow
def _load_gray_seq(path, w=FLOW_W):
    cv2, _ = _cv()
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        h, ww = f.shape[:2]
        f = cv2.resize(f, (w, int(h * w / ww)), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    if len(frames) < 2:
        raise ValueError(f"need >=2 frames, got {len(frames)}")
    return frames


def _roi_px(shape, roi):
    h, w = shape
    x0, y0 = int(roi[0] * w), int(roi[1] * h)
    x1, y1 = min(w, x0 + int(roi[2] * w)), min(h, y0 + int(roi[3] * h))
    return x0, y0, x1, y1


def roi_flow_mag(path, roi, cam_subtract=True, w=FLOW_W, mid=(0.3, 0.7)):
    """Mean optical-flow magnitude inside a fractional ROI, over the middle band
    of frames. cam_subtract removes the per-frame median flow (global camera
    move) so what remains is motion *relative* to the frame -- flow_roi's trick."""
    cv2, np = _cv()
    frames = _load_gray_seq(path, w)
    n = len(frames)
    a, b = int(n * mid[0]), int(n * mid[1])
    x0, y0, x1, y1 = _roi_px(frames[0].shape, roi)
    mags = []
    for i in range(max(0, a), min(n - 1, b)):
        fl = cv2.calcOpticalFlowFarneback(frames[i], frames[i + 1], None,
                                          FB["pyr_scale"], FB["levels"],
                                          FB["winsize"], FB["iterations"],
                                          FB["poly_n"], FB["poly_sigma"],
                                          FB["flags"])
        if cam_subtract:
            cam = np.median(fl.reshape(-1, 2), axis=0)
            fl = fl - cam
        roi_fl = fl[y0:y1, x0:x1]
        mags.append(float(np.linalg.norm(roi_fl, axis=-1).mean()))
    return float(np.mean(mags)) if mags else 0.0


def depth_diff(path, fg, bg, w=FLOW_W):
    """foreground/background flow ratio, raw and camera-subtracted.
    ratio_raw    -- absolute displacement (near should move more in a dolly).
    ratio_rel    -- after removing the global camera move (isolates parallax)."""
    fg_raw = roi_flow_mag(path, fg, cam_subtract=False, w=w)
    bg_raw = roi_flow_mag(path, bg, cam_subtract=False, w=w)
    fg_rel = roi_flow_mag(path, fg, cam_subtract=True, w=w)
    bg_rel = roi_flow_mag(path, bg, cam_subtract=True, w=w)
    return {
        "fg_raw": round(fg_raw, 4), "bg_raw": round(bg_raw, 4),
        "ratio_raw": round(fg_raw / max(bg_raw, 1e-6), 3),
        "fg_rel": round(fg_rel, 4), "bg_rel": round(bg_rel, 4),
        "ratio_rel": round(fg_rel / max(bg_rel, 1e-6), 3),
    }


# ---------------------------------------------------------------- self-test
def _run():
    """Pure-logic checks that don't need real media (opencv/numpy still needed
    for the synthetic frames). Prints PASS/FAIL per the project's test spirit."""
    ok = True
    try:
        _cv()
    except Exception as e:  # opencv/numpy absent -- skip, don't fail the spike
        print(f"SKIP selftest (opencv/numpy absent: {e})")
        return True
    cv2, np = _cv()

    # 1. _roi_px maps fractions to pixels correctly.
    x0, y0, x1, y1 = _roi_px((100, 200), (0.5, 0.5, 0.25, 0.25))
    c1 = (x0, y0, x1, y1) == (100, 50, 150, 75)
    print(("PASS" if c1 else "FAIL") + " roi_px fraction->pixel")
    ok &= c1

    # 2. a high-frequency checkerboard has far higher Laplacian variance than a
    #    flat grey field (the sharpness metric responds to detail).
    flat = np.full((256, 256), 128, np.uint8)
    chk = np.indices((256, 256)).sum(0) % 2 * 255
    c2 = _lap_var(chk.astype(np.uint8)) > 50 * _lap_var(flat)
    print(("PASS" if c2 else "FAIL") + " laplacian: checker >> flat")
    ok &= c2

    # 3. a synthetic clip where a foreground strip translates fast and the
    #    background is static must yield ratio_raw >> 1 (the dive signal).
    import tempfile
    import os
    frames = []
    for t in range(12):
        f = np.zeros((180, 320, 3), np.uint8)
        f[20:60, :, :] = 60                                   # static bg band
        # moving fg band: a bright bar sliding right, fresh texture each frame
        x = (t * 12) % 300
        f[120:160, x:x + 20, :] = 255
        frames.append(f)
    td = tempfile.mkdtemp()
    vp = os.path.join(td, "syn.mp4")
    vw = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"mp4v"), 12, (320, 180))
    for f in frames:
        vw.write(f)
    vw.release()
    try:
        d = depth_diff(vp, fg=(0.0, 0.66, 1.0, 0.25), bg=(0.0, 0.11, 1.0, 0.22))
        c3 = d["ratio_raw"] > 3.0
        print(("PASS" if c3 else "FAIL") + f" depth_diff: fg-moving ratio_raw={d['ratio_raw']} (>3)")
        ok &= c3
    finally:
        try:
            os.remove(vp)
            os.rmdir(td)
        except OSError:
            pass
    print("ALL PASS" if ok else "SOME FAIL")
    return ok


# ---------------------------------------------------------------- CLI
def main(argv):
    if not argv or argv[0] in ("selftest", "test"):
        sys.exit(0 if _run() else 1)
    cmd, a = argv[0], argv[1:]
    if cmd == "sharp":
        p = a[0]
        v = sharp_image(p) if p.lower().endswith((".png", ".jpg", ".jpeg")) \
            else sharp_video_mid(p)
        print(json.dumps({"sharp": round(v, 2), "path": p}))
    elif cmd == "flow":
        cam = "--cam" in a
        a = [x for x in a if x != "--cam"]
        v = roi_flow_mag(a[0], tuple(float(x) for x in a[1:5]), cam_subtract=cam)
        print(json.dumps({"flow_mag": round(v, 4), "cam_subtract": cam}))
    elif cmd == "diff":
        d = depth_diff(a[0], tuple(float(x) for x in a[1:5]),
                       tuple(float(x) for x in a[5:9]))
        print(json.dumps(d))
    elif cmd == "retention":
        vid, img = a[0], a[1]
        sv, si = sharp_video_mid(vid), sharp_image(img)
        print(json.dumps({"vid_sharp": round(sv, 2), "src_sharp": round(si, 2),
                          "retention": round(sv / max(si, 1e-6), 3)}))
    else:
        print(f"unknown command {cmd}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main(sys.argv[1:])
