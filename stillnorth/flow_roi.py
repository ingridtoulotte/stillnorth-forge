"""ROI motion diagnostics for a clip -- optical-flow velocity and temporal
activity inside a fractional region.

Promoted from session scratchpad to the package because it now makes a real
production accept/reject call, not just a diagnostic: frozen-waterfall sources
render the falls as a near-static long-exposure sheet (Section 5.11), and the
ONLY reliable check is measuring the rendered clip (a textured-LOOKING still
does not guarantee flow -- Section 5.19).

TWO metrics, because they answer different questions:

* roi_flow -- mean optical-flow vector (camera-subtracted) in the ROI. Good for
  PAN glides. CAVEAT: under a Zoom-In / dolly pose the whole lower frame gains
  downward parallax, so a genuinely FROZEN falls master can still read
  mean_dy ~ +0.4 (measured on the frozen C_076 catalog master). Do NOT gate
  falls on raw mean_dy under a dolly pose -- use the activity ratio instead.

* roi_activity_ratio -- temporal pixel-change in a falls ROI divided by the same
  in a static-reference ROI (rock/cliff). Parallax moves BOTH ROIs, so the
  ratio cancels it: a flowing fall churns far more than the rock (ratio > 1,
  "livelier than the rest" -- 5.11 r2), a frozen silky sheet churns far LESS
  than the parallax-moved rock (ratio ~ 0.16 on the frozen C_076 master). This
  is the reliable frozen-vs-flowing gate.

CLI:
  python -m stillnorth.flow_roi <clip> <fx> <fy> <fw> <fh> [label]
  python -m stillnorth.flow_roi activity <clip> <fx fy fw fh> <rfx rfy rfw rfh>
  (all coords are fractions of frame size, 0..1)
"""
import json
import sys

SCALE_W = 960


def _load_gray(path, scale_w=SCALE_W):
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        f = cv2.resize(f, (scale_w, int(h * scale_w / w)),
                       interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    if len(frames) < 2:
        raise ValueError(f"need >=2 frames, got {len(frames)}: {path}")
    return frames, fps


def _roi_px(shape, roi):
    h, w = shape
    x0, y0 = int(roi[0] * w), int(roi[1] * h)
    x1, y1 = min(w, x0 + int(roi[2] * w)), min(h, y0 + int(roi[3] * h))
    return x0, y0, x1, y1


def roi_flow(path, fx, fy, fw, fh, scale_w=SCALE_W):
    """Mean optical-flow vector (camera-subtracted) inside the fractional ROI.
    +dy = downward, -dy = upward, px/frame at scale_w. See module caveat about
    parallax under dolly poses before gating on this."""
    import cv2
    import numpy as np
    frames, fps = _load_gray(path, scale_w)
    x0, y0, x1, y1 = _roi_px(frames[0].shape, (fx, fy, fw, fh))
    dxs, dys, mags = [], [], []
    for i in range(len(frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            frames[i], frames[i + 1], None, 0.5, 3, 21, 3, 5, 1.2, 0)
        cam = np.median(flow.reshape(-1, 2), axis=0)
        roi = flow[y0:y1, x0:x1] - cam
        dxs.append(float(roi[..., 0].mean()))
        dys.append(float(roi[..., 1].mean()))
        mags.append(float(np.linalg.norm(roi, axis=-1).mean()))
    dxs, dys, mags = np.array(dxs), np.array(dys), np.array(mags)
    per_sec_dy = [round(float(dys[int(s * fps):int((s + 1) * fps)].mean()), 3)
                  for s in range(int(len(dys) / fps))]
    return {
        "clip": path.replace("\\", "/").split("/")[-1],
        "roi_px": [x0, y0, x1 - x0, y1 - y0],
        "frames": len(frames),
        "mean_dx": round(float(dxs.mean()), 3),
        "mean_dy": round(float(dys.mean()), 3),
        "mean_mag": round(float(mags.mean()), 3),
        "per_sec_dy": per_sec_dy,
    }


def _activity(frames, roi):
    import numpy as np
    x0, y0, x1, y1 = _roi_px(frames[0].shape, roi)
    g = [f[y0:y1, x0:x1].astype(np.float32) for f in frames]
    return float(np.mean([np.abs(g[i + 1] - g[i]).mean()
                          for i in range(len(g) - 1)]))


def roi_activity_ratio(path, roi, ref_roi, scale_w=SCALE_W):
    """Temporal pixel-change in `roi` (the falls) over the same in `ref_roi`
    (a static rock/cliff). Parallax moves both, so the ratio cancels it:
    ratio > 1 = the falls churn more than the rock (FLOWING), ratio well
    below 1 = a near-static long-exposure sheet (FROZEN). Reliable frozen-vs-
    flowing gate where roi_flow's mean_dy is parallax-confounded."""
    frames, _ = _load_gray(path, scale_w)
    fa, ra = _activity(frames, roi), _activity(frames, ref_roi)
    ratio = fa / max(ra, 1e-6)
    return {
        "clip": path.replace("\\", "/").split("/")[-1],
        "frames": len(frames),
        "roi_activity": round(fa, 3),
        "ref_activity": round(ra, 3),
        "ratio": round(ratio, 3),
        "flowing": ratio > 1.0,
    }


def main(argv):
    if argv and argv[0] == "activity":
        a = argv[1:]
        out = roi_activity_ratio(
            a[0], (float(a[1]), float(a[2]), float(a[3]), float(a[4])),
            (float(a[5]), float(a[6]), float(a[7]), float(a[8])))
        print(json.dumps(out))
        return
    a = argv
    out = roi_flow(a[0], float(a[1]), float(a[2]), float(a[3]), float(a[4]))
    if len(a) > 5:
        out["label"] = a[5]
    print(json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1:])
