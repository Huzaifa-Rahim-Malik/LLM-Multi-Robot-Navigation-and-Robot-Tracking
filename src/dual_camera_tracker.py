#!/usr/bin/env python3
"""
Dual-ZED YOLO tracker with global re-identification by world position.

Reads:
  - config/cam1_points.json, config/cam2_points.json   (from tools/calibrate_capture.py)
  - YOLO weights (.pt)

Pipeline (per frame):
  1. Each camera thread: grab frame -> YOLO -> XYZ from ZED -> affine xform to world(x,y)
  2. Main loop: pull latest detections from both cams
  3. Fuse near-duplicate detections ACROSS cameras (same physical robot seen by both)
  4. Hungarian-match fused observations to existing global tracks
  5. Update / spawn / retire tracks; same global ID survives camera handoff

Run:
  python dual_camera_tracker.py
or override defaults:
  python dual_camera_tracker.py --weights /path/to/yolo.pt
"""

import argparse
import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pyzed.sl as sl
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO


# ============================================================
#  Calibration: ZED camera-frame (X,Y) -> arena world (x,y)
# ============================================================

def fit_camera_to_world_transform(points):
    """
    Fits 2D affine:
        world_x = a*Xc + b*Yc + c
        world_y = d*Xc + e*Yc + f
    `points`: list of dicts with keys cam_X, cam_Y, world_x, world_y.
    Returns (a, b, c, d, e, f).
    """
    n = len(points)
    A = np.zeros((2 * n, 6))
    B = np.zeros((2 * n,))
    for i, p in enumerate(points):
        Xc, Yc = p["cam_X"], p["cam_Y"]
        wx, wy = p["world_x"], p["world_y"]
        A[i] = [Xc, Yc, 1, 0, 0, 0]
        B[i] = wx
        A[i + n] = [0, 0, 0, Xc, Yc, 1]
        B[i + n] = wy
    params, *_ = np.linalg.lstsq(A, B, rcond=None)
    return tuple(float(v) for v in params.flatten())


def load_transform(json_path):
    with open(json_path) as f:
        data = json.load(f)
    params = fit_camera_to_world_transform(data["points"])
    a, b, c, d, e, f = params
    errs = []
    for p in data["points"]:
        wx_pred = a * p["cam_X"] + b * p["cam_Y"] + c
        wy_pred = d * p["cam_X"] + e * p["cam_Y"] + f
        errs.append(np.hypot(wx_pred - p["world_x"], wy_pred - p["world_y"]))
    rms = float(np.sqrt(np.mean(np.array(errs) ** 2)))
    print(f"  {json_path}: SN={data['serial']}, "
          f"{len(data['points'])} pts, RMS={rms*100:.1f} cm, "
          f"max={max(errs)*100:.1f} cm")
    return params


def cam_to_world(params, X_cam, Y_cam):
    a, b, c, d, e, f = params
    return a * X_cam + b * Y_cam + c, d * X_cam + e * Y_cam + f


# ============================================================
#  Per-camera worker (thread): grab + detect + project to world
# ============================================================

@dataclass
class Detection:
    cam_id: int
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2 (in original frame coords)
    cx: int
    cy: int
    cam_X: float
    cam_Y: float
    cam_Z: float
    world_x: float
    world_y: float
    class_name: str
    conf: float
    track_id: Optional[int] = None    # filled by GlobalTracker.update()


class CameraWorker:
    def __init__(self, cam_id, serial, transform_params, model, model_lock, name):
        self.cam_id = cam_id
        self.serial = serial
        self.params = transform_params
        self.model = model
        self.model_lock = model_lock
        self.name = name
        self.zed: Optional[sl.Camera] = None
        self._latest: Optional[Tuple[np.ndarray, List[Detection]]] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def open(self):
        self.zed = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD720
        init.depth_mode = sl.DEPTH_MODE.ULTRA
        # init.sensors_required = False
 
        init.coordinate_units = sl.UNIT.METER
        init.set_from_serial_number(self.serial)
        
        err = self.zed.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED SN {self.serial}: {err}")
        print(f"  ✓ Opened {self.name} (SN {self.serial})")

    def start(self):
        self.thread.start()

    def stop(self):
        self._stop.set()
        self.thread.join(timeout=2)
        if self.zed is not None:
            self.zed.close()

    def get_latest(self):
        with self._lock:
            return self._latest

    def _run(self):
        runtime = sl.RuntimeParameters()
        image_mat = sl.Mat()
        pc_mat = sl.Mat()

        while not self._stop.is_set():
            if self.zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue
            self.zed.retrieve_image(image_mat, sl.VIEW.LEFT)
            self.zed.retrieve_measure(pc_mat, sl.MEASURE.XYZRGBA)
            frame = np.ascontiguousarray(image_mat.get_data()[:, :, :3])

            # YOLO inference (serialize across cameras to share GPU cleanly)
            with self.model_lock:
                results = self.model(frame, conf=0.25, verbose=False)

            detections: List[Detection] = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = self.model.names[cls]
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    ok, xyz = pc_mat.get_value(cx, cy)
                    X, Y, Z, _ = xyz
                    if not (np.isfinite(X) and np.isfinite(Y) and np.isfinite(Z)):
                        continue

                    wx, wy = cam_to_world(self.params, X, Y)
                    detections.append(Detection(
                        cam_id=self.cam_id,
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        cx=cx, cy=cy,
                        cam_X=float(X), cam_Y=float(Y), cam_Z=float(Z),
                        world_x=float(wx), world_y=float(wy),
                        class_name=name, conf=conf,
                    ))

            with self._lock:
                self._latest = (frame, detections)


# ============================================================
#  Global tracker: position-based assignment across cameras
# ============================================================

@dataclass
class GlobalTrack:
    id: int
    x: float
    y: float
    class_name: str
    last_seen: int
    hits: int = 1
    missed: int = 0
    history: List[Tuple[float, float]] = field(default_factory=list)


class GlobalTracker:
    """
    gating_dist : detection within this distance (m) of an existing track
                  may be associated with it
    merge_dist  : detections from DIFFERENT cameras within this distance (m)
                  are fused into a single observation (same physical robot)
    max_missed  : track is retired after this many consecutive frames with
                  no associated observation
    ema         : smoothing factor for track position update (0..1)
    """

    def __init__(self, gating_dist=0.6, merge_dist=0.4, max_missed=30, ema=0.5):
        self.gating = gating_dist
        self.merge = merge_dist
        self.max_missed = max_missed
        self.ema = ema
        self.tracks: List[GlobalTrack] = []
        self.next_id = 1
        self.frame_idx = 0

    def _fuse(self, dets_per_cam):
        """
        Group detections seen by multiple cameras into single observations.
        Same-camera detections are NEVER merged (already separate objects).
        Returns: list of (x, y, class_name, conf, refs)
                 refs is a list of (cam_idx, det_idx_within_cam).
        """
        flat = []
        for ci, dets in enumerate(dets_per_cam):
            for di, d in enumerate(dets):
                flat.append((ci, di, d))

        used = [False] * len(flat)
        observations = []
        for i, (ci_i, _, d_i) in enumerate(flat):
            if used[i]:
                continue
            cluster = [flat[i]]
            used[i] = True
            for j in range(i + 1, len(flat)):
                if used[j]:
                    continue
                ci_j, _, d_j = flat[j]
                if ci_j == ci_i:
                    continue                          # never merge within same cam
                if d_j.class_name != d_i.class_name:
                    continue
                if np.hypot(d_j.world_x - d_i.world_x,
                            d_j.world_y - d_i.world_y) <= self.merge:
                    cluster.append(flat[j])
                    used[j] = True

            ws = np.array([d.conf for _, _, d in cluster])
            xs = np.array([d.world_x for _, _, d in cluster])
            ys = np.array([d.world_y for _, _, d in cluster])
            observations.append((
                float(np.average(xs, weights=ws)),
                float(np.average(ys, weights=ws)),
                cluster[0][2].class_name,
                float(ws.max()),
                [(ci, di) for ci, di, _ in cluster],
            ))
        return observations

    def update(self, dets_per_cam):
        """Mutates each Detection's track_id in place; returns active tracks."""
        self.frame_idx += 1
        observations = self._fuse(dets_per_cam)

        matched_tracks, matched_obs = set(), set()

        if self.tracks and observations:
            T, O = len(self.tracks), len(observations)
            INF = 1e6
            cost = np.full((T, O), INF)
            for ti, t in enumerate(self.tracks):
                for oi, (ox, oy, ocls, _, _) in enumerate(observations):
                    if ocls != t.class_name:
                        continue
                    d = np.hypot(t.x - ox, t.y - oy)
                    if d <= self.gating:
                        cost[ti, oi] = d
            row_ind, col_ind = linear_sum_assignment(cost)
            for ti, oi in zip(row_ind, col_ind):
                if cost[ti, oi] >= INF / 2:
                    continue
                ox, oy, _, _, refs = observations[oi]
                t = self.tracks[ti]
                t.x = (1 - self.ema) * t.x + self.ema * ox
                t.y = (1 - self.ema) * t.y + self.ema * oy
                t.last_seen = self.frame_idx
                t.hits += 1
                t.missed = 0
                t.history.append((t.x, t.y))
                if len(t.history) > 60:
                    t.history.pop(0)
                for ci, di in refs:
                    dets_per_cam[ci][di].track_id = t.id
                matched_tracks.add(ti)
                matched_obs.add(oi)

        # New tracks for unmatched observations
        for oi, (ox, oy, ocls, _, refs) in enumerate(observations):
            if oi in matched_obs:
                continue
            track = GlobalTrack(
                id=self.next_id, x=ox, y=oy, class_name=ocls,
                last_seen=self.frame_idx,
            )
            track.history.append((ox, oy))
            self.next_id += 1
            self.tracks.append(track)
            for ci, di in refs:
                dets_per_cam[ci][di].track_id = track.id

        # Increment missed counter for unmatched tracks; retire stale ones
        for ti, t in enumerate(self.tracks):
            if ti not in matched_tracks:
                t.missed += 1
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

        return self.tracks


# ============================================================
#  Drawing
# ============================================================

def _color_for_id(tid):
    rng = np.random.default_rng(tid * 9871)
    h = float(rng.integers(0, 180))
    hsv = np.uint8([[[h, 200, 230]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_camera_view(frame, detections, n_tracks, cam_name):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 32), (40, 40, 40), -1)
    cv2.putText(frame,
                f"{cam_name}  |  detections: {len(detections)}  |  global tracks: {n_tracks}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    for d in detections:
        x1, y1, x2, y2 = d.bbox
        col = _color_for_id(d.track_id) if d.track_id is not None else (160, 160, 160)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        lbl = d.class_name
        cv2.putText(frame, lbl, (x1, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)


def draw_topdown(tracks, arena_w=8.0, arena_h=4.0, scale=80, margin=30):
    W = int(arena_w * scale + 2 * margin)
    H = int(arena_h * scale + 2 * margin)
    img = np.full((H, W, 3), 25, dtype=np.uint8)

    cv2.rectangle(img, (margin, margin), (W - margin, H - margin), (90, 90, 90), 2)

    for x in range(int(arena_w) + 1):
        u = int(margin + x * scale)
        cv2.line(img, (u, margin), (u, H - margin), (55, 55, 55), 1)
        cv2.putText(img, str(x), (u - 5, H - margin + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (170, 170, 170), 1)
    for y in range(int(arena_h) + 1):
        v = int(margin + y * scale)
        cv2.line(img, (margin, v), (W - margin, v), (55, 55, 55), 1)
        cv2.putText(img, str(y), (5, v + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (170, 170, 170), 1)

    # Origin
    cv2.circle(img, (margin, margin), 5, (0, 255, 255), -1)
    cv2.putText(img, "(0,0)", (margin + 8, margin - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    # Camera positions (signs as the user provided: cam1 at (-0.88, -0.42), cam2 at (3.72, -0.55))
    cams = [(-0.88, -0.42, "C1"), (3.72, -0.55, "C2")]
    for cx, cy, lbl in cams:
        u = int(margin + cx * scale)
        v = int(margin + cy * scale)
        cv2.drawMarker(img, (u, v), (255, 200, 0),
                       markerType=cv2.MARKER_TRIANGLE_DOWN,
                       markerSize=14, thickness=2)
        cv2.putText(img, lbl, (u + 8, v + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)

    for t in tracks:
        col = _color_for_id(t.id)
        for i in range(1, len(t.history)):
            p1 = (int(margin + t.history[i - 1][0] * scale),
                  int(margin + t.history[i - 1][1] * scale))
            p2 = (int(margin + t.history[i][0] * scale),
                  int(margin + t.history[i][1] * scale))
            cv2.line(img, p1, p2, col, 1)
        u = int(margin + t.x * scale)
        v = int(margin + t.y * scale)
        cv2.circle(img, (u, v), 9, col, -1)
        cv2.putText(img, f"#{t.id} {t.class_name}", (u + 11, v + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

    return img


# ============================================================
#  Main
# ============================================================

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam1-serial", type=int, default=35090194)
    ap.add_argument("--cam2-serial", type=int, default=31496664)
    ap.add_argument("--cam1-cal", type=str, default=os.path.join(_REPO_ROOT, "config", "cam1_points.json"))
    ap.add_argument("--cam2-cal", type=str, default=os.path.join(_REPO_ROOT, "config", "cam2_points.json"))
    ap.add_argument("--weights", type=str,
                    default=os.path.join(_REPO_ROOT, "models", "yolov8_zed_2class.pt"))
    ap.add_argument("--gating", type=float, default=0.6,
                    help="max distance (m) between detection and track to associate")
    ap.add_argument("--merge", type=float, default=0.4,
                    help="max distance (m) to fuse detections across cameras")
    ap.add_argument("--max-missed", type=int, default=30,
                    help="frames a track may persist with no detection before retiring")
    args = ap.parse_args()

    print("Loading calibrations...")
    p1 = load_transform(args.cam1_cal)
    p2 = load_transform(args.cam2_cal)

    print("Loading YOLO model...")
    model = YOLO(args.weights)
    model_lock = threading.Lock()

    print("Opening cameras...")
    cam1 = CameraWorker(0, args.cam1_serial, p1, model, model_lock, "Camera 1")
    cam2 = CameraWorker(1, args.cam2_serial, p2, model, model_lock, "Camera 2")
    cam1.open()
    cam2.open()
    cam1.start()
    cam2.start()

    tracker = GlobalTracker(
        gating_dist=args.gating,
        merge_dist=args.merge,
        max_missed=args.max_missed,
    )

    print("\n🚀 Running. ESC in any window to quit.\n")

    # Wait briefly for both cameras to produce a first frame
    t0 = time.time()
    while time.time() - t0 < 5:
        if cam1.get_latest() is not None and cam2.get_latest() is not None:
            break
        time.sleep(0.05)

    last_print = 0.0
    try:
        while True:
            l1 = cam1.get_latest()
            l2 = cam2.get_latest()
            if l1 is None or l2 is None:
                if cv2.waitKey(10) & 0xFF == 27:
                    break
                continue

            # Copy frames so we can draw freely; detections will be mutated by tracker
            f1 = l1[0].copy()
            f2 = l2[0].copy()
            d1 = l1[1]
            d2 = l2[1]

            tracks = tracker.update([d1, d2])

            draw_camera_view(f1, d1, len(tracks), "Camera 1")
            draw_camera_view(f2, d2, len(tracks), "Camera 2")
            top = draw_topdown(tracks)

            cv2.imshow("Camera 1", f1)
            cv2.imshow("Camera 2", f2)
            cv2.imshow("Arena top-down", top)

            now = time.time()
            if tracks and now - last_print > 0.25:
                summary = "  ".join(
                    f"ID{t.id}({t.x:+.2f},{t.y:+.2f})" for t in tracks
                )
                print(f"\rTracks: {summary}".ljust(120), end="", flush=True)
                last_print = now                                                                                                     

            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        print("\nShutting down...")
        cam1.stop()
        cam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()