"""
ZED 2i Calibration Capture
==========================
Captures (arena world x,y) ↔ (ZED camera X,Y,Z) point pairs that we'll later
fit an affine transform to, so each camera can report robot positions in
arena coordinates.

Run TWICE, once per camera:

    python calibrate_capture.py --serial 35090194 --output cam1_points.json
    python calibrate_capture.py --serial 31496664 --output cam2_points.json

Workflow:
  STEP 1 (terminal):  type the world (x, y) points you'll click, in order
                      pick 8-12 markers spread across that camera's view
                      (corners + middle, near + far)
  STEP 2 (window):    click each marker in the order you listed
                      [u] undoes last point  [s] saves & exits  [ESC] aborts
"""

import argparse
import json

import cv2
import numpy as np
import pyzed.sl as sl


def collect_world_points():
    """Prompt user for the list of world (x, y) markers they'll click, in order."""
    print("\n" + "=" * 60)
    print("STEP 1 — Enter world (x, y) markers you'll click")
    print("=" * 60)
    print("Format: x,y per line  (e.g. '0,0' or '2.5,1')")
    print("Empty line to finish.  Minimum 6 points, 8-12 recommended.")
    print("Pick markers spread across the camera's view (near/far/left/right).")
    print()

    worlds = []
    while True:
        line = input(f"  Point #{len(worlds) + 1} (x,y) > ").strip()
        if not line:
            if len(worlds) >= 6:
                break
            print(f"  Need at least 6 points; you have {len(worlds)}.")
            continue
        try:
            parts = [float(v) for v in line.replace(" ", "").split(",")]
            if len(parts) != 2:
                raise ValueError
            worlds.append(tuple(parts))
        except ValueError:
            print("  Invalid format. Use 'x,y' e.g. '2.0,1.0'")

    print(f"\n✓ Will capture {len(worlds)} points: {worlds}\n")
    return worlds


def main():
    ap = argparse.ArgumentParser(description="Capture ZED ↔ arena calibration points")
    ap.add_argument("--serial", type=int, required=True, help="ZED camera serial number")
    ap.add_argument("--output", type=str, required=True, help="Output JSON file path")
    args = ap.parse_args()

    worlds = collect_world_points()

    # ---- Open the specific ZED camera by serial number ----
    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER
    init.set_from_serial_number(args.serial)

    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"❌ Failed to open ZED with SN {args.serial}: {err}")
        return
    print(f"✅ Opened ZED SN {args.serial}")

    runtime = sl.RuntimeParameters()
    image_mat = sl.Mat()
    pc_mat = sl.Mat()

    captured = []
    pending_pixel = [None]  # list for mutable closure

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pending_pixel[0] = (x, y)

    cv2.namedWindow("Calibration capture", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Calibration capture", on_mouse)

    print("\n" + "=" * 60)
    print("STEP 2 — Click each marker in the order you listed")
    print("=" * 60)
    print("  click   capture current target marker")
    print("  [u]     undo last captured point")
    print("  [s]     save and exit")
    print("  [ESC]   abort without saving")
    print("=" * 60 + "\n")

    while True:
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_mat, sl.VIEW.LEFT)
            zed.retrieve_measure(pc_mat, sl.MEASURE.XYZRGBA)
            frame = np.copy(image_mat.get_data()[:, :, :3])

            # Process pending click
            if pending_pixel[0] is not None and len(captured) < len(worlds):
                u, v = pending_pixel[0]
                pending_pixel[0] = None
                ok, xyz = pc_mat.get_value(u, v)
                X, Y, Z, _ = xyz
                if not (np.isfinite(X) and np.isfinite(Y) and np.isfinite(Z)):
                    print(f"⚠ Invalid depth at ({u},{v}). Try a clearer spot on the marker.")
                else:
                    wx, wy = worlds[len(captured)]
                    captured.append({
                        "world_x": wx, "world_y": wy,
                        "cam_X": float(X), "cam_Y": float(Y), "cam_Z": float(Z),
                        "pixel_u": int(u), "pixel_v": int(v),
                    })
                    print(f"  [{len(captured)}/{len(worlds)}] world({wx},{wy}) → "
                          f"cam(X={X:+.3f}, Y={Y:+.3f}, Z={Z:+.3f}) at pixel({u},{v})")

            # Draw previously captured points
            for pt in captured:
                cv2.circle(frame, (pt["pixel_u"], pt["pixel_v"]), 6, (0, 255, 0), -1)
                cv2.putText(frame, f"({pt['world_x']:g},{pt['world_y']:g})",
                            (pt["pixel_u"] + 8, pt["pixel_v"] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # HUD
            n_done, n_total = len(captured), len(worlds)
            if n_done < n_total:
                tx, ty = worlds[n_done]
                cv2.putText(frame, f"CLICK ON marker ({tx:g}, {ty:g})  —  {n_done}/{n_total}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(frame, f"DONE — press [s] to save  ({n_done}/{n_total})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "[s]ave  [u]ndo  [ESC]abort", (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Calibration capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("Aborted, nothing saved.")
            break
        elif key == ord("u"):
            if captured:
                removed = captured.pop()
                print(f"  ↶ Undo: removed ({removed['world_x']},{removed['world_y']})")
        elif key == ord("s"):
            if len(captured) < 4:
                print(f"⚠ Need at least 4 captured points to save (have {len(captured)})")
                continue
            with open(args.output, "w") as f:
                json.dump({"serial": args.serial, "points": captured}, f, indent=2)
            print(f"\n✅ Saved {len(captured)} points to {args.output}")
            break

    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()