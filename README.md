# Multi-Robot Navigation & Tracking — TurtleBot 4 leader + mini swarm follower

Extension of [LLM-Based-Navigation-Using-Turtlebot4](https://github.com/Huzaifa-Rahim-Malik/LLM-Based-Navigation-Using-Turtlebot4)
to a **leader–follower, multi-robot setup** in an indoor arena. A TurtleBot 4 (the leader) is driven by
natural-language text/voice commands; two ceiling-mounted ZED 2i cameras track every robot in arena
coordinates; and the leader publishes its arrival position so the custom
[Mini Swarm Vehicle](https://github.com/Huzaifa-Rahim-Malik/Mini_Swarm_vehicle) (the follower) can be sent to it.

The base project has **no camera feed and no object tracking**. This repo adds exactly that, plus the pose
hand-off to the second robot. Everything that is only about LLM/STT/TTS evaluation, result videos, install
scripts and the Piper voice model stays in the base repo and is **not duplicated here**.

## What is new compared with the base repo

| Area | File | What it adds |
|---|---|---|
| Global tracking | `src/dual_camera_tracker.py` | Two ZED 2i cameras → YOLOv8 detection → depth (XYZ) → affine transform to arena (x, y) → cross-camera fusion → Hungarian matching to persistent global track IDs (same ID survives camera hand-off) |
| Calibration | `tools/calibrate_capture.py`, `config/cam*_points.json` | Click-based capture of (arena x,y) ↔ (ZED X,Y,Z) pairs per camera; the tracker fits the affine transform from them |
| Detector | `models/yolov8_zed_2class.pt` | 2-class YOLOv8 model trained for the ZED views |
| Robot → follower hand-off | `src/pose_publisher.py` | `/robot/pose` (10 Hz) and `/robot_pos_data` (`"x:<val> y:<val>"`, published once on arrival) |
| Multi-robot GUI | `src/gui.py` | 3-column Tk GUI: chat · Camera 1 / arena tracking / leader sensor data (`/odom`, `/scan`, `/imu`) · Camera 2 / locations / controls |
| Cross-domain link | `ros/domain_bridge.yaml` | Bridges `/robot_pos_data` between ROS domain 23 (leader) and 35 (follower side) |
| Network config | `ros/env.sh`, `ros/fastdds_pc.example.xml`, `ros/cyclonedds_pc.xml` | Per-terminal ROS environment and Fast DDS unicast discovery for the PC ↔ robot Wi-Fi link |

The language-understanding pipeline (`intent_detector.py`, `location_extractor.py`, `response_generator.py`,
`voice_input.py`, `voice_output.py`, `robot_navigator.py`, `location_manager.py`) is the base project's. It is
included only so this repo runs on its own; `chat_interface.py` and `main.py` are the versions that call the
pose publisher.

## Architecture

```
   speech / text ──► Whisper ──► intent + location (rules, Ollama LLM fallback) ──► Nav2 goal
                                                                                      │
                  ┌───────────────────────────────────────────────────────────────────▼─────────┐
                  │ Leader: TurtleBot 4  (ROS 2 Jazzy, domain 23)                                 │
                  │   /odom  /scan  /imu ──► GUI sensor panel                                     │
                  │   on arrival ──► pose_publisher ──► /robot/pose, /robot_pos_data "x:.. y:.."  │
                  └───────────────────────────────────────────────┬─────────────────────────────┘
                                                                  │ domain_bridge  (23 ⇄ 35)
                                                                  ▼
                                              Follower: Mini Swarm Vehicle (domain 35)

   ZED 2i #1 ─┐
              ├─► YOLOv8 ─► depth XYZ ─► affine ─► fuse ─► Hungarian tracks ─► arena view in GUI
   ZED 2i #2 ─┘
```

The follower's own ROS-side code is not part of this repo; its hardware, Altium design and Arduino firmware
are in the Mini_Swarm_vehicle repo.

## Repository layout

```
src/        application code (run from here)
  gui.py                  multi-robot GUI (entry point)
  main.py                 terminal-only entry point (no GUI, no cameras)
  chat_interface.py       chat loop; notifies the pose publisher on navigate / arrive
  pose_publisher.py       /robot/pose and /robot_pos_data
  dual_camera_tracker.py  dual-ZED YOLO tracker (also runnable standalone)
  locations.yaml          named goals for the arena map
  + base-project modules  intent_detector, location_extractor, location_manager,
                          response_generator, robot_navigator, voice_input, voice_output
config/     cam1_points.json, cam2_points.json     camera ↔ arena calibration points
models/     yolov8_zed_2class.pt
maps/       robotic_arena.pgm / .yaml              map used by this setup
tools/      calibrate_capture.py
ros/        env.sh, fastdds_pc.example.xml, cyclonedds_pc.xml, domain_bridge.yaml
```

## Requirements

- Ubuntu 24.04, ROS 2 Jazzy, TurtleBot 4 with Nav2 + SLAM Toolbox, `ros-jazzy-domain-bridge`
- 2 × ZED 2i with the Stereolabs ZED SDK and `pyzed`
- Ollama server (for the LLM fallback), NVIDIA GPU recommended for YOLO + Whisper
- `pip install -r requirements.txt`

## Quick start

```bash
# 0. One-time: voice models and robot bring-up come from the base repo
#    (scripts/install_voice.sh, scripts/install_piper.sh, scripts/robot_launch.sh).
#    Use maps/robotic_arena.yaml from this repo for the map.

# 1. Network config
cp ros/fastdds_pc.example.xml ros/fastdds_pc.xml     # then put in your PC and robot IPs
source ros/env.sh                                    # in EVERY terminal that talks to the robot

# 2. Start the robot stack (base repo: scripts/robot_launch.sh), then:
cd src
python3 gui.py                    # GUI + dual-camera tracking
python3 gui.py --text-only        # no voice
# python3 main.py                 # terminal only

# 3. Hand-off to the follower domain (separate terminal)
ros2 run domain_bridge domain_bridge ros/domain_bridge.yaml
```

Run the tracker alone: `python3 src/dual_camera_tracker.py [--weights ...] [--gating 0.6] [--merge 0.4]`.

### Camera calibration (once per camera, or after moving a camera)

```bash
python3 tools/calibrate_capture.py --serial 35090194 --output config/cam1_points.json
python3 tools/calibrate_capture.py --serial 31496664 --output config/cam2_points.json
```

Enter 8–12 arena marker coordinates, then click each marker in the ZED image in the same order.
The ZED serial numbers are set in `src/gui.py` (`CameraWorker(0, 35090194, ...)`) and as the defaults in
`src/dual_camera_tracker.py`; change them for your cameras.

## ROS interfaces published by the leader

| Topic | Type | When |
|---|---|---|
| `/robot/pose` | `geometry_msgs/PoseWithCovarianceStamped` | 10 Hz, frame `map` |
| `/robot_pos_data` | `std_msgs/String`, `"x:<val> y:<val>"` | once on arrival, using the goal coordinates (falls back to the live AMCL pose) |

## Known limitations

- Arena goals (`src/locations.yaml`) and the map are specific to the original lab.
- Camera serial numbers are hard-coded (see above).
- The follower-side controller is not included here.

## License

MIT — see `LICENSE`.
