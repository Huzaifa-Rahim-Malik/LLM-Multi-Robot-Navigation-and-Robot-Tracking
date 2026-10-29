#!/usr/bin/env python3
"""
gui.py — Multi-Robot Navigation Assistant  (leader TurtleBot 4 + dual-ZED tracking)

3-column layout:
  Col 1 — Chat (full height, LLM output starts at top)
  Col 2 — Camera 1  |  Arena Tracking  |  Sensor Fusion (Odom XY + LiDAR + IMU)
  Col 3 — Camera 2  |  Locations list  |  Control buttons

Sensor fusion draws from three ROS2 topics:
  /odom  → x, y position only
  /scan  → LaserScan ranges (polar plot)
  /imu   → orientation + angular velocity + linear acceleration
"""

import os
os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
os.environ.setdefault("ROS_DOMAIN_ID", "23")
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if "FASTRTPS_DEFAULT_PROFILES_FILE" not in os.environ:      # fallback if ros/env.sh was not sourced
    _c = os.path.join(_REPO_ROOT, "ros", "fastdds_pc.xml")
    if os.path.exists(_c):
        os.environ["FASTRTPS_DEFAULT_PROFILES_FILE"] = _c

import sys
import threading
import queue
import time
import math
import collections
import tkinter as tk
from tkinter import font as tkfont

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ── print / input bridge ──────────────────────────────────────────────────────
_gui_queue: "queue.Queue | None" = None
_input_event: threading.Event    = threading.Event()
_input_value: list               = [""]


def _gui_print(*args, **kwargs):
    sep  = kwargs.get("sep", " ")
    end  = kwargs.get("end", "\n")
    text = sep.join(str(a) for a in args) + end
    if _gui_queue:
        _gui_queue.put(("print", text))
    else:
        import builtins; builtins._real_print(*args, **kwargs)


def _gui_input(prompt=""):
    if _gui_queue:
        _gui_queue.put(("input_prompt", prompt))
        _input_event.clear(); _input_event.wait()
        return _input_value[0]
    import builtins; return builtins._real_input(prompt)


import builtins
builtins._real_print = builtins.print
builtins._real_input = builtins.input
builtins.print  = _gui_print
builtins.input  = _gui_input
# ─────────────────────────────────────────────────────────────────────────────

import numpy as _np_gui

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import BatteryState, LaserScan, Imu
from nav_msgs.msg import Odometry

from location_manager   import LocationManager
from location_extractor import LocationExtractor
from response_generator import ResponseGenerator
from robot_navigator    import RobotNavigator
from intent_detector    import IntentDetector
from chat_interface     import ChatInterface
from pose_publisher     import PosePublisher

try:
    from voice_input  import VoiceInput
    from voice_output import VoiceOutput
    VOICE_AVAILABLE = True
except ImportError as e:
    VOICE_AVAILABLE = False
    builtins._real_print(f"⚠  Voice unavailable: {e}")

try:
    from dual_camera_tracker import (
        CameraWorker, GlobalTracker,
        draw_camera_view, load_transform,
    )
    TRACKER_AVAILABLE = True
except ImportError as e:
    TRACKER_AVAILABLE = False
    builtins._real_print(f"⚠  dual_camera_tracker unavailable: {e}")


# ═══════════════════════════════════════════════════════════════════
#  NavigationAssistant Node
# ═══════════════════════════════════════════════════════════════════
class NavigationAssistant(Node):
    def __init__(self, enable_voice=True, voice_model="base", sensitivity="medium"):
        super().__init__('navigation_assistant')

        self.location_manager   = LocationManager(self.get_logger())
        self.location_extractor = LocationExtractor(self.get_logger(), self.location_manager)
        self.response_generator = ResponseGenerator(self.get_logger(), self.location_manager)
        self.robot_navigator    = RobotNavigator(self.get_logger(), self.location_manager, self)
        self.intent_detector    = IntentDetector(self.get_logger())
        self.pose_publisher     = PosePublisher(self, self.robot_navigator)

        self.voice_input  = None
        self.voice_output = None

        if enable_voice and VOICE_AVAILABLE:
            try:
                self.voice_input = VoiceInput(
                    self.get_logger(), model_size=voice_model, noise_reduce=True)
                self.voice_input.set_sensitivity(sensitivity)
                self.voice_output = VoiceOutput(
                    self.get_logger(), engine_type="piper", rate=165, volume=0.6)
                self.get_logger().info("✓ Voice modules initialized")
            except Exception as e:
                self.get_logger().error(f"Voice init failed: {e}")
                self.voice_input = self.voice_output = None

        from response_generator import InferenceConfig
        self.inference_config = InferenceConfig()

        # Pass shared InferenceConfig to modules that call Ollama
        self.location_extractor = LocationExtractor(
            self.get_logger(), self.location_manager,
            inference_config=self.inference_config)
        self.response_generator = ResponseGenerator(
            self.get_logger(), self.location_manager,
            inference_config=self.inference_config)

        self.chat_interface = ChatInterface(
            self.get_logger(), self.location_manager, self.location_extractor,
            self.response_generator, self.robot_navigator, self.intent_detector, self,
            voice_input=self.voice_input, voice_output=self.voice_output,
            pose_publisher=self.pose_publisher,
        )

        self.battery_percentage = None
        self.latest_odom        = None
        self.latest_scan        = None
        self.latest_imu         = None

        self.create_subscription(BatteryState, '/battery_state', self._bat_cb,  10)
        self.create_subscription(Odometry,     '/odom',          self._odom_cb, 10)
        self.create_subscription(LaserScan,    '/scan',          self._scan_cb, 10)
        self.create_subscription(Imu,          '/imu',           self._imu_cb,  10)

        self.get_logger().info("Navigation Assistant Ready!")

    def _bat_cb(self,  msg): self.battery_percentage = msg.percentage
    def _odom_cb(self, msg): self.latest_odom        = msg
    def _scan_cb(self, msg): self.latest_scan        = msg
    def _imu_cb(self,  msg): self.latest_imu         = msg

    def run(self): self.chat_interface.chat_loop()


# ═══════════════════════════════════════════════════════════════════
#  Design tokens
# ═══════════════════════════════════════════════════════════════════
# Teal-cyan dark palette matching the TurtleBot Navigation Assistant style
C = dict(
    bg0        = "#080c18",   # deepest background — deep navy-black
    bg1        = "#0d1626",   # panel background
    bg2        = "#192338",   # section header
    bg3        = "#111c30",   # inner canvas / listbox
    bg4        = "#1e2e48",   # input field
    border     = "#2a3f5e",   # panel border / divider
    border_hi  = "#06b6d4",   # active / focused border — teal

    # accents
    blue       = "#06b6d4",   # primary accent — teal/cyan
    blue_dim   = "#0c4a6e",   # dim teal for header backgrounds
    green      = "#10b981",   # success / nav active
    green_dim  = "#064e3b",
    amber      = "#f59e0b",   # warning / system msg
    red        = "#ef4444",   # error / stop
    purple     = "#a855f7",   # IMU accent
    cyan       = "#22d3ee",   # LiDAR accent — lighter teal
    orange     = "#f97316",   # velocity
    rose       = "#fb7185",   # angular vel
    # text
    tx0        = "#f0f6fc",   # body text
    tx1        = "#a0b4cc",   # secondary — better contrast
    tx2        = "#6b82a0",   # dimmed
    tx3        = "#3d4f6a",   # very dim
    # chart series
    ch_x       = "#38bdf8",   # odom x
    ch_y       = "#34d399",   # odom y
    ch_vx      = "#f97316",   # lin vel
    ch_vy      = "#fbbf24",
    ch_wz      = "#f43f5e",
    ch_yaw     = "#a78bfa",
    ch_lid     = "#22d3ee",   # lidar
    ch_imu_ax  = "#fb923c",
    ch_imu_ay  = "#a3e635",
    ch_imu_az  = "#818cf8",
    ch_imu_gz  = "#f43f5e",
)

# Font family names — resolved after Tk root is created (see _init_fonts)
_SANS_FAM = "TkDefaultFont"
_MONO_FAM = "TkFixedFont"

FONT_MONO       = (_MONO_FAM, 10)
FONT_UI         = (_SANS_FAM, 10)
FONT_UI_B       = (_SANS_FAM, 10, "bold")
FONT_UI_SM      = (_SANS_FAM,  9)
FONT_TITLE      = (_SANS_FAM, 14, "bold")
FONT_INPUT      = (_SANS_FAM, 11)
FONT_PLOT_TITLE = (_SANS_FAM,  9, "bold")
FONT_PLOT_AXIS  = (_SANS_FAM,  8)
FONT_PLOT_VAL   = (_SANS_FAM,  8)


def _init_fonts(root):
    """Detect best available fonts after Tk root exists, then update globals."""
    global _SANS_FAM, _MONO_FAM
    global FONT_MONO, FONT_UI, FONT_UI_B, FONT_UI_SM, FONT_TITLE, FONT_INPUT
    global FONT_PLOT_TITLE, FONT_PLOT_AXIS, FONT_PLOT_VAL
    available = set(tkfont.families(root=root))
    def _pick(candidates):
        for f in candidates:
            if f in available:
                return f
        return "TkDefaultFont"
    _SANS_FAM = _pick(["Ubuntu", "Noto Sans", "DejaVu Sans", "Liberation Sans", "Helvetica"])
    _MONO_FAM = _pick(["Ubuntu Mono", "JetBrains Mono", "DejaVu Sans Mono", "Courier New"])
    FONT_MONO       = (_MONO_FAM, 10)
    FONT_UI         = (_SANS_FAM, 10)
    FONT_UI_B       = (_SANS_FAM, 10, "bold")
    FONT_UI_SM      = (_SANS_FAM,  9)
    FONT_TITLE      = (_SANS_FAM, 14, "bold")
    FONT_INPUT      = (_SANS_FAM, 11)
    FONT_PLOT_TITLE = (_SANS_FAM,  9, "bold")
    FONT_PLOT_AXIS  = (_SANS_FAM,  8)
    FONT_PLOT_VAL   = (_SANS_FAM,  8)


def _tag(hex_color: str) -> str:
    """Ensure hex colour is valid tkinter format."""
    return hex_color


# ── Arena drawing helpers (ported from gui_dual_camera_tracker.py) ────────────

def _hex_to_bgr(hex_str):
    h = hex_str.lstrip('#')
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def _draw_rotated_label(img, text, pos, color, bg_color, scale=0.45, thickness=1):
    import cv2 as _cv
    (tw, th), baseline = _cv.getTextSize(text, _cv.FONT_HERSHEY_SIMPLEX, scale, thickness)
    pad = 4
    canvas = _np_gui.full((th + baseline + pad * 2, tw + pad * 2, 3), bg_color, dtype=_np_gui.uint8)
    _cv.putText(canvas, text, (pad, th + pad),
                _cv.FONT_HERSHEY_SIMPLEX, scale, color, thickness, _cv.LINE_AA)
    rotated = _cv.rotate(canvas, _cv.ROTATE_90_COUNTERCLOCKWISE)
    rh, rw = rotated.shape[:2]
    x, y = pos
    if x < 0 or y < 0 or x + rw > img.shape[1] or y + rh > img.shape[0]:
        return
    img[y:y + rh, x:x + rw] = rotated


def _color_for_id_arena(tid):
    rng = _np_gui.random.default_rng(tid * 9871)
    h = float(rng.integers(0, 180))
    import cv2 as _cv
    hsv = _np_gui.uint8([[[h, 200, 230]]])
    bgr = _cv.cvtColor(hsv, _cv.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_arena(tracks, theme, arena_w=8.0, arena_h=4.0,
               cam1_pos=(-0.88, -0.42), cam2_pos=(3.72, -0.55),
               size=(900, 470)):
    """Top-down arena render: floor, grid, camera markers, track trails + badges."""
    import cv2 as _cv
    W_target, H_target = size
    m_top, m_bottom, m_left, m_right = 10, 55, 70, 60
    sx = (W_target - m_left - m_right) / arena_w
    sy = (H_target - m_top - m_bottom) / arena_h
    scale = min(sx, sy)
    W = int(arena_w * scale + m_left + m_right)
    H = int(arena_h * scale + m_top + m_bottom)

    bg       = _hex_to_bgr(theme['bg'])
    surface  = _hex_to_bgr(theme['surface'])
    border   = _hex_to_bgr(theme['border'])
    text_col = _hex_to_bgr(theme['subtext'])
    muted    = _hex_to_bgr(theme['muted'])
    accent   = _hex_to_bgr(theme['accent'])
    warning  = _hex_to_bgr(theme['warning'])

    img = _np_gui.full((H, W, 3), bg, dtype=_np_gui.uint8)

    # Arena floor
    _cv.rectangle(img, (m_left, m_top), (W - m_right, H - m_bottom), surface, -1)
    _cv.rectangle(img, (m_left, m_top), (W - m_right, H - m_bottom), border, 2)

    # Grid lines + labels (Y flipped: y=0 at bottom)
    for x in range(int(arena_w) + 1):
        u = int(m_left + x * scale)
        _cv.line(img, (u, m_top), (u, H - m_bottom), border, 1, _cv.LINE_AA)
        _cv.putText(img, str(x), (u - 4, H - m_bottom + 22),
                    _cv.FONT_HERSHEY_SIMPLEX, 0.45, text_col, 1, _cv.LINE_AA)
    for y in range(int(arena_h) + 1):
        v = int(m_top + (arena_h - y) * scale)
        _cv.line(img, (m_left, v), (W - m_right, v), border, 1, _cv.LINE_AA)
        _cv.putText(img, str(y), (m_left - 22, v + 5),
                    _cv.FONT_HERSHEY_SIMPLEX, 0.45, text_col, 1, _cv.LINE_AA)

    # Axis titles
    _cv.putText(img, "X (m)", (W - m_right + 18, H - m_bottom + 22),
                _cv.FONT_HERSHEY_SIMPLEX, 0.45, muted, 1, _cv.LINE_AA)
    floor_h = H - m_top - m_bottom
    _draw_rotated_label(img, "Y (m)", (8, m_top + floor_h // 2 - 28), muted, bg)

    # Origin marker
    origin_v = int(m_top + arena_h * scale)
    _cv.circle(img, (m_left, origin_v), 6, warning, -1)
    _cv.putText(img, "(0,0)", (m_left + 10, origin_v - 10),
                _cv.FONT_HERSHEY_SIMPLEX, 0.42, warning, 1, _cv.LINE_AA)

    # Camera position markers
    for (cx, cy, lbl) in [(*cam1_pos, "CAM 1"), (*cam2_pos, "CAM 2")]:
        u = int(m_left + cx * scale)
        v = int(m_top + (arena_h - cy) * scale)
        u = max(14, min(W - 14, u))
        v = max(14, min(H - 14, v))
        _cv.drawMarker(img, (u, v), accent,
                       markerType=_cv.MARKER_TRIANGLE_DOWN, markerSize=12, thickness=2)
        _cv.putText(img, lbl, (u + 9, v + 4),
                    _cv.FONT_HERSHEY_SIMPLEX, 0.42, accent, 1, _cv.LINE_AA)

    # Tracks: fading trail + dot + ID badge + coords
    for t in tracks:
        col = _color_for_id_arena(t.id)
        for i in range(1, len(t.history)):
            p1 = (int(m_left + t.history[i - 1][0] * scale),
                  int(m_top  + (arena_h - t.history[i - 1][1]) * scale))
            p2 = (int(m_left + t.history[i][0] * scale),
                  int(m_top  + (arena_h - t.history[i][1]) * scale))
            thick = 1 + int((i / len(t.history)) * 2)
            _cv.line(img, p1, p2, col, thick, _cv.LINE_AA)
        u = int(m_left + t.x * scale)
        v = int(m_top  + (arena_h - t.y) * scale)
        _cv.circle(img, (u, v), 11, col, -1, _cv.LINE_AA)
        _cv.circle(img, (u, v), 11, surface, 2, _cv.LINE_AA)
        label = t.class_name
        (tw, th), _ = _cv.getTextSize(label, _cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        _cv.rectangle(img, (u + 14, v - th - 4), (u + 14 + tw + 8, v + 4), col, -1)
        _cv.putText(img, label, (u + 18, v),
                    _cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, _cv.LINE_AA)

    return img


_ARENA_THEME = {
    'bg':      C["bg3"],
    'surface': C["bg1"],
    'border':  C["border"],
    'text':    C["tx0"],
    'subtext': C["tx1"],
    'muted':   C["tx2"],
    'accent':  C["blue"],
    'success': C["green"],
    'danger':  C["red"],
    'warning': C["amber"],
}


# ═══════════════════════════════════════════════════════════════════
#  Small reusable helpers
# ═══════════════════════════════════════════════════════════════════

def _section_hdr(parent, row: int, col: int, title: str,
                 badge: str = "", badge_var: tk.StringVar = None,
                 zoom_cmd=None) -> tk.Frame:
    """Create a styled section header bar and return it."""
    bar = tk.Frame(parent, bg=C["bg2"], pady=0)
    bar.grid(row=row, column=col, sticky="ew")
    # Left accent strip
    tk.Frame(bar, bg=C["blue"], width=4).pack(side="left", fill="y", padx=(0, 6))
    tk.Label(bar, text=title, bg=C["bg2"], fg=C["blue"],
             font=FONT_UI_B, pady=7).pack(side="left", padx=(4, 0))
    # Zoom button — packed first from the right so it sits at the far-right edge
    if zoom_cmd:
        tk.Button(bar, text="⤢", bg=C["bg2"], fg=C["tx2"],
                  font=FONT_UI_SM, relief="flat", bd=0,
                  cursor="hand2", padx=5, pady=0,
                  activeforeground=C["blue"], activebackground=C["bg3"],
                  command=zoom_cmd).pack(side="right", padx=3)
    if badge_var is not None:
        tk.Label(bar, textvariable=badge_var, bg=C["bg2"], fg=C["tx2"],
                 font=FONT_UI_SM).pack(side="right", padx=8)
    elif badge:
        tk.Label(bar, text=badge, bg=C["bg2"], fg=C["tx2"],
                 font=FONT_UI_SM).pack(side="right", padx=8)
    return bar


def _divider(parent, row: int, col: int = 0, colspan: int = 1):
    tk.Frame(parent, bg=C["border"], height=1).grid(
        row=row, column=col, columnspan=colspan, sticky="ew", padx=2)


def _canvas_panel(parent, row: int) -> tk.Canvas:
    """Dark canvas panel for sensor plots."""
    f = tk.Frame(parent, bg=C["bg3"])
    f.grid(row=row, column=0, sticky="nsew", padx=2, pady=(0, 2))
    f.columnconfigure(0, weight=1); f.rowconfigure(0, weight=1)
    cv = tk.Canvas(f, bg=C["bg3"], bd=0, highlightthickness=0)
    cv.pack(fill="both", expand=True)
    return cv


# ═══════════════════════════════════════════════════════════════════
#  TurtlebotGUI
# ═══════════════════════════════════════════════════════════════════
class TurtlebotGUI:
    def __init__(self, root: tk.Tk, args):
        self.root  = root
        self.args  = args
        self.assistant: NavigationAssistant | None = None
        self._queue           = queue.Queue()
        self._mode            = "text"
        self._nav_active      = False
        self._voice_recording = False
        self._ready_shown     = False

        global _gui_queue
        _gui_queue = self._queue

        self._build_ui()
        self._start_backend()
        self._poll_queue()

    # ══════════════════════════════════════════════════════════════════
    #  BUILD UI
    # ══════════════════════════════════════════════════════════════════
    def _build_ui(self):
        _init_fonts(self.root)
        self.root.title("Autonomous Unified Robotics Analytics Platform")
        self.root.configure(bg=C["bg0"])
        self.root.minsize(1280, 768)
        # Maximise to fill screen on every platform / WM
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{sw}x{sh}+0+0")
        try:
            self.root.attributes('-zoomed', True)   # X11 WMs (i3, GNOME, KDE, …)
        except Exception:
            pass

        # ── Header colour palette (dark — unified with body) ─────────────
        HDR_BG        = "#0d1929"   # dark navy — matches body
        HDR_ACCENT    = "#06b6d4"   # teal brand colour
        HDR_TITLE     = "#e2e8f0"   # bright white-blue — platform title
        HDR_SUB       = "#94a3b8"   # muted slate — subtitle
        HDR_LAB       = "#cbd5e1"   # light slate — lab name
        HDR_UNDER     = "#06b6d4"   # teal accent line under header
        PILL_BG       = "#1c2840"   # dark pill fill
        PILL_BORDER   = "#2d4a6e"   # pill border
        PILL_TX       = "#e2e8f0"   # bright pill text
        PILL_AMBER    = "#f59e0b"   # status amber
        PILL_GREEN    = "#10b981"   # battery green
        PILL_BLUE     = "#38bdf8"   # mode blue
        PILL_DIM      = "#6b82a0"   # spk/mic dimmed

        # ── Local fonts scaled for visibility ───────────────────────────
        FONT_HDR_TITLE = (_SANS_FAM, 14, "bold")
        FONT_HDR_SUB   = (_SANS_FAM, 10)
        FONT_HDR_LAB   = (_SANS_FAM, 12, "bold")
        FONT_HDR_CTR   = (_SANS_FAM, 10)
        FONT_HDR_PILL  = (_SANS_FAM,  9)
        FONT_LOGO      = (_MONO_FAM, 18, "bold")

        # ══ HEADER BAR ══════════════════════════════════════════════════
        hdr = tk.Frame(self.root, bg=HDR_BG, pady=0)
        hdr.pack(fill="x")

        # left accent bar
        tk.Frame(hdr, bg=HDR_ACCENT, width=5).pack(side="left", fill="y")

        lft = tk.Frame(hdr, bg=HDR_BG, pady=10)
        lft.pack(side="left", padx=(12, 0))

        # ── AURA logo badge ──────────────────────────────────────────────
        logo_frame = tk.Frame(lft, bg=HDR_ACCENT, padx=10, pady=6)
        logo_frame.pack(side="left", padx=(0, 16))
        tk.Label(
            logo_frame,
            text="A U R A",
            bg=HDR_ACCENT,
            fg="#FFFFFF",
            font=FONT_LOGO,
        ).pack()

        # ── Platform title & subtitle ────────────────────────────────────
        tlbl = tk.Frame(lft, bg=HDR_BG)
        tlbl.pack(side="left")
        tk.Label(tlbl,
                 text="Autonomous Unified Robotics Analytics Platform",
                 bg=HDR_BG, fg=HDR_TITLE,
                 font=FONT_HDR_TITLE).pack(anchor="w")
        tk.Label(tlbl,
                 text="Intelligent Indoor Swarm Navigation  •  Global Localization ",
                 bg=HDR_BG, fg=HDR_SUB,
                 font=FONT_HDR_SUB).pack(anchor="w")

        # ── Vertical divider ─────────────────────────────────────────────
        tk.Frame(lft, bg=HDR_ACCENT, width=1).pack(
            side="left", fill="y", padx=(18, 16))

        # ── Lab / Centre info ────────────────────────────────────────────
        lab_info = tk.Frame(lft, bg=HDR_BG)
        lab_info.pack(side="left")
        tk.Label(lab_info,
                 text="Autonomous AI & Decision Support Lab",
                 bg=HDR_BG, fg=HDR_LAB,
                 font=FONT_HDR_LAB).pack(anchor="w")
        tk.Label(lab_info,
                 text="Artificial Intelligence Technology Centre",
                 bg=HDR_BG, fg=HDR_SUB,
                 font=FONT_HDR_CTR).pack(anchor="w")

        # ── Right pill indicators ────────────────────────────────────────
        rgt = tk.Frame(hdr, bg=HDR_BG)
        rgt.pack(side="right", padx=14)

        def _pill(parent, var, fg=PILL_TX):
            return tk.Label(parent, textvariable=var,
                            bg=PILL_BG, fg=fg,
                            font=FONT_HDR_PILL, padx=9, pady=4,
                            relief="solid", bd=1,
                            highlightbackground=PILL_BORDER,
                            highlightthickness=1)

        self._status_var  = tk.StringVar(value="⬤ Initializing")
        self._bat_var     = tk.StringVar(value="🔋 --%")
        self._spk_var     = tk.StringVar(value="🔊 —")
        self._mic_var     = tk.StringVar(value="🎤 —")
        self._mode_var    = tk.StringVar(value="TEXT ⌨")

        self._status_lbl  = _pill(rgt, self._status_var, PILL_AMBER)
        self._bat_lbl     = _pill(rgt, self._bat_var,    PILL_GREEN)
        self._spk_lbl     = _pill(rgt, self._spk_var,    PILL_DIM)
        self._mic_lbl     = _pill(rgt, self._mic_var,    PILL_DIM)
        _mode_lbl         = _pill(rgt, self._mode_var,   PILL_BLUE)

        for w in (_mode_lbl, self._mic_lbl, self._spk_lbl,
                  self._bat_lbl, self._status_lbl):
            w.pack(side="right", padx=4)

        # 1px accent line under header
        tk.Frame(self.root, bg=HDR_UNDER, height=1).pack(fill="x")

        # ══ STATUS BAR (bottom) ══════════════════════════════════════════
        sbar = tk.Frame(self.root, bg=C["bg1"], pady=4)
        sbar.pack(fill="x", side="bottom")
        tk.Frame(sbar, bg=C["blue"], width=4).pack(side="left", fill="y")

        self._nav_var       = tk.StringVar(value="")
        self._pos_var       = tk.StringVar(value="AMCL: —")
        self._odom_stat_var = tk.StringVar(value="Odom: —")
        self._scan_stat_var = tk.StringVar(value="LiDAR: —")
        self._imu_stat_var  = tk.StringVar(value="IMU: —")

        tk.Label(sbar, textvariable=self._nav_var,
                 bg=C["bg1"], fg=C["green"],    font=FONT_UI).pack(side="left",  padx=12)
        for var, fg in [
            (self._imu_stat_var,  C["purple"]),
            (self._scan_stat_var, C["cyan"]),
            (self._odom_stat_var, C["ch_x"]),
            (self._pos_var,       C["tx2"]),
        ]:
            tk.Label(sbar, textvariable=var, bg=C["bg1"], fg=fg,
                     font=FONT_UI).pack(side="right", padx=10)

        # ══ MAIN GRID — chat | right pane ═══════════════════════════════
        content = tk.Frame(self.root, bg=C["bg0"])
        content.pack(fill="both", expand=True, padx=4, pady=4)
        content.columnconfigure(0, weight=1, minsize=360)
        content.columnconfigure(1, weight=2, minsize=720)
        content.rowconfigure(0, weight=1)

        # ────────────────────────────────────────────────────────────────
        #  COLUMN 1 — Chat
        # ────────────────────────────────────────────────────────────────
        col1 = tk.Frame(content, bg=C["bg1"],
                        highlightthickness=1, highlightbackground=C["border"])
        col1.grid(row=0, column=0, sticky="nsew", padx=(0, 3))
        col1.columnconfigure(0, weight=1)
        col1.rowconfigure(1, weight=1)   # chat text gets all space

        _section_hdr(col1, 0, 0, "💬  CHAT",
                     badge_var=None)

        chat_wrap = tk.Frame(col1, bg=C["bg0"])
        chat_wrap.grid(row=1, column=0, sticky="nsew")
        chat_wrap.columnconfigure(0, weight=1)
        chat_wrap.rowconfigure(0, weight=1)

        self._chat = tk.Text(
            chat_wrap, bg=C["bg0"], fg=C["tx0"],
            font=FONT_MONO, insertbackground=C["blue"],
            relief="flat", bd=0, wrap="word", state="disabled",
            pady=8, padx=10, spacing1=3, spacing3=3,
        )
        _csb = tk.Scrollbar(chat_wrap, orient="vertical",
                            command=self._chat.yview, width=8,
                            bg=C["bg1"], troughcolor=C["bg3"],
                            activebackground=C["border"])
        self._chat.configure(yscrollcommand=_csb.set)
        _csb.grid(row=0, column=1, sticky="ns")
        self._chat.grid(row=0, column=0, sticky="nsew")

        _BOT_FONT  = (*FONT_MONO[:1], FONT_MONO[1], "bold") if len(FONT_MONO)==2 \
                     else (FONT_MONO[0], FONT_MONO[1], "bold")
        self._chat.tag_configure("bot",    foreground=C["blue"],
                                 font=_BOT_FONT)
        self._chat.tag_configure("you",    foreground=C["green"],
                                 font=_BOT_FONT)
        self._chat.tag_configure("system", foreground=C["amber"])
        self._chat.tag_configure("warn",   foreground=C["red"])
        self._chat.tag_configure("body",   foreground=C["tx0"])
        self._chat.tag_configure("nav",    foreground=C["green"])
        self._chat.tag_configure("dim",    foreground=C["tx3"])

        # Generating indicator
        self._generating_var      = tk.StringVar(value="")
        self._generating_blink_id = None
        tk.Label(col1, textvariable=self._generating_var,
                 bg=C["bg1"], fg=C["blue"], font=FONT_UI_SM,
                 anchor="w").grid(row=2, column=0, sticky="ew", padx=8, pady=(1, 0))

        _divider(col1, 3)

        # Input row
        inp = tk.Frame(col1, bg=C["bg1"], pady=6)
        inp.grid(row=4, column=0, sticky="ew", padx=6, pady=(2, 6))
        inp.columnconfigure(0, weight=1)

        self._entry = tk.Entry(
            inp, bg=C["bg4"], fg=C["tx0"], font=FONT_INPUT,
            insertbackground=C["blue"], relief="flat", bd=0,
            highlightthickness=2, highlightbackground=C["border"],
            highlightcolor=C["blue"])
        self._entry.grid(row=0, column=0, sticky="ew", ipady=7)
        self._entry.bind("<Return>", self._on_send)
        self._entry.bind("<Up>",     self._history_up)
        self._entry.bind("<Down>",   self._history_down)

        self._send_btn = tk.Button(
            inp, text="Send ↵", bg=C["blue"], fg="#ffffff",
            font=FONT_INPUT, relief="flat", bd=0, padx=14,
            activebackground="#22c9e8", activeforeground="#ffffff",
            cursor="hand2", command=self._on_send)
        self._send_btn.bind("<Enter>", lambda e: self._send_btn.configure(bg="#22c9e8"))
        self._send_btn.bind("<Leave>", lambda e: self._send_btn.configure(bg=C["blue"]))
        self._send_btn.grid(row=0, column=1, padx=(6, 0), ipady=7)

        self._rec_btn = tk.Button(
            inp, text="🎤", bg=C["green"], fg="#ffffff",
            font=tkfont.Font(family=_SANS_FAM, size=12),
            relief="flat", bd=0, padx=10,
            activebackground="#2ea043", cursor="hand2",
            state="disabled", command=self._on_toggle_recording)
        self._rec_btn.grid(row=0, column=2, padx=(4, 0), ipady=7)
        self._rec_btn.grid_remove()

        self._rec_buf        = None
        self._rec_stream_obj = None
        self._rec_native_sr  = 48000

        # ────────────────────────────────────────────────────────────────
        #  RIGHT PANE — cameras share a row (equal height) + lower panels
        # ────────────────────────────────────────────────────────────────
        right_pane = tk.Frame(content, bg=C["bg0"])
        right_pane.grid(row=0, column=1, sticky="nsew")
        right_pane.columnconfigure(0, weight=1, uniform="col")
        right_pane.columnconfigure(1, weight=1, uniform="col")
        right_pane.rowconfigure(0, weight=3, minsize=300)   # cameras — same row = guaranteed equal height
        right_pane.rowconfigure(1, weight=5)                # lower panels

        # ── Camera 1 and Camera 2 — in the same parent row ───────────────
        cam1_frame = tk.Frame(right_pane, bg=C["bg1"],
                              highlightthickness=1, highlightbackground=C["border"])
        cam1_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 2), pady=(0, 2))
        cam1_frame.columnconfigure(0, weight=1)
        cam1_frame.rowconfigure(1, weight=1)

        cam2_frame = tk.Frame(right_pane, bg=C["bg1"],
                              highlightthickness=1, highlightbackground=C["border"])
        cam2_frame.grid(row=0, column=1, sticky="nsew", pady=(0, 2))
        cam2_frame.columnconfigure(0, weight=1)
        cam2_frame.rowconfigure(1, weight=1)

        self._cam1_badge = tk.StringVar(value="")
        _section_hdr(cam1_frame, 0, 0, "📷  CAMERA 1  — Global Localization",
                     badge_var=self._cam1_badge,
                     zoom_cmd=lambda: self._open_camera_zoom("cam1", "Camera 1 — Live Feed"))
        self._cam1_panel = self._video_label(cam1_frame, 1)

        self._cam2_badge = tk.StringVar(value="")
        _section_hdr(cam2_frame, 0, 0, "📷  CAMERA 2  — Global Localization",
                     badge_var=self._cam2_badge,
                     zoom_cmd=lambda: self._open_camera_zoom("cam2", "Camera 2 — Live Feed"))
        self._cam2_panel = self._video_label(cam2_frame, 1)

        # ── Lower-left — Arena Tracking + Sensor Fusion ──────────────────
        lower_left = tk.Frame(right_pane, bg=C["bg1"],
                              highlightthickness=1, highlightbackground=C["border"])
        lower_left.grid(row=1, column=0, sticky="nsew", padx=(0, 2))
        lower_left.columnconfigure(0, weight=1)
        lower_left.rowconfigure(1, weight=2, minsize=200)   # tracking
        lower_left.rowconfigure(4, weight=3, minsize=240)   # fusion

        self._track_badge = tk.StringVar(value="")
        _section_hdr(lower_left, 0, 0, "🗺  ARENA TRACKING  — Top-Down",
                     badge_var=self._track_badge,
                     zoom_cmd=lambda: self._open_camera_zoom("track", "Arena Tracking — Top-Down"))
        self._track_panel = self._video_label(lower_left, 1)

        _divider(lower_left, 2)

        self._fusion_badge = tk.StringVar(value="odom • lidar • imu")
        _section_hdr(lower_left, 3, 0, "📡 Leader Robot SENSOR Data",
                     badge_var=self._fusion_badge,
                     zoom_cmd=lambda: self._open_fusion_zoom())

        fusion_wrap = tk.Frame(lower_left, bg=C["bg0"])
        fusion_wrap.grid(row=4, column=0, sticky="nsew", padx=2, pady=(0, 2))
        fusion_wrap.columnconfigure(0, weight=1)
        fusion_wrap.rowconfigure(0, weight=1)   # odom
        fusion_wrap.rowconfigure(1, weight=1)   # lidar
        fusion_wrap.rowconfigure(2, weight=1)   # imu

        def _fixed_canvas(parent, row):
            f = tk.Frame(parent, bg=C["bg3"])
            f.grid(row=row, column=0, sticky="nsew", padx=2, pady=(0, 1))
            f.columnconfigure(0, weight=1)
            f.rowconfigure(0, weight=1)
            cv = tk.Canvas(f, bg=C["bg3"], bd=0, highlightthickness=0)
            cv.grid(row=0, column=0, sticky="nsew")
            return cv

        self._odom_cv  = _fixed_canvas(fusion_wrap, 0)
        self._lidar_cv = _fixed_canvas(fusion_wrap, 1)
        self._imu_cv   = _fixed_canvas(fusion_wrap, 2)

        self._odom_cv .bind("<Configure>", lambda e: self._draw_odom())
        self._lidar_cv.bind("<Configure>", lambda e: self._draw_lidar())
        self._imu_cv  .bind("<Configure>", lambda e: self._draw_imu())

        _BUF = 400
        self._ox  = collections.deque(maxlen=_BUF)
        self._oy  = collections.deque(maxlen=_BUF)
        self._scan_ranges    = []
        self._scan_angle_min = 0.0
        self._scan_angle_inc = 0.0
        self._scan_range_max = 10.0
        self._imu_ax = collections.deque(maxlen=_BUF)
        self._imu_ay = collections.deque(maxlen=_BUF)
        self._imu_az = collections.deque(maxlen=_BUF)
        self._imu_gz = collections.deque(maxlen=_BUF)

        # ── Lower-right — Available Locations + Controls ─────────────────
        lower_right = tk.Frame(right_pane, bg=C["bg1"],
                               highlightthickness=1, highlightbackground=C["border"])
        lower_right.grid(row=1, column=1, sticky="nsew")
        lower_right.columnconfigure(0, weight=1)
        lower_right.rowconfigure(1, weight=1)   # locations list expands
        lower_right.rowconfigure(4, weight=0)   # controls fixed height

        _section_hdr(lower_right, 0, 0, "📍  AVAILABLE LOCATIONS")

        loc_wrap = tk.Frame(lower_right, bg=C["bg3"])
        loc_wrap.grid(row=1, column=0, sticky="nsew", padx=2, pady=(0, 1))
        loc_wrap.columnconfigure(0, weight=1)
        loc_wrap.rowconfigure(0, weight=1)

        self._loc_list = tk.Listbox(
            loc_wrap, bg=C["bg3"], fg=C["tx1"],
            font=FONT_UI,
            selectbackground=C["blue_dim"],
            selectforeground=C["tx0"],
            relief="flat", bd=0, activestyle="none",
            highlightthickness=0,
        )
        _lsb = tk.Scrollbar(loc_wrap, orient="vertical",
                            command=self._loc_list.yview, width=8,
                            bg=C["bg1"], troughcolor=C["bg3"])
        self._loc_list.configure(yscrollcommand=_lsb.set)
        _lsb.grid(row=0, column=1, sticky="ns")
        self._loc_list.grid(row=0, column=0, sticky="nsew")
        self._loc_list.bind("<Double-Button-1>", self._on_location_click)

        _divider(lower_right, 2)

        _section_hdr(lower_right, 3, 0, "🎛  CONTROLS")

        ctrl = tk.Frame(lower_right, bg=C["bg1"], pady=8)
        ctrl.grid(row=4, column=0, sticky="ew", padx=8, pady=(2, 8))
        ctrl.columnconfigure(0, weight=1)
        ctrl.columnconfigure(1, weight=1)

        def _btn(text, r, cl, bg, fg="#ffffff", cmd=None):
            _hover_map = {
                C["green"]: "#13c994",
                C["bg4"]:   "#253655",
                "#3d0c0c":  "#5a1010",
                C["blue"]:  "#22c9e8",
            }
            _hover = _hover_map.get(bg, bg)
            b = tk.Button(ctrl, text=text, bg=bg, fg=fg, font=FONT_UI_B,
                          relief="flat", bd=0, cursor="hand2",
                          activebackground=_hover, activeforeground=fg,
                          pady=7, padx=8, command=cmd)
            b.bind("<Enter>", lambda e: b.configure(bg=_hover))
            b.bind("<Leave>", lambda e: b.configure(bg=bg))
            b.grid(row=r, column=cl, sticky="ew", padx=4, pady=4)
            return b

        self._goto_btn  = _btn("🚀  Go to Selected",  0, 0, C["green"],   cmd=self._on_goto)
        self._pos_btn   = _btn("📍  My Position",     0, 1, C["bg4"],     cmd=self._ask_position)
        self._mode_btn  = _btn("🎤  Switch to Voice", 1, 0, C["bg4"],     cmd=self._toggle_mode)
        self._locs_btn  = _btn("📋  List Locations",  1, 1, C["bg4"],     cmd=self._list_locations)
        self._stop_btn  = _btn("⏹   Stop Speaking",   2, 0, "#3d0c0c",   fg=C["red"],   cmd=self._stop_speaking)
        self._clear_btn = _btn("🗑   Clear Chat",      2, 1, "#3d0c0c",   fg=C["red"],   cmd=self._clear_chat)
        self._stopnav_btn = _btn("🛑  Stop Navigation", 3, 0, "#3d0c0c",  fg=C["amber"], cmd=self._stop_navigation)
        self._stopnav_btn.configure(state="disabled")

        # ── LLM Inference dropdown button — sits in column 1, same row ──
        self._gpu_available  = False
        self._inference_var  = tk.StringVar(value="CPU 🖥")
        self._gpu_status_var = tk.StringVar(value="detecting…")

        self._infer_btn = tk.Button(
            ctrl, textvariable=self._inference_var,
            bg=C["bg4"], fg=C["tx0"], activebackground=C["bg3"],
            activeforeground=C["tx0"], relief="flat", cursor="hand2",
            font=FONT_UI_SM, padx=6, pady=4,
            command=self._show_inference_menu)
        self._infer_btn.grid(row=3, column=1, sticky="ew", padx=4, pady=4)

        # Build the popup menu (options filled after GPU detection)
        self._inference_menu = tk.Menu(
            ctrl, tearoff=0,
            bg=C["bg2"], fg=C["tx0"],
            activebackground=C["green"], activeforeground=C["bg0"],
            font=FONT_UI_SM, bd=0, relief="flat")

        # Queue depth label — row 4, col 0 (below Stop Nav)
        self._navqueue_var = tk.StringVar(value="")
        tk.Label(ctrl, textvariable=self._navqueue_var, bg=C["bg1"],
                 fg=C["amber"], font=FONT_UI_SM).grid(
            row=4, column=0, columnspan=2, sticky="w", padx=6, pady=(0,4))

        # ── LLM Model dropdown — row 5, spans both columns ────────────────
        # Populated asynchronously from `ollama list` — only models that are
        # actually installed show up as choices (see _detect_llm_models_async).
        from response_generator import InferenceConfig as _InfCfg
        self._llm_catalog = _InfCfg.MODEL_CATALOG
        self._llm_var = tk.StringVar(value="LLM: detecting…")

        self._llm_btn = tk.Button(
            ctrl, textvariable=self._llm_var,
            bg=C["bg4"], fg=C["tx0"], activebackground=C["bg3"],
            activeforeground=C["tx0"], relief="flat", cursor="hand2",
            font=FONT_UI_SM, padx=6, pady=4, anchor="w",
            command=self._show_llm_menu)
        self._llm_btn.grid(row=5, column=0, columnspan=2, sticky="ew",
                            padx=4, pady=(0, 4))

        self._llm_menu = tk.Menu(
            ctrl, tearoff=0,
            bg=C["bg2"], fg=C["tx0"],
            activebackground=C["green"], activeforeground=C["bg0"],
            font=FONT_UI_SM, bd=0, relief="flat")
        self._llm_menu.add_command(label="detecting installed models…",
                                    state="disabled")

        # ── State refs ──────────────────────────────────────────────────
        self._tracker_lock  = threading.Lock()
        self._cam1_frame    = None
        self._cam2_frame    = None
        self._track_topdown = None
        self._tracker_ref   = None
        self._cam1_worker   = None
        self._cam2_worker   = None
        self._cam1_photo    = None
        self._cam2_photo    = None
        self._track_photo   = None

        self._history:  list[str] = []
        self._hist_idx: int       = -1
        self._robot_x   = 0.0
        self._robot_y   = 0.0
        self._audio_input_devices  = []
        self._audio_output_devices = []
        self._selected_input_idx   = None
        self._selected_output_idx  = None

    # ── video label helper ─────────────────────────────────────────────
    def _video_label(self, parent, row: int) -> tk.Label:
        wrap = tk.Frame(parent, bg=C["bg0"])
        wrap.grid(row=row, column=0, sticky="nsew", padx=2, pady=0)
        wrap.columnconfigure(0, weight=1)
        wrap.rowconfigure(0, weight=1)
        wrap.grid_propagate(False)   # ← prevent camera image from inflating the row
        lbl = tk.Label(wrap, bg=C["bg0"], fg=C["tx3"],
                       text="No feed — camera not connected",
                       font=FONT_UI_SM)
        lbl.grid(row=0, column=0, sticky="nsew")
        return lbl

    # ══════════════════════════════════════════════════════════════════
    #  Backend startup
    # ══════════════════════════════════════════════════════════════════
    def _start_backend(self):
        ev = "--text-only" not in self.args
        vm = "large-v3"; sv = "medium"
        if "--voice-model"  in self.args:
            try: i=self.args.index("--voice-model");  vm=self.args[i+1]
            except: pass
        if "--sensitivity"  in self.args:
            try: i=self.args.index("--sensitivity"); sv=self.args[i+1]
            except: pass
        self._voice_enabled = ev and VOICE_AVAILABLE
        threading.Thread(target=self._backend_thread,
                         args=(ev, vm, sv), daemon=True).start()

    def _backend_thread(self, ev, vm, sv):
        try:
            try: import os as _os; _os.nice(5)
            except: pass
            rclpy.init()
            self.assistant = NavigationAssistant(
                enable_voice=ev, voice_model=vm, sensitivity=sv)
            self.assistant.chat_interface.gui_mode = True
            self.assistant.chat_interface.gui_mode_changed_callback = \
                lambda m: self._queue.put(("mode_changed", m))
            self._queue.put(("ready", None))
            self.assistant.run()
        except Exception as e:
            self._queue.put(("error", str(e)))
        finally:
            if self.assistant:
                try:
                    self.assistant.robot_navigator.shutdown()
                    self.assistant.destroy_node()
                    rclpy.shutdown()
                except: pass

    # ══════════════════════════════════════════════════════════════════
    #  Queue polling
    # ══════════════════════════════════════════════════════════════════
    def _poll_queue(self):
        try:
            for _ in range(5):   # max 5 items per tick — prevents burst stalls
                t, d = self._queue.get_nowait()
                if   t == "ready":        self._on_ready()
                elif t == "print":        self._route_print(d)
                elif t == "error":        self._append("warn", f"ERROR: {d}\n")
                elif t == "mode_changed": self._sync_mode_widgets(d)
                elif t == "_badge":
                    try:
                        n, n1, n2 = d
                        self._cam1_badge.set(f"{n1} det")
                        self._cam2_badge.set(f"{n2} det")
                        self._track_badge.set(f"{n} tracks")
                    except Exception:
                        pass
        except queue.Empty: pass
        except Exception: pass
        finally: self.root.after(50, self._poll_queue)   # 20 Hz polling

    # ══════════════════════════════════════════════════════════════════
    #  Chat routing
    # ══════════════════════════════════════════════════════════════════
    def _route_print(self, text: str):
        try:
            t = text.rstrip("\n")
            if not t:
                return
            if t == "__GUI_GENERATING__":
                self._set_generating(True)
                return

            # ── Filter noisy / internal log lines ─────────────────────
            _DP = ("[DEBUG]","[STOP]","[WAKE]","[REC]","[QT]",
                   "⚠️  Voice","🔄 Transcrib","faster-whisper","INFO","WARN","ERROR")
            _DS = ("noise floor","Calibrat","energy=","buf=","floor=","Cycle ",
                   "ring buffer","transcribed =","Partial transcription","end word",
                   "End word","Speech detected","Silence detected","Recording...",
                   "speech_thresh","noise=","inline command","Stripped inline",
                   "Listener thread","Listener active","stop word","STOP WORD","interrupted",
                   # Camera / tracker / OpenCV noise — keep out of chatbox
                   "Camera","camera","YOLO","yolo","OpenCV","cv2","tracker",
                   "Tracker","CameraWorker","GlobalTracker","frame","Frame",
                   "detection","Detection","bbox","BBox","ultralytics","Ultralytics",
                   "load_transform","cam1","cam2","✓ Global","⚠  Tracker",
                   "⚠  Camera","libEGL","libGL","MESA","DRI","drm","wayland",
                   "Gst","gst","GLib","pulse","ALSA","jack","pipewire")
            if any(t.startswith(p) for p in _DP):
                return
            if any(s in t for s in _DS):
                return

            # ── Route to chat display ──────────────────────────────────
            if t.startswith("__NAV_STARTED__"):
                dest = t.replace("__NAV_STARTED__","").replace("__","").strip()
                self._set_status(f"🚀 Navigating → {dest}", C["green"])
                self._nav_active = True
                try: self._stopnav_btn.configure(state="normal")
                except Exception: pass
                q = getattr(
                    getattr(getattr(self.assistant,"chat_interface",None),"_nav_queue",None),
                    "__len__", lambda: 0)()
                self._navqueue_var.set(f"+{q} queued" if q else "")
                return
            elif t.startswith("__NAV_ARRIVED__"):
                dest = t.replace("__NAV_ARRIVED__","").replace("__","").strip()
                # Check if more stops remain
                ci = getattr(self.assistant, "chat_interface", None)
                remaining = len(ci._nav_queue) if ci else 0
                if remaining:
                    self._set_status(f"✓ Reached {dest} — {remaining} stop(s) left", C["blue"])
                    self._navqueue_var.set(f"+{remaining} queued")
                else:
                    self._set_status(f"✓ Arrived at {dest}", C["blue"])
                    self._nav_active = False
                    self._navqueue_var.set("")
                    try: self._stopnav_btn.configure(state="disabled")
                    except Exception: pass
                return
            elif t.startswith("TURTLEBOT:"):
                self._set_generating(False)
                body = t[len("TURTLEBOT: "):]
                self._append("bot", "TURTLEBOT: ")
                self._append("body", body+"\n")
                if any(k in body.lower() for k in ("heading to","navigating to","on my way")):
                    self._nav_active = True
            elif t.startswith("YOU (voice):") or t.startswith("YOU:"):
                lbl = t.split(":")[0]+": "
                self._append("you", lbl)
                self._append("body", t[len(lbl):]+"\n")
            elif "⏹" in t:
                self._append("warn", t+"\n")
            elif t.startswith("📍 Navigation:") or "→" in t:
                self._append("nav", t+"\n"); self._nav_var.set(t); self._nav_active = True
            elif t.startswith("✓") or "arrived" in t.lower() or "successfully" in t.lower():
                self._append("system", t+"\n"); self._nav_active = False; self._nav_var.set("✓ Arrived")
            elif any(t.startswith(p) for p in ["⚠","WARNING","Error","Failed"]):
                self._append("warn", t+"\n")
            elif t.startswith("💬 Speaking"):
                self._set_status("🔊 Speaking…", C["green"])
            elif t.startswith("🎤") and "Recording" in t:
                self._set_status("🎤 Recording…", C["green"])
            elif t.strip().startswith("═") or t.strip().startswith("─"):
                self._append("dim", t+"\n")
            elif t.strip() == "" or all(c in "═─ \t" for c in t):
                pass
            else:
                self._append("body", t+"\n")
        except Exception:
            pass

    def _append(self, tag, text):
        self._chat.configure(state="normal")
        self._chat.insert("end", text, tag)
        self._chat.configure(state="disabled")
        self._chat.see("end")

    # ══════════════════════════════════════════════════════════════════
    #  Ready callback
    # ══════════════════════════════════════════════════════════════════
    def _on_ready(self):
        if not self.assistant or self._ready_shown: return
        self._ready_shown = True
        self._set_status("⬤ Ready", C["green"])

        # Populate locations
        self._loc_list.delete(0, "end")
        for i, nm in enumerate(self.assistant.location_manager.get_all_locations(), 1):
            lc = self.assistant.location_manager.get_location(nm)
            self._loc_list.insert("end",
                f"  {i:2d}.  {nm:<22}  ({lc['x']:.1f}, {lc['y']:.1f})")
        # Alternating row tint
        for i in range(0, self._loc_list.size(), 2):
            self._loc_list.itemconfigure(i, bg="#1a2030")

        # Welcome box
        self._chat.configure(state="normal")
        self._chat.tag_configure("welcome", foreground=C["amber"],
            font=FONT_MONO)
        self._chat.tag_configure("welcome_dim", foreground=C["tx3"],
            font=FONT_MONO)
        lines = [
            ("welcome",
             "╔════════════════════════════════════════════════════╗\n"),
            ("welcome",
             "║   🤖  TURTLEBOT NAVIGATION ASSISTANT  🤖          ║\n"),
            ("welcome_dim",
             "╠════════════════════════════════════════════════════╣\n"),
            ("welcome_dim",
             '║  Navigate  → "go to [location]"                   ║\n'),
            ("welcome_dim",
             '║  Chat      → ask me anything!                     ║\n'),
            ("welcome_dim",
             '║  Position  → "where am i"                         ║\n'),
            ("welcome_dim",
             '║  Locations → "show locations"                     ║\n'),
            ("welcome_dim",
             '║  Exit      → "quit" / "exit" / Ctrl+C             ║\n'),
            ("welcome",
             "╚═══════════════════════════════════════════════════╝\n"),
            ("body", "\n"),
        ]
        for tag, line in lines:
            self._chat.insert("end", line, tag)
        self._chat.configure(state="disabled")

        # Mic / mode
        if self.assistant.voice_input:
            self._mic_var.set("🎤 ✓"); self._mic_lbl.configure(fg=C["green"])
            self._mode_var.set("TEXT ⌨  •  Voice ✓")
        else:
            self._mic_var.set("🎤 ✗"); self._mic_lbl.configure(fg=C["red"])
            self._mode_var.set("TEXT ⌨  •  No voice hw")
            self._mode_btn.configure(state="disabled", fg=C["tx3"])

        # Start loops
        self.root.after(2000, self._refresh_position_loop)
        self.root.after(1000, self._refresh_battery_loop)
        self.root.after(600,  self._check_bt_status)
        self.root.after(60,   self._camera_update_loop)   # ~15 fps
        self.root.after(100,  self._sensor_refresh_loop)  # 10 Hz sensor fusion

        if TRACKER_AVAILABLE:
            threading.Thread(target=self._start_global_tracker, daemon=True).start()

        # Poll arrival queue every 300 ms so announcements fire even when
        # the chat loop is blocked waiting for user input
        self.root.after(300, self._poll_arrivals)
        # Detect GPU in background so UI stays responsive
        import threading as _th
        _th.Thread(target=self._detect_gpu_async, daemon=True).start()
        # Detect which Ollama models are actually installed (dropdown only
        # ever offers models that are really pulled on this machine)
        _th.Thread(target=self._detect_llm_models_async, daemon=True).start()

    # ══════════════════════════════════════════════════════════════════
    #  Global Tracker
    # ══════════════════════════════════════════════════════════════════
    def _start_global_tracker(self):
        cal1 = os.path.join(_REPO_ROOT, "config", "cam1_points.json")
        cal2 = os.path.join(_REPO_ROOT, "config", "cam2_points.json")
        wts  = os.path.join(_REPO_ROOT, "models", "yolov8_zed_2class.pt")
        try:
            p1 = load_transform(cal1); p2 = load_transform(cal2)
            from ultralytics import YOLO
            model = YOLO(wts); lock = threading.Lock()
            cam1 = CameraWorker(0, 35090194, p1, model, lock, "Camera 1")
            cam2 = CameraWorker(1, 31496664, p2, model, lock, "Camera 2")
            cam1.open(); cam2.open(); cam1.start(); cam2.start()
            tracker = GlobalTracker(gating_dist=0.6, merge_dist=0.4, max_missed=30)
            with self._tracker_lock:
                self._cam1_worker = cam1; self._cam2_worker = cam2
                self._tracker_ref = tracker
            builtins._real_print("✓ Global tracker running")
            while True:
                l1 = cam1.get_latest(); l2 = cam2.get_latest()
                if l1 is None or l2 is None: time.sleep(0.05); continue
                f1,d1 = l1[0].copy(), l1[1]
                f2,d2 = l2[0].copy(), l2[1]
                tracks = tracker.update([d1, d2])
                n = len(tracks)
                draw_camera_view(f1, d1, n, "Camera 1")
                draw_camera_view(f2, d2, n, "Camera 2")
                top = draw_arena(tracks, _ARENA_THEME)
                with self._tracker_lock:
                    self._cam1_frame=f1; self._cam2_frame=f2; self._track_topdown=top
                # update badges via queue (safe for Tk)
                self._queue.put(("_badge", (n, len(d1), len(d2))))
                time.sleep(0.04)
        except Exception as e:
            builtins._real_print(f"⚠  Tracker error: {e}")
            self._queue.put(("print", f"⚠  Camera tracker unavailable: {e}\n"))

    # ══════════════════════════════════════════════════════════════════
    #  Camera feed
    # ══════════════════════════════════════════════════════════════════
    def _bgr_to_photo(self, bgr, mw, mh):
        if not PIL_AVAILABLE or bgr is None: return None
        try:
            import cv2 as _cv
            h, w = bgr.shape[:2]
            if w < 1 or h < 1: return None
            # Scale to FILL the panel (allow upscale)
            s = min(mw / w, mh / h)
            nw, nh = max(1, int(w * s)), max(1, int(h * s))
            resized = _cv.resize(bgr, (nw, nh), interpolation=_cv.INTER_LINEAR)
            return ImageTk.PhotoImage(
                Image.fromarray(_cv.cvtColor(resized, _cv.COLOR_BGR2RGB)))
        except Exception:
            return None

    def _camera_update_loop(self):
        """
        Camera display loop — image conversion runs on a background thread so
        
        the Tk main loop is never blocked by cv2 resize / PIL conversion.
        """
        try:
            with self._tracker_lock:
                f1 = self._cam1_frame
                f2 = self._cam2_frame
                tp = self._track_topdown

            def _convert_and_apply():
                results = {}
                for key, frm, lbl in [
                    ("cam1", f1, self._cam1_panel),
                    ("cam2", f2, self._cam2_panel),
                    ("track", tp, self._track_panel),
                ]:
                    if frm is None:
                        continue
                    try:
                        master = lbl.master
                        w = master.winfo_width()
                        h = master.winfo_height()
                        if w < 10 or h < 10:
                            continue
                        ph = self._bgr_to_photo(frm, w, h)
                        if ph:
                            results[key] = (ph, lbl)
                    except Exception:
                        pass

                # Apply results back on main thread
                def _apply():
                    for key, (ph, lbl) in results.items():
                        attr = f"_{key}_photo"
                        setattr(self, attr, ph)
                        lbl.configure(image=ph, text="")
                self.root.after(0, _apply)

            threading.Thread(target=_convert_and_apply, daemon=True).start()
        except Exception:
            pass
        self.root.after(100, self._camera_update_loop)   # ~10 fps — less CPU pressure

    # ══════════════════════════════════════════════════════════════════
    #  Zoom popups
    # ══════════════════════════════════════════════════════════════════
    def _open_camera_zoom(self, cam_key: str, title: str):
        """Open a resizable popup showing an enlarged live camera feed."""
        popup = tk.Toplevel(self.root)
        popup.title(title)
        popup.configure(bg=C["bg0"])
        popup.geometry("960x540")
        popup.resizable(True, True)

        hdr = tk.Frame(popup, bg=C["bg1"])
        hdr.pack(fill="x")
        tk.Frame(hdr, bg=C["blue"], width=3).pack(side="left", fill="y")
        tk.Label(hdr, text=title, bg=C["bg1"], fg=C["blue"],
                 font=FONT_UI_B, pady=5).pack(side="left", padx=(8, 0))
        tk.Button(hdr, text="✕  Close", bg=C["bg4"], fg=C["tx1"],
                  font=FONT_UI_SM, relief="flat", bd=0, cursor="hand2",
                  padx=10, pady=3, command=popup.destroy
                  ).pack(side="right", padx=8, pady=4)

        lbl = tk.Label(popup, bg=C["bg0"], fg=C["tx3"],
                       text="No feed — camera not connected", font=FONT_UI)
        lbl.pack(fill="both", expand=True)

        _photo_ref = [None]
        _after_id  = [None]

        def _update():
            if not popup.winfo_exists():
                return
            with self._tracker_lock:
                if cam_key == "cam1":
                    frm = self._cam1_frame
                elif cam_key == "cam2":
                    frm = self._cam2_frame
                else:
                    frm = self._track_topdown
            if frm is not None:
                w = lbl.winfo_width()
                h = lbl.winfo_height()
                if w > 10 and h > 10:
                    ph = self._bgr_to_photo(frm, w, h)
                    if ph:
                        _photo_ref[0] = ph
                        lbl.configure(image=ph, text="")
            _after_id[0] = popup.after(60, _update)

        def _on_close():
            if _after_id[0]:
                popup.after_cancel(_after_id[0])
            popup.destroy()

        popup.protocol("WM_DELETE_WINDOW", _on_close)
        popup.after(100, _update)

    def _open_fusion_zoom(self):
        """Open a resizable popup showing all three sensor fusion plots enlarged."""
        popup = tk.Toplevel(self.root)
        popup.title("Sensor Fusion — Zoomed View")
        popup.configure(bg=C["bg0"])
        popup.geometry("1100x780")
        popup.resizable(True, True)

        hdr = tk.Frame(popup, bg=C["bg1"])
        hdr.pack(fill="x")
        tk.Frame(hdr, bg=C["blue"], width=3).pack(side="left", fill="y")
        tk.Label(hdr, text="📡  SENSOR FUSION — Zoomed View",
                 bg=C["bg1"], fg=C["blue"], font=FONT_UI_B, pady=5
                 ).pack(side="left", padx=(8, 0))
        tk.Button(hdr, text="✕  Close", bg=C["bg4"], fg=C["tx1"],
                  font=FONT_UI_SM, relief="flat", bd=0, cursor="hand2",
                  padx=10, pady=3, command=popup.destroy
                  ).pack(side="right", padx=8, pady=4)

        content = tk.Frame(popup, bg=C["bg0"])
        content.pack(fill="both", expand=True, padx=4, pady=4)
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)
        content.rowconfigure(1, weight=1)
        content.rowconfigure(2, weight=1)

        def _make_cv(row):
            f = tk.Frame(content, bg=C["bg3"])
            f.grid(row=row, column=0, sticky="nsew", pady=(0, 3))
            f.columnconfigure(0, weight=1)
            f.rowconfigure(0, weight=1)
            cv = tk.Canvas(f, bg=C["bg3"], bd=0, highlightthickness=0)
            cv.pack(fill="both", expand=True)
            return cv

        z_odom  = _make_cv(0)
        z_lidar = _make_cv(1)
        z_imu   = _make_cv(2)

        _after_id = [None]

        def _update():
            if not popup.winfo_exists():
                return
            self._draw_odom(z_odom)
            self._draw_lidar(z_lidar)
            self._draw_imu(z_imu)
            _after_id[0] = popup.after(200, _update)

        def _on_close():
            if _after_id[0]:
                popup.after_cancel(_after_id[0])
            popup.destroy()

        popup.protocol("WM_DELETE_WINDOW", _on_close)
        popup.after(100, _update)

    # ══════════════════════════════════════════════════════════════════
    #  Sensor fusion refresh loop  (200 ms — 5 Hz keeps GUI responsive)
    # ══════════════════════════════════════════════════════════════════
    def _sensor_refresh_loop(self):
        try:
            if hasattr(self, '_odom_cv'):
                self._refresh_odom()
        except Exception:
            pass
        try:
            if hasattr(self, '_lidar_cv'):
                self._refresh_lidar()
        except Exception:
            pass
        try:
            if hasattr(self, '_imu_cv'):
                self._refresh_imu()
        except Exception:
            pass
        self.root.after(200, self._sensor_refresh_loop)   # 5 Hz — much lighter on main thread

    # ──────────────────────────────────────────────────────────────────
    #  ODOM  (x, y position only)
    # ──────────────────────────────────────────────────────────────────
    def _refresh_odom(self):
        if not self.assistant or not self.assistant.latest_odom: return
        if not hasattr(self, '_ox'): return
        msg = self.assistant.latest_odom
        try:
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            self._ox.append(x); self._oy.append(y)
            self._odom_stat_var.set(
                f"Odom  x={x:+.3f}  y={y:+.3f}")
            self._draw_odom()
        except: pass

    def _draw_odom(self, cv=None):
        """XY trajectory plot (path trace) + live position dot."""
        if cv is None:
            cv = self._odom_cv
        cv.delete("all")
        W = cv.winfo_width()  or 380
        H = cv.winfo_height() or 120
        if W < 30 or H < 30: return

        PL, PR, PT, PB = 46, 10, 18, 22
        pw = W - PL - PR
        ph = H - PT - PB

        # title strip
        cv.create_rectangle(0, 0, W, 20, fill=C["bg2"], outline="")
        cv.create_text(6, 10, text="▸ ODOM  — X/Y Position (m)",
                       fill=C["ch_x"], font=FONT_PLOT_TITLE, anchor="w")

        ox = list(self._ox); oy = list(self._oy)
        if len(ox) < 2:
            cv.create_text(W//2, H//2, text="waiting for /odom…",
                           fill=C["tx3"], font=FONT_PLOT_VAL); return

        xmin,xmax = min(ox),max(ox)
        ymin,ymax = min(oy),max(oy)
        xr = max(xmax-xmin, 0.1); yr = max(ymax-ymin, 0.1)
        xpad = xr*0.15; ypad = yr*0.15
        x0,x1 = xmin-xpad, xmax+xpad
        y0,y1 = ymin-ypad, ymax+ypad
        xspan = x1-x0; yspan = y1-y0

        def tp(x,y):
            px = PL + int((x-x0)/xspan * pw)
            py = PT + int((1-(y-y0)/yspan) * ph)
            return px, py

        # grid lines
        for gv in [x0+(x1-x0)*i/4 for i in range(5)]:
            gx,_ = tp(gv, y0)
            cv.create_line(gx, PT, gx, PT+ph, fill=C["bg3"], width=1)
            cv.create_text(gx, PT+ph+4, text=f"{gv:.2f}",
                           fill=C["tx3"], font=FONT_PLOT_AXIS, anchor="n")
        for gv in [y0+(y1-y0)*i/4 for i in range(5)]:
            _,gy = tp(x0, gv)
            cv.create_line(PL, gy, PL+pw, gy, fill=C["bg3"], width=1)
            cv.create_text(PL-3, gy, text=f"{gv:.2f}",
                           fill=C["tx3"], font=FONT_PLOT_AXIS, anchor="e")

        # axis
        cv.create_line(PL, PT, PL, PT+ph, fill=C["border"], width=1)
        cv.create_line(PL, PT+ph, PL+pw, PT+ph, fill=C["border"], width=1)
        cv.create_text(W-4, PT+ph+2, text="x →", fill=C["ch_x"],
                       font=FONT_PLOT_AXIS, anchor="ne")
        cv.create_text(PL+2, PT+2, text="y↑", fill=C["ch_y"],
                       font=FONT_PLOT_AXIS, anchor="nw")

        # trajectory — decimate to max 150 segments so Tk stays fast
        n = len(ox)
        step = max(1, n // 150)
        disp_x = ox[::step]; disp_y = oy[::step]
        nd = len(disp_x)
        for i in range(1, nd):
            alpha = 0.3 + 0.7*(i/nd)
            r = int(0x38 * alpha); g_ch = int(0xbd * alpha); b = int(0xf8 * alpha)
            col = f"#{r:02x}{g_ch:02x}{b:02x}"
            x_a,y_a = tp(disp_x[i-1], disp_y[i-1])
            x_b,y_b = tp(disp_x[i],   disp_y[i])
            cv.create_line(x_a,y_a, x_b,y_b, fill=col, width=1)

        # current position dot
        cx,cy = tp(ox[-1], oy[-1])
        cv.create_oval(cx-5,cy-5,cx+5,cy+5, fill=C["ch_x"], outline=C["bg0"], width=2)
        # origin
        ox0,oy0 = tp(0,0)
        if PL<=ox0<=PL+pw and PT<=oy0<=PT+ph:
            cv.create_oval(ox0-3,oy0-3,ox0+3,oy0+3,
                           fill=C["amber"], outline="", tags="origin")

        # live readout
        cv.create_text(W-4, 10, text=f"x={ox[-1]:+.3f}  y={oy[-1]:+.3f}",
                       fill=C["tx1"], font=FONT_PLOT_VAL, anchor="ne")

    # ──────────────────────────────────────────────────────────────────
    #  LIDAR  (polar scan plot)
    # ──────────────────────────────────────────────────────────────────
    def _refresh_lidar(self):
        if not self.assistant or not self.assistant.latest_scan: return
        if not hasattr(self, '_scan_ranges'): return
        msg = self.assistant.latest_scan
        try:
            self._scan_ranges    = list(msg.ranges)
            self._scan_angle_min = msg.angle_min
            self._scan_angle_inc = msg.angle_increment
            self._scan_range_max = msg.range_max if msg.range_max > 0 else 10.0
            n = len(self._scan_ranges)
            valid = sum(1 for r in self._scan_ranges
                        if math.isfinite(r) and 0 < r < self._scan_range_max)
            self._scan_stat_var.set(f"LiDAR  {n} pts  {valid} valid")
            self._draw_lidar()
        except: pass

    def _draw_lidar(self, cv=None):
        """Polar plot of LiDAR scan."""
        if cv is None:
            cv = self._lidar_cv
        cv.delete("all")
        W = cv.winfo_width()  or 380
        H = cv.winfo_height() or 120
        if W < 30 or H < 30: return

        # title strip
        cv.create_rectangle(0, 0, W, 20, fill=C["bg2"], outline="")
        cv.create_text(6, 10, text="▸ LIDAR  — Polar Scan  (/scan)",
                       fill=C["cyan"], font=FONT_PLOT_TITLE, anchor="w")

        ranges = self._scan_ranges
        if not ranges:
            cv.create_text(W//2, H//2, text="waiting for /scan…",
                           fill=C["tx3"], font=FONT_PLOT_VAL); return

        # polar canvas setup
        margin = 20
        cx_c = W // 2
        cy_c = (H + 18) // 2
        radius = min((W - 2*margin)//2, (H - margin - 18)//2)

        rng_max = self._scan_range_max
        scale   = radius / max(rng_max, 0.1)

        # range rings
        for frac in (0.25, 0.5, 0.75, 1.0):
            r = int(frac * radius)
            cv.create_oval(cx_c-r, cy_c-r, cx_c+r, cy_c+r,
                           outline=C["bg3"], width=1)
            cv.create_text(cx_c+r+2, cy_c, text=f"{rng_max*frac:.1f}m",
                           fill=C["tx3"], font=FONT_PLOT_AXIS, anchor="w")

        # crosshair
        cv.create_line(cx_c-radius, cy_c, cx_c+radius, cy_c,
                       fill=C["bg3"], width=1)
        cv.create_line(cx_c, cy_c-radius, cx_c, cy_c+radius,
                       fill=C["bg3"], width=1)

        # plot scan points — decimate to max 180 pts so Tk doesn't create 1000 ovals
        n = len(ranges)
        step = max(1, n // 180)
        pts = []
        for i in range(0, n, step):
            r = ranges[i]
            if not (math.isfinite(r) and 0 < r < rng_max): continue
            angle = self._scan_angle_min + i * self._scan_angle_inc
            px = cx_c + r * scale * math.cos(angle)
            py = cy_c - r * scale * math.sin(angle)
            pts.append((px, py))

        # draw dots
        for px, py in pts:
            cv.create_oval(px-1.5, py-1.5, px+1.5, py+1.5,
                           fill=C["cyan"], outline="")

        # robot dot at center
        cv.create_oval(cx_c-4, cy_c-4, cx_c+4, cy_c+4,
                       fill=C["amber"], outline=C["bg0"], width=2)
        cv.create_text(6, 20, text=f"{n} beams",
                       fill=C["tx2"], font=FONT_PLOT_VAL, anchor="nw")

    # ──────────────────────────────────────────────────────────────────
    #  IMU  (accel + gyro time-series)
    # ──────────────────────────────────────────────────────────────────
    def _refresh_imu(self):
        if not self.assistant or not self.assistant.latest_imu: return
        if not hasattr(self, '_imu_ax'): return
        msg = self.assistant.latest_imu
        try:
            ax = msg.linear_acceleration.x
            ay = msg.linear_acceleration.y
            az = msg.linear_acceleration.z
            gz = msg.angular_velocity.z
            self._imu_ax.append(ax); self._imu_ay.append(ay)
            self._imu_az.append(az); self._imu_gz.append(gz)
            self._imu_stat_var.set(
                f"IMU  ax={ax:+.2f}  ay={ay:+.2f}  gz={gz:+.3f}")
            self._draw_imu()
        except: pass

    def _draw_imu(self, cv=None):
        """Two-row IMU graph: top=linear accel (ax,ay,az), bottom=angular vel gz."""
        if cv is None:
            cv = self._imu_cv
        cv.delete("all")
        W = cv.winfo_width()  or 380
        H = cv.winfo_height() or 120
        if W < 30 or H < 30: return

        # title strip
        cv.create_rectangle(0, 0, W, 20, fill=C["bg2"], outline="")
        cv.create_text(6, 10, text="▸ IMU  — Accel & Gyro  (/imu)",
                       fill=C["purple"], font=FONT_PLOT_TITLE, anchor="w")

        PL,PR,PT,PB = 46, 8, 20, 4
        GAP   = 4
        row_h = max(1, (H - PT - PB - GAP) // 2)
        pw    = W - PL - PR

        # series_dict: { key: (data_list, color_str) }
        def _draw_strip(row_top, series_dict, title):
            yb = row_top + row_h
            cv.create_rectangle(PL, row_top, PL+pw, yb,
                                fill=C["bg3"], outline=C["border"])
            cv.create_text(PL+3, row_top+2, text=title,
                           fill=C["tx3"], font=FONT_PLOT_AXIS, anchor="nw")
            # Correctly extract only the numeric data lists (index 0 of each tuple)
            all_data_lists = [item[0] for item in series_dict.values()]
            all_v = [v for lst in all_data_lists for v in lst
                     if isinstance(v, (int, float))]
            if len(all_v) < 2:
                cv.create_text(PL+pw//2, (row_top+yb)//2,
                               text="waiting for /imu…",
                               fill=C["tx3"], font=FONT_PLOT_VAL); return
            vmin = min(all_v); vmax = max(all_v)
            span = max(abs(vmax-vmin), 0.01)
            pad  = span*0.12; v0=vmin-pad; v1=vmax+pad; vs=max(v1-v0, 1e-9)
            inner = max(1, row_h-14)

            def _px(i, v, n):
                return (PL + int(i/max(n-1,1)*pw),
                        yb - 4 - int((v-v0)/vs*inner))

            # zero line
            if v0 < 0 < v1:
                zy = yb - 4 - int((0-v0)/vs*inner)
                cv.create_line(PL, zy, PL+pw, zy, fill=C["border"], dash=(3,3))

            # Y ticks
            for tv in [v0+pad, (v0+v1)/2, v1-pad]:
                py = yb - 4 - int((tv-v0)/vs*inner)
                cv.create_line(PL-3, py, PL, py, fill=C["border"])
                cv.create_text(PL-4, py, text=f"{tv:.2f}",
                               fill=C["tx3"], font=FONT_PLOT_AXIS, anchor="e")

            lx = PL+pw-4
            for key, (data, ser_col) in series_dict.items():
                n   = len(data)
                pts = [_px(i, data[i], n) for i in range(n)]
                for i in range(len(pts)-1):
                    cv.create_line(*pts[i], *pts[i+1], fill=ser_col, width=1)
                if pts:
                    px,py = pts[-1]
                    cv.create_oval(px-2,py-2,px+2,py+2, fill=ser_col, outline="")
                val = f"{key}={data[-1]:+.2f}" if data else key
                cv.create_text(lx, yb-3, text=val, fill=ser_col,
                               font=FONT_PLOT_TITLE, anchor="se")
                lx -= len(val)*5+4

        _draw_strip(PT,
            {"ax": (list(self._imu_ax), C["ch_imu_ax"]),
             "ay": (list(self._imu_ay), C["ch_imu_ay"]),
             "az": (list(self._imu_az), C["ch_imu_az"])},
            "Linear Accel (m/s²)")

        _draw_strip(PT + row_h + GAP,
            {"gz": (list(self._imu_gz), C["purple"])},
            "Angular Vel gz (rad/s)")

    # ══════════════════════════════════════════════════════════════════
    #  Position / battery / BT refresh
    # ══════════════════════════════════════════════════════════════════
    def _refresh_position(self):
        """Run position lookup on background thread — never blocks Tk main loop."""
        def _do():
            if not self.assistant: return
            try:
                pos = self.assistant.robot_navigator.get_current_position()
                if pos:
                    self._robot_x = pos['x']; self._robot_y = pos['y']
                    near, d = self.assistant.robot_navigator.find_nearest_location(pos)
                    txt = f"AMCL  x={pos['x']:+.2f}  y={pos['y']:+.2f}  ▸ {near} ({d:.1f}m)"
                    self.root.after(0, lambda: self._pos_var.set(txt))
            except Exception:
                pass
        threading.Thread(target=_do, daemon=True).start()

    def _refresh_position_loop(self):
        self._refresh_position()
        self.root.after(500 if not self._nav_active else 300, self._refresh_position_loop)

    def _refresh_battery_loop(self):
        if self.assistant:
            pct = self.assistant.battery_percentage
            if pct is not None:
                p = max(0, min(100, int(round(float(pct)*100))))
                fg = C["green"] if p>50 else (C["amber"] if p>20 else C["red"])
                self._bat_var.set(f"🔋 {p}%"); self._bat_lbl.configure(fg=fg)
            else:
                self._bat_var.set("🔋 --%")
        self.root.after(5000, self._refresh_battery_loop)

    def _check_bt_status(self):
        def _do():
            try:
                import subprocess
                r = subprocess.run(["pactl","list","short","sinks"],
                                   capture_output=True, text=True, timeout=3)
                found = any(l.split()[1].startswith("bluez_output.")
                            for l in r.stdout.splitlines() if len(l.split())>=2)
                fg = C["green"] if found else C["red"]
                txt = "🔊 ✓" if found else "🔊 ✗"
                self.root.after(0, lambda: (
                    self._spk_var.set(txt), self._spk_lbl.configure(fg=fg)))
            except: pass
        threading.Thread(target=_do, daemon=True).start()

    # ══════════════════════════════════════════════════════════════════
    #  Input / interaction
    # ══════════════════════════════════════════════════════════════════
    def _on_send(self, event=None):
        text = self._entry.get().strip()
        if not text: return
        self._append("you", "YOU: "); self._append("body", text+"\n")
        self._entry.delete(0,"end")
        if not self._history or self._history[-1]!=text:
            self._history.append(text)
        self._hist_idx = -1
        _input_value[0]=text; _input_event.set()

    def _history_up(self, _=None):
        if not self._history: return
        self._hist_idx = (len(self._history)-1 if self._hist_idx==-1
                          else max(0, self._hist_idx-1))
        self._entry.delete(0,"end"); self._entry.insert(0, self._history[self._hist_idx])

    def _history_down(self, _=None):
        if self._hist_idx==-1: return
        if self._hist_idx < len(self._history)-1:
            self._hist_idx += 1
            self._entry.delete(0,"end"); self._entry.insert(0, self._history[self._hist_idx])
        else:
            self._hist_idx=-1; self._entry.delete(0,"end")

    def _on_toggle_recording(self):
        if not self._voice_recording: self._on_start_recording()
        else:                         self._on_stop_recording()

    def _on_start_recording(self):
        if not self.assistant or not self.assistant.voice_input:
            self._append("warn","Voice input not available.\n"); return
        if self._voice_recording: return
        self._voice_recording = True
        self._rec_btn.configure(text="⏹", bg=C["red"])
        self._set_status("🎤 Recording…", C["green"])
        import sounddevice as _sd
        SR   = getattr(self.assistant.voice_input,'_native_sr',48000)
        dev  = getattr(self.assistant.voice_input,'_preferred_input_device',None)
        nch  = getattr(self.assistant.voice_input,'_mic_channels',1)
        CHUNK= int(0.1*SR)
        self._rec_buf = collections.deque(); self._rec_native_sr = SR
        def _cb(indata,frames,t,s): self._rec_buf.append(indata[:,0].copy())
        kw = dict(samplerate=SR,channels=nch,dtype="float32",blocksize=CHUNK,callback=_cb)
        if dev is not None: kw['device']=dev
        try:
            s=_sd.InputStream(**kw); s.start(); self._rec_stream_obj=s
        except Exception as e:
            self._append("warn",f"Mic error: {e}\n"); self._voice_recording=False
            self.root.after(0,self._reset_rec_btn)

    def _reset_rec_btn(self):
        self._rec_btn.configure(text="🎤", bg=C["green"])

    def _on_stop_recording(self):
        if not self._voice_recording or not self._rec_stream_obj: return
        s=self._rec_stream_obj; self._rec_stream_obj=None
        s.stop(); s.close(); self._voice_recording=False; self._reset_rec_btn()
        buf=list(self._rec_buf); sr=self._rec_native_sr; self._rec_buf=None
        def _do():
            import numpy as _np
            try:
                if not buf: self._queue.put(("print","⚠ No audio.\n")); return
                audio=_np.concatenate(buf)
                locs=self.assistant.location_manager.get_all_locations()
                vi=self.assistant.voice_input; _sr=getattr(vi,'_native_sr',48000)
                vi._native_sr=sr
                try:    text=vi.transcribe_audio(audio, context_locations=locs)
                finally: vi._native_sr=_sr
                if text: self._queue.put(("print",f"YOU (voice): {text}\n")); \
                         _input_value[0]=text; _input_event.set()
                else:    self._queue.put(("print","⚠ Could not understand audio.\n"))
            except Exception as e:
                self._queue.put(("print",f"Transcription error: {e}\n"))
            finally:
                self.root.after(0, lambda: self._set_status("⬤ Ready", C["green"]))
        threading.Thread(target=_do, daemon=True).start()

    def _on_location_click(self, _=None):
        sel=self._loc_list.curselection()
        if not sel: return
        raw=self._loc_list.get(sel[0]).strip()
        name=raw.split("(")[0].strip().lstrip("0123456789. ")
        self._entry.delete(0,"end"); self._entry.insert(0,f"go to {name}")

    def _on_goto(self):
        sel=self._loc_list.curselection()
        if not sel: return
        raw=self._loc_list.get(sel[0]).strip()
        name=raw.split("(")[0].strip().lstrip("0123456789. ")
        cmd=f"go to {name}"
        self._append("you","YOU: "); self._append("body",cmd+"\n")
        _input_value[0]=cmd; _input_event.set()

    # ══════════════════════════════════════════════════════════════════
    #  Mode toggle
    # ══════════════════════════════════════════════════════════════════
    def _toggle_mode(self):
        if not self.assistant: return
        if self._mode=="text":
            if not self.assistant.voice_input:
                self._append("warn","No voice hardware.\n"); return
            self._mode="voice"
            self.assistant.voice_input.wake_word_enabled=False
            self.assistant.chat_interface.set_mode("voice")
            self._mode_var.set("VOICE 🎤")
            self._mode_btn.configure(text="⌨  Switch to Text")
            self._send_btn.grid_remove(); self._entry.configure(state="disabled")
            self._rec_btn.configure(state="normal"); self._rec_btn.grid()
            self._append("system","🎤 Voice mode active.\n")
        else:
            self._mode="text"
            if self.assistant.voice_input:
                self.assistant.voice_input.wake_word_enabled=True
            self.assistant.chat_interface.set_mode("text")
            self._mode_var.set("TEXT ⌨")
            self._mode_btn.configure(text="🎤  Switch to Voice")
            self._rec_btn.grid_remove(); self._rec_btn.configure(state="disabled")
            self._entry.configure(state="normal"); self._send_btn.grid()
            self._append("system","⌨ Text mode active.\n")

    def _sync_mode_widgets(self, new_mode):
        if new_mode==self._mode: return
        self._mode=new_mode
        if new_mode=="voice":
            self._mode_var.set("VOICE 🎤")
            self._mode_btn.configure(text="⌨  Switch to Text")
            self._send_btn.grid_remove(); self._entry.configure(state="disabled")
            if self.assistant and self.assistant.voice_input:
                self._rec_btn.configure(state="normal")
            self._rec_btn.grid()
        else:
            self._mode_var.set("TEXT ⌨")
            self._mode_btn.configure(text="🎤  Switch to Voice")
            self._rec_btn.grid_remove(); self._rec_btn.configure(state="disabled")
            self._entry.configure(state="normal"); self._send_btn.grid()

    # ══════════════════════════════════════════════════════════════════
    #  Misc
    # ══════════════════════════════════════════════════════════════════
    def _ask_position(self):
        _input_value[0]="where am i"; _input_event.set()

    def _list_locations(self):
        if not self.assistant: return
        self._append("system","\n📍 Available Locations:\n")
        for i,nm in enumerate(self.assistant.location_manager.get_all_locations(),1):
            self._append("body",f"  {i:2d}. {nm}\n")
        self._append("dim","─"*40+"\n")

    def _stop_speaking(self):
        if self.assistant and self.assistant.voice_output:
            self.assistant.voice_output.stop()
            self._append("warn","⏹  Speech stopped.\n")

    def _clear_chat(self):
        self._chat.configure(state="normal"); self._chat.delete("1.0","end")
        self._chat.configure(state="disabled")
        self._append("system","🗑  Chat cleared. (Memory retained)\n")

    def _detect_gpu_async(self):
        """Run GPU detection off the main thread; update UI via root.after."""
        try:
            from response_generator import InferenceConfig
            found = InferenceConfig.detect_gpu()
        except Exception:
            found = False
        self.root.after(0, lambda: self._apply_gpu_detection(found))

    def _apply_gpu_detection(self, found: bool):
        """Called on main thread once GPU detection completes — builds menu."""
        self._gpu_available = found
        m = self._inference_menu
        m.delete(0, "end")
        m.add_command(label="🖥  CPU", command=lambda: self._set_inference("cpu"))
        if found:
            m.add_command(label="⚡ GPU", command=lambda: self._set_inference("gpu"))
            self._gpu_status_var.set("GPU detected")
        else:
            m.add_command(label="⚡ GPU  (not available)",
                          state="disabled", foreground=C["tx3"])
            self._gpu_status_var.set("CPU only")
        m.add_separator()
        m.add_command(label=f"Status: {self._gpu_status_var.get()}",
                      state="disabled", foreground=C["tx3"])

    def _show_inference_menu(self):
        """Pop up the inference dropdown below the button."""
        btn = self._infer_btn
        x = btn.winfo_rootx()
        y = btn.winfo_rooty() + btn.winfo_height()
        self._inference_menu.tk_popup(x, y)

    def _set_inference(self, mode: str):
        """Apply CPU/GPU selection to InferenceConfig and update button label."""
        use_gpu = (mode == "gpu")
        label = "⚡ GPU" if use_gpu else "🖥  CPU"
        self._inference_var.set(label)
        try:
            cfg = getattr(self.assistant, "inference_config", None)
            if cfg:
                cfg.set_gpu(use_gpu)
            display = "GPU ⚡" if use_gpu else "CPU 🖥"
            self._append("info", f"ℹ  LLM inference → {display}\n")
        except Exception as e:
            self._append("warn", f"⚠  Inference config error: {e}\n")

    def _on_inference_change(self):
        """Legacy shim — no longer used (kept for safety)."""
        pass

    def _detect_llm_models_async(self):
        """Run `ollama list` off the main thread; update the dropdown via root.after."""
        try:
            from response_generator import InferenceConfig
            installed = InferenceConfig.detect_installed_models()
        except Exception:
            installed = []
        self.root.after(0, lambda: self._apply_llm_models(installed))

    def _apply_llm_models(self, installed: list):
        """Called on main thread once model detection completes — (re)builds
        the LLM dropdown so it only lists models that are actually installed."""
        self._llm_installed = installed
        m = self._llm_menu
        m.delete(0, "end")

        if not installed:
            m.add_command(
                label="No Ollama models found — run 'ollama pull <model>'",
                state="disabled", foreground=C["tx3"])
            m.add_separator()
            m.add_command(label="⟳  Rescan installed models",
                           command=self._rescan_llm_models)
            self._llm_var.set("LLM: none installed")
            return

        from response_generator import InferenceConfig as _InfCfg
        for tag in installed:
            info = _InfCfg.get_model_info(tag)
            m.add_command(
                label=info["label"],
                command=lambda t=tag: self._set_llm_model(t))
        m.add_separator()
        m.add_command(label="⟳  Rescan installed models",
                       command=self._rescan_llm_models)

        # Keep current selection if it's still installed, else fall back to
        # the first installed model (and actually apply that fallback).
        cfg = getattr(self.assistant, "inference_config", None)
        current = cfg.model if cfg else _InfCfg.DEFAULT_MODEL
        if current not in installed:
            current = installed[0]
            if cfg:
                cfg.set_model(current)
        info = _InfCfg.get_model_info(current)
        self._llm_var.set(info["label"])

    def _rescan_llm_models(self):
        """Manual refresh — e.g. after pulling a new model without restarting the GUI."""
        self._llm_var.set("LLM: rescanning…")
        import threading as _th
        _th.Thread(target=self._detect_llm_models_async, daemon=True).start()

    def _show_llm_menu(self):
        """Pop up the LLM-model dropdown below the button."""
        btn = self._llm_btn
        x = btn.winfo_rootx()
        y = btn.winfo_rooty() + btn.winfo_height()
        self._llm_menu.tk_popup(x, y)

    def _set_llm_model(self, model_tag: str):
        """Apply the selected Ollama model to the shared InferenceConfig,
        update the button label, and tell the user how current that
        model's knowledge is (its training-data cutoff)."""
        from response_generator import InferenceConfig as _InfCfg
        info = _InfCfg.get_model_info(model_tag)
        self._llm_var.set(info["label"])
        try:
            cfg = getattr(self.assistant, "inference_config", None)
            if cfg:
                cfg.set_model(model_tag)
            msg = f"🤖  LLM switched → {info['label']}\n" \
                  f"     Data current through: {info['cutoff']}\n"
            if info.get("note"):
                msg += f"     {info['note']}\n"
            self._append("info", msg)
        except Exception as e:
            self._append("warn", f"⚠  LLM model switch error: {e}\n")

    def _poll_arrivals(self):
        """Timer callback: drain arrival queue and trigger announcements."""
        try:
            ci = getattr(getattr(self, "assistant", None), "chat_interface", None)
            if ci:
                ci._process_arrivals()
        except Exception:
            pass
        self.root.after(300, self._poll_arrivals)

    def _stop_navigation(self):
        """Stop Navigation button handler."""
        ci = getattr(getattr(self, "assistant", None), "chat_interface", None)
        if ci:
            ci.cancel_navigation()
            self._append("warn", "🛑 Navigation cancelled.\n")
            self._set_status("Navigation cancelled", C["red"])
            self._navqueue_var.set("")
            try: self._stopnav_btn.configure(state="disabled")
            except Exception: pass

    def _set_status(self, text, colour=None):
        self._status_var.set(text)
        if colour: self._status_lbl.configure(fg=colour)

    def _set_generating(self, active):
        if active:
            self._generating_var.set("⏳ Generating response…")
            self._blink()
        else:
            if self._generating_blink_id:
                self.root.after_cancel(self._generating_blink_id)
                self._generating_blink_id=None
            self._generating_var.set("")

    def _blink(self):
        v=self._generating_var.get()
        if not v: return
        self._generating_var.set(
            "⏳ Generating response" if v.endswith("…") else "⏳ Generating response…")
        self._generating_blink_id=self.root.after(500, self._blink)

    def on_close(self):
        _input_value[0]="exit"; _input_event.set()
        self.root.after(500, self.root.destroy)


# ═══════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════
def main():
    args=sys.argv[1:]
    if "--help" in args or "-h" in args:
        builtins._real_print("""
TurtleBot Navigation Assistant — GUI Edition
Usage:  python3 gui.py [options]
  --text-only            Disable voice
  --voice-model SIZE     tiny|base|small|medium|large-v3
  --sensitivity LEVEL    low|medium|high|very_high
"""); return
    root=tk.Tk()
    app =TurtlebotGUI(root, args)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

if __name__=="__main__":
    main()