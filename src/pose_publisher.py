#!/usr/bin/env python3
"""
pose_publisher.py
=================
Publishes robot state on two topics:

    /robot/pose      → geometry_msgs/PoseWithCovarianceStamped  (10 Hz)
    /robot_pos_data  → std_msgs/String  (on arrival, published 3× with 0.3 s gaps
                        to ensure the subscriber receives it despite executor jitter)

Status string format (robot_pos_data):
    "x:<val> y:<val>"   — published on arrival only
"""

import threading
import time

from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import String

# How many times to repeat the arrival publish and the gap between repeats.
_ARRIVAL_REPEAT_COUNT = 1
_ARRIVAL_REPEAT_DELAY = 0   # seconds


class PosePublisher:

    def __init__(self, node, robot_navigator):
        self.node = node
        self.robot_navigator = robot_navigator
        self._status = "idle"

        self._pose_pub = node.create_publisher(PoseWithCovarianceStamped, '/robot/pose', 10)
        self._status_pub = node.create_publisher(String, '/robot_pos_data', 10)

        node.create_timer(0.1, self._publish_pose)

        node.get_logger().info("PosePublisher ready — /robot/pose  /robot_pos_data")

    # ── Internal ──────────────────────────────────────────────────────────

    def _publish_pose(self):
        if self.robot_navigator.current_pose is None:
            return
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.pose = self.robot_navigator.current_pose
        self._pose_pub.publish(msg)

    def _publish_status(self, text: str):
        msg = String()
        msg.data = text
        self._status_pub.publish(msg)
        self.node.get_logger().info(f"[PosePublisher] → {text}")

    def _publish_arrival_repeated(self, payload: str):
        """
        Publish the arrival payload _ARRIVAL_REPEAT_COUNT times on a background
        thread so the chat loop is not blocked, and so transient executor/
        subscriber delays don't cause the message to be silently dropped.
        """
        def _worker():
            for i in range(_ARRIVAL_REPEAT_COUNT):
                self._publish_status(payload)
                if i < _ARRIVAL_REPEAT_COUNT - 1:
                    time.sleep(_ARRIVAL_REPEAT_DELAY)

        t = threading.Thread(target=_worker, daemon=True, name="arrival_pub")
        t.start()

    # ── Public API ────────────────────────────────────────────────────────

    def set_status(self, status: str, location=None):
        """
        status   : "navigating" | "arrived" | "idle"
        location : location name string, coordinate dict, or None

        /robot_pos_data is published on arrival only, in the format:
            "x:<val> y:<val>"

        Coordinate priority on arrival:
          1. Target location from YAML / coordinate dict  ← most reliable at
             the exact moment Nav2 reports SUCCEEDED
          2. Live AMCL pose                               ← may lag by one
             callback cycle; used as cross-check only
        Nothing is published for "navigating" or "idle".
        """
        self._status = status

        if status == "arrived":
            x, y = None, None

            # ── Priority 1: use the known target coordinates ──────────────
            # These are always available and represent where Nav2 was told to
            # go, which is correct for the follower robot's needs.
            if isinstance(location, dict) and 'x' in location:
                x = round(location['x'], 2)
                y = round(location['y'], 2)
            elif isinstance(location, str):
                loc_data = self.robot_navigator.location_manager.get_location(location)
                if loc_data:
                    x = round(loc_data['x'], 2)
                    y = round(loc_data['y'], 2)

            # ── Priority 2: fall back to live AMCL pose if target unknown ─
            if x is None:
                pose = self.robot_navigator.current_pose
                if pose is not None:
                    x = round(pose.position.x, 2)
                    y = round(pose.position.y, 2)
                else:
                    x, y = 0.0, 0.0

            payload = f"x:{x} y:{y}"
            self.node.get_logger().info(
                f"[PosePublisher] Arrival — publishing '{payload}' "
                f"({_ARRIVAL_REPEAT_COUNT}× every {_ARRIVAL_REPEAT_DELAY}s)"
            )
            self._publish_arrival_repeated(payload)

        # "navigating" and "idle" — no publish on /robot_pos_data