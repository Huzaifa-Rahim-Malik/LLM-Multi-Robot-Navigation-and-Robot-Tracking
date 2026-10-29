#!/usr/bin/env python3

import math
import threading
import time

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from rclpy.executors import SingleThreadedExecutor
from transforms3d.euler import euler2quat, quat2euler


class RobotNavigator:
    """Handles robot navigation via Nav2 and tracks pose from AMCL."""

    _CARDINALS = ['East', 'Northeast', 'North', 'Northwest',
                  'West', 'Southwest', 'South', 'Southeast']

    def __init__(self, logger, location_manager, node):
        self.logger = logger
        self.location_manager = location_manager
        self.node = node
        self.navigator = BasicNavigator()

        self.current_pose = None
        self.pose_received = False

        # Subscribe to AMCL pose
        self.node.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self._pose_callback, 10
        )
        self.logger.info("Subscribed to /amcl_pose")

        # Dedicated ROS executor thread — keeps callbacks firing without blocking callers
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin, daemon=True, name="ros_executor"
        )
        self._spin_thread.start()
        self.logger.info("ROS executor thread started")

        self._wait_for_initial_pose()

    # ── Pose ──────────────────────────────────────────────────────────────

    def _pose_callback(self, msg):
        self.current_pose = msg.pose.pose
        self.pose_received = True

    def _wait_for_initial_pose(self, timeout: float = 30.0):
        self.logger.info("Waiting for initial AMCL pose…")
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.pose_received:
                self.logger.info("Initial pose received")
                return
            time.sleep(0.1)
        self.logger.warn(f"No AMCL pose after {timeout}s — continuing anyway")

    def _poll_pose(self, timeout: float = 3.0):
        """Block briefly until pose is available (executor handles callbacks)."""
        deadline = time.time() + timeout
        while time.time() < deadline and not self.pose_received:
            time.sleep(0.05)

    def get_current_position(self) -> dict | None:
        if not self.pose_received:
            self._poll_pose()
        if self.current_pose is None:
            return None
        try:
            return {'x': self.current_pose.position.x,
                    'y': self.current_pose.position.y}
        except Exception as e:
            self.logger.error(f"get_current_position error: {e}")
            return None

    def get_current_orientation(self) -> dict | None:
        if not self.pose_received:
            self._poll_pose()
        if self.current_pose is None:
            return None
        try:
            q = self.current_pose.orientation
            _, _, yaw = quat2euler([q.w, q.x, q.y, q.z])
            deg = math.degrees(yaw) % 360
            return {
                'yaw_radians': yaw,
                'yaw_degrees': deg,
                'direction': self._cardinal(deg),
            }
        except Exception as e:
            self.logger.error(f"get_current_orientation error: {e}")
            return None

    def _cardinal(self, degrees: float) -> str:
        return self._CARDINALS[int((degrees + 22.5) / 45) % 8]

    # ── Navigation ────────────────────────────────────────────────────────

    def _make_pose(self, x: float, y: float, theta: float) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.navigator.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        quat = euler2quat(0, 0, theta)
        pose.pose.orientation.x = quat[1]
        pose.pose.orientation.y = quat[2]
        pose.pose.orientation.z = quat[3]
        pose.pose.orientation.w = quat[0]
        return pose

    def navigate_to_location(self, target) -> bool:
        """Navigate to a named location string or a coordinate dict."""
        if isinstance(target, dict) and 'is_coordinate' in target:
            x, y, theta = target['x'], target['y'], target.get('theta', 0.0)
            self.logger.info(f"Navigating to coords ({x}, {y})")
        elif isinstance(target, str):
            if not self.location_manager.location_exists(target):
                self.logger.error(f"Unknown location: '{target}'")
                return False
            loc = self.location_manager.get_location(target)
            x, y, theta = loc['x'], loc['y'], loc['theta']
            self.logger.info(f"Navigating to {target} ({x}, {y})")
        else:
            self.logger.error("navigate_to_location: invalid target type")
            return False

        # Skip if already within tolerance
        pos = self.get_current_position()
        if pos and math.hypot(pos['x'] - x, pos['y'] - y) < 0.3:
            self.logger.info("Already at destination")
            return True

        self.logger.info("Waiting for Nav2…")
        self.navigator.waitUntilNav2Active()

        self.navigator.goToPose(self._make_pose(x, y, theta))

        start = time.time()
        TIMEOUT = 300
        last_log = 0.0
        while not self.navigator.isTaskComplete():
            time.sleep(0.1)
            elapsed = time.time() - start
            if elapsed - last_log >= 10.0:
                self.logger.info(f"Navigating… {int(elapsed)}s")
                last_log = elapsed
            if elapsed > TIMEOUT:
                self.logger.error("Navigation timeout")
                self.navigator.cancelTask()
                return False

        result = self.navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            self.logger.info("Arrived at destination")
            return True
        self.logger.warn(f"Navigation ended: {result}")
        return False

    def find_nearest_location(self, pos: dict) -> tuple[str | None, float]:
        best_name, best_dist = None, float('inf')
        for name in self.location_manager.get_all_locations():
            loc = self.location_manager.get_location(name)
            d = math.hypot(pos['x'] - loc['x'], pos['y'] - loc['y'])
            if d < best_dist:
                best_dist, best_name = d, name
        return best_name, best_dist

    # ── Shutdown ──────────────────────────────────────────────────────────

    def shutdown(self):
        try:
            self._executor.shutdown()
        except Exception:
            pass
        try:
            self.navigator.lifecycleShutdown()
        except Exception:
            pass