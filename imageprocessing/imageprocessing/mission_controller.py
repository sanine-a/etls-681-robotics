import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Imu
from std_msgs.msg import String, Float32, Bool


class MissionController(Node):
    def __init__(self):
        super().__init__('mission_controller')

        # Subscribers
        self.create_subscription(String, '/vision/gesture_labels', self.gesture_callback, 10)
        self.create_subscription(String, '/vision/object_labels', self.object_callback, 10)
        self.create_subscription(Float32, '/vision/line_error', self.line_error_callback, 10)
        self.create_subscription(Bool, '/vision/line_detected', self.line_detected_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(Imu, '/imu', self.imu_callback, 10)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_pub = self.create_publisher(String, '/arm/command', 10)

        # State machine definitions
        self.STATE_FOLLOWING_LINE = 1
        self.STATE_PAUSED = 2
        self.STATE_EVALUATING = 3
        self.STATE_WATER_BOTTLE_SEQUENCE = 4
        self.STATE_AVOID_OBSTACLE = 5
        self.STATE_DONE = 6

        self.current_state = self.STATE_FOLLOWING_LINE

        # Variables
        self.current_gesture = "None"
        self.current_objects = []
        self.line_error = 0.0
        self.line_detected = True
        self.obstacle_detected_ahead = False
        self.navigation_paused = False
        self.current_yaw = 0.0
        self.lidar_front_min = float('inf')
        self.lidar_left_min = float('inf')
        self.lidar_right_min = float('inf')
        self.turn_target_yaw = None

        # Sub-timers or step counters for multi-stage maneuvers
        self.maneuver_step = 0
        self.maneuver_timer = self.now_seconds()

        # Control Loop Timer (10 Hz)
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("MissionController node fully initialized with complete state machine.")

    def now_seconds(self):
        return self.get_clock().now().nanoseconds / 1e9

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def quaternion_to_yaw(self, orientation):
        x = orientation.x
        y = orientation.y
        z = orientation.z
        w = orientation.w
        roll_pitch_yaw = 2.0 * (w * z + x * y)
        denom = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(roll_pitch_yaw, denom)

    def set_state(self, new_state, message):
        if new_state != self.current_state:
            self.current_state = new_state
            self.maneuver_step = 0
            self.maneuver_timer = self.now_seconds()
            self.turn_target_yaw = None
            self.get_logger().info(message)

    def gesture_callback(self, msg):
        self.current_gesture = msg.data
        if self.current_gesture == "open_hand":
            self.navigation_paused = True
            self.get_logger().info("Navigation paused via Open Hand gesture.")
        elif self.current_gesture == "thumbs_up":
            self.navigation_paused = False
            self.get_logger().info("Navigation resumed via Thumbs Up gesture.")

    def object_callback(self, msg):
        if msg.data:
            self.current_objects = [item.strip().lower() for item in msg.data.split(',') if item.strip()]
        else:
            self.current_objects = []

    def line_error_callback(self, msg):
        self.line_error = float(msg.data)

    def line_detected_callback(self, msg):
        self.line_detected = bool(msg.data)

    def scan_callback(self, msg):
        front_ranges = msg.ranges[0:15] + msg.ranges[345:360]
        left_ranges = msg.ranges[45:135]
        right_ranges = msg.ranges[225:315]

        valid_front = [r for r in front_ranges if not math.isnan(r) and r > 0.05]
        valid_left = [r for r in left_ranges if not math.isnan(r) and r > 0.05]
        valid_right = [r for r in right_ranges if not math.isnan(r) and r > 0.05]

        self.lidar_front_min = min(valid_front) if valid_front else float('inf')
        self.lidar_left_min = min(valid_left) if valid_left else float('inf')
        self.lidar_right_min = min(valid_right) if valid_right else float('inf')

        self.obstacle_detected_ahead = self.lidar_front_min < 0.24

    def imu_callback(self, msg):
        orientation = msg.orientation
        self.current_yaw = self.quaternion_to_yaw(orientation)

    def rotate_to_yaw(self, twist, target_yaw, turn_rate=0.7):
        if self.turn_target_yaw is None:
            self.turn_target_yaw = target_yaw

        angle_error = self.normalize_angle(self.turn_target_yaw - self.current_yaw)
        if abs(angle_error) < 0.12:
            self.turn_target_yaw = None
            twist.angular.z = 0.0
            return True

        twist.angular.z = max(-turn_rate, min(turn_rate, angle_error * 1.5))
        twist.linear.x = 0.0
        return False

    def control_loop(self):
        twist = Twist()
        now = self.now_seconds()

        # 1. Manual override via gesture check
        if self.navigation_paused:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)
            return

        if self.current_state == self.STATE_DONE:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)
            return

        # 2. State machine execution
        if self.current_state == self.STATE_FOLLOWING_LINE:
            if not self.line_detected or self.obstacle_detected_ahead:
                self.set_state(self.STATE_EVALUATING, "Line lost or obstacle detected. Switching to evaluation state.")
                return

            kp = 0.8
            twist.linear.x = 0.12
            twist.angular.z = -kp * self.line_error
            self.cmd_vel_pub.publish(twist)

        elif self.current_state == self.STATE_EVALUATING:
            if self.maneuver_step == 0:
                twist.linear.x = -0.08
                twist.angular.z = 0.0
                self.cmd_vel_pub.publish(twist)
                if now - self.maneuver_timer > 0.8:
                    self.maneuver_step = 1
                    self.maneuver_timer = now
            elif self.maneuver_step == 1:
                if "ending_object" in self.current_objects:
                    self.set_state(self.STATE_DONE, "Ending object reached. Stopping navigation.")
                    return
                if any(label in self.current_objects for label in ["water bottle", "bottle"]):
                    self.set_state(self.STATE_WATER_BOTTLE_SEQUENCE, "Water bottle detected. Starting pickup sequence.")
                    return
                self.set_state(self.STATE_AVOID_OBSTACLE, "Generic obstacle detected. Starting avoidance maneuver.")
                return

        elif self.current_state == self.STATE_WATER_BOTTLE_SEQUENCE:
            if self.maneuver_step == 0:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.cmd_vel_pub.publish(twist)
                self.arm_pub.publish(String(data="grab"))
                self.get_logger().info("Arm command published: grab")
                self.maneuver_step = 1
                self.maneuver_timer = now

            elif self.maneuver_step == 1:
                if now - self.maneuver_timer > 3.0:
                    self.maneuver_step = 2
                    self.turn_target_yaw = self.normalize_angle(self.current_yaw + math.pi)

            elif self.maneuver_step == 2:
                if self.rotate_to_yaw(twist, self.turn_target_yaw, turn_rate=0.8):
                    self.maneuver_step = 3
                    self.maneuver_timer = now
                else:
                    self.cmd_vel_pub.publish(twist)
                    return

            elif self.maneuver_step == 3:
                twist.linear.x = -0.15
                twist.angular.z = 0.0
                self.cmd_vel_pub.publish(twist)
                if now - self.maneuver_timer > 2.5:
                    self.maneuver_step = 4
                    self.maneuver_timer = now

            elif self.maneuver_step == 4:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.cmd_vel_pub.publish(twist)
                self.arm_pub.publish(String(data="release"))
                self.get_logger().info("Arm command published: release")
                self.maneuver_step = 5
                self.maneuver_timer = now

            elif self.maneuver_step == 5:
                if self.rotate_to_yaw(twist, self.normalize_angle(self.current_yaw + math.pi), turn_rate=0.8):
                    self.set_state(self.STATE_FOLLOWING_LINE, "Water bottle task complete. Resuming line following.")
                    return
                self.cmd_vel_pub.publish(twist)
                return

        elif self.current_state == self.STATE_AVOID_OBSTACLE:
            if self.maneuver_step == 0:
                self.turn_target_yaw = None
                if self.lidar_left_min >= self.lidar_right_min:
                    direction = 1.0
                else:
                    direction = -1.0

                self.turn_target_yaw = self.normalize_angle(self.current_yaw + direction * 0.9)
                self.maneuver_step = 1
                self.maneuver_timer = now

            elif self.maneuver_step == 1:
                if self.rotate_to_yaw(twist, self.turn_target_yaw, turn_rate=0.6):
                    self.maneuver_step = 2
                    self.maneuver_timer = now
                else:
                    self.cmd_vel_pub.publish(twist)
                    return

            elif self.maneuver_step == 2:
                twist.linear.x = 0.12
                twist.angular.z = 0.0
                self.cmd_vel_pub.publish(twist)
                if self.lidar_front_min > 0.45 and self.line_detected:
                    self.maneuver_step = 3
                    self.maneuver_timer = now

            elif self.maneuver_step == 3:
                if self.rotate_to_yaw(twist, self.normalize_angle(self.current_yaw - self.turn_target_yaw + self.current_yaw), turn_rate=0.6):
                    self.set_state(self.STATE_FOLLOWING_LINE, "Line re-acquired! Resuming normal navigation.")
                    return
                self.cmd_vel_pub.publish(twist)
                return

        if self.current_state == self.STATE_PAUSED:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        elif self.current_state == self.STATE_DONE:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        self.cmd_vel_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = MissionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()