import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Imu
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String, Float32, Bool


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class MissionController(Node):
    def __init__(self):
        super().__init__('mission_controller')

        # --- Tunable Parameters ---
        self.kp = 0.8
        self.line_drive_speed = 0.12

        # Camera is 8 cm to the right of robot center; tune this setpoint during testing
        self.camera_offset_setpoint = -0.4

        # LiDAR & backup settings (~24 cm bottom-of-camera reference)
        self.obstacle_threshold_m = 0.24
        self.backup_speed = -0.10
        self.backup_duration_sec = 0.6

        # Nav2 bypass settings (~50 cm ahead, matching original orientation)
        self.nav2_bypass_distance_m = 0.50

        # --- Subscribers ---
        self.create_subscription(String, '/vision/gesture_labels', self.gesture_callback, 10)
        self.create_subscription(Float32, '/vision/line_error', self.line_error_callback, 10)
        self.create_subscription(Bool, '/vision/line_detected', self.line_detected_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Imu, '/imu', self.imu_callback, 10)

        # --- Publishers ---
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.nav2_goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        # --- State ---
        self.navigation_paused = False
        self.line_error = 0.0
        self.line_detected = False
        self.obstacle_detected_ahead = False

        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0
        self.has_odom = False
        self.imu_yaw = 0.0

        self.mode = "FOLLOW_LINE"  # FOLLOW_LINE, BACKING_UP, NAV2_BYPASS
        self.step_start_time = None
        self.saved_yaw_at_object = 0.0
        self.nav2_goal_sent = False
        self.nav2_target_x = 0.0
        self.nav2_target_y = 0.0

        # 10 Hz control loop
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("MissionController initialized with line following and obstacle bypass.")

    def now_sec(self):
        return self.get_clock().now().nanoseconds / 1e9

    def current_yaw(self):
        return self.odom_yaw if self.has_odom else self.imu_yaw

    def gesture_callback(self, msg):
        gesture = msg.data.strip()
        if gesture == "open_hand" and not self.navigation_paused:
            self.navigation_paused = True
            self.publish_stop()
            self.get_logger().info("Open hand: paused.")
        elif gesture == "thumbs_up" and self.navigation_paused:
            self.navigation_paused = False
            self.get_logger().info("Thumbs up: resumed.")

    def line_error_callback(self, msg):
        self.line_error = float(msg.data)

    def line_detected_callback(self, msg):
        self.line_detected = bool(msg.data)

    def scan_callback(self, msg):
        if not msg.ranges:
            self.obstacle_detected_ahead = False
            return

        angle_window = math.radians(15.0) # 15 degrees
        min_dist = float('inf')

        for i, r in enumerate(msg.ranges):
            if math.isnan(r) or math.isinf(r) or r <= 0.05:
                continue
            angle = normalize_angle(msg.angle_min + i * msg.angle_increment)
            if abs(angle) <= angle_window and r < min_dist:
                min_dist = r

        self.obstacle_detected_ahead = (min_dist < self.obstacle_threshold_m)

    def odom_callback(self, msg):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        self.odom_yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self.has_odom = True

    def imu_callback(self, msg):
        self.imu_yaw = yaw_from_quaternion(msg.orientation)

    def publish_twist(self, linear_x=0.0, angular_z=0.0):
        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(twist)

    def publish_stop(self):
        self.publish_twist(0.0, 0.0)

    def send_nav2_bypass_goal(self):
        goal_x = self.odom_x + self.nav2_bypass_distance_m * math.cos(self.saved_yaw_at_object)
        goal_y = self.odom_y + self.nav2_bypass_distance_m * math.sin(self.saved_yaw_at_object)

        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = "map"
        goal.pose.position.x = goal_x
        goal.pose.position.y = goal_y
        goal.pose.position.z = 0.0
        goal.pose.orientation.z = math.sin(self.saved_yaw_at_object / 2.0)
        goal.pose.orientation.w = math.cos(self.saved_yaw_at_object / 2.0)

        self.nav2_goal_pub.publish(goal)
        self.nav2_target_x = goal_x
        self.nav2_target_y = goal_y
        self.nav2_goal_sent = True
        self.get_logger().info(
            f"Nav2 bypass goal sent to ({goal_x:.2f}, {goal_y:.2f}), yaw={self.saved_yaw_at_object:.2f} rad."
        )

    def control_loop(self):
        if self.navigation_paused:
            self.publish_stop()
            return

        if self.mode == "FOLLOW_LINE":
            if not self.line_detected or self.obstacle_detected_ahead:
                self.publish_stop()
                self.saved_yaw_at_object = self.current_yaw()
                self.step_start_time = self.now_sec()
                self.mode = "BACKING_UP"
                self.get_logger().info("Line lost or obstacle ahead: backing up.")
                return

            error = self.line_error - self.camera_offset_setpoint
            angular_z = -self.kp * error
            self.publish_twist(self.line_drive_speed, angular_z)

        elif self.mode == "BACKING_UP":
            if (self.now_sec() - self.step_start_time) < self.backup_duration_sec:
                self.publish_twist(self.backup_speed, 0.0)
            else:
                self.publish_stop()
                self.nav2_goal_sent = False
                self.mode = "NAV2_BYPASS"

        elif self.mode == "NAV2_BYPASS":
            if not self.nav2_goal_sent:
                self.send_nav2_bypass_goal()
                return

            dist_to_goal = math.hypot(
                self.nav2_target_x - self.odom_x, self.nav2_target_y - self.odom_y
            )
            yaw_err = normalize_angle(self.saved_yaw_at_object - self.current_yaw())

            if dist_to_goal < 0.10:
                if abs(yaw_err) > 0.08:
                    self.publish_twist(0.0, max(-0.6, min(0.6, 1.5 * yaw_err)))
                else:
                    self.publish_stop()
                    self.mode = "FOLLOW_LINE"
                    self.get_logger().info("Bypass complete and reoriented: resuming line following.")


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