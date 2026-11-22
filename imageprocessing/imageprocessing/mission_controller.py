import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float32, Bool


class MissionController(Node):
    def __init__(self):
        super().__init__('mission_controller')

        # --- Tunable Parameters ---
        self.kp = 0.8
        self.line_drive_speed = 0.12

        # Camera is 8 cm to the right of robot center; tune this setpoint during testing
        self.camera_offset_setpoint = -0.4

        # --- Subscribers ---
        self.create_subscription(String, '/vision/gesture_labels', self.gesture_callback, 10)
        self.create_subscription(Float32, '/vision/line_error', self.line_error_callback, 10)
        self.create_subscription(Bool, '/vision/line_detected', self.line_detected_callback, 10)

        # --- Publishers ---
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # --- State ---
        self.navigation_paused = False
        self.line_error = 0.0
        self.line_detected = False

        # 10 Hz control loop
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("MissionController initialized.")

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

    def publish_twist(self, linear_x=0.0, angular_z=0.0):
        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(twist)

    def publish_stop(self):
        self.publish_twist(0.0, 0.0)

    def control_loop(self):
        if self.navigation_paused:
            self.publish_stop()
            return

        if not self.line_detected:
            self.publish_stop()
            self.get_logger().info("No line detected: stopping.", throttle_duration_sec=2.0)
            return

        error = self.line_error - self.camera_offset_setpoint
        angular_z = -self.kp * error
        self.publish_twist(self.line_drive_speed, angular_z)


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