import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float32, Bool


class MissionController(Node):
    def __init__(self):
        super().__init__('mission_controller')

        # --- Line Following Parameters ---
        self.kp = 0.8
        self.line_drive_speed = 0.12
        self.camera_offset_setpoint = -0.4
        self.line_align_kp = 0.45
        self.line_lost_trigger_count = 3

        # --- Backup & Creep Parameters ---
        self.backup_speed = -0.08
        self.backup_duration_sec = 0.6
        self.post_drop_retreat_duration_sec = 1.0

        self.creep_speed = 0.08
        self.creep_duration_sec = 2.0

        # --- Object Inspection Window (after backing up) ---
        self.inspect_duration_sec = 1.0
        self.bottle_votes_needed = 3

        # --- Turn Parameters (180 deg = pi rad) ---
        self.turn_speed = 0.55  # Keep gentle when carrying a bottle
        self.turn_180_duration_sec = math.pi / self.turn_speed
        self.min_turn_before_line_align_sec = self.turn_180_duration_sec * 0.75

        # --- Gesture Debounce ---
        self.gesture_trigger_count = 3

        # --- Subscribers ---
        self.create_subscription(String, '/vision/gesture_labels', self.gesture_callback, 10)
        self.create_subscription(Float32, '/vision/line_error', self.line_error_callback, 10)
        self.create_subscription(Bool, '/vision/line_detected', self.line_detected_callback, 10)
        self.create_subscription(String, '/vision/object_labels', self.object_callback, 10)
        self.create_subscription(String, '/arm/status', self.arm_status_callback, 10)

        # --- Publishers ---
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.arm_cmd_pub = self.create_publisher(String, '/arm/command', 10)

        # --- Latest Topic State ---
        self.current_gesture = "None"
        self.current_objects = []
        self.line_error = 0.0
        self.line_detected = False
        self.line_lost_count = 0
        self.arm_status = "IDLE"

        self.navigation_paused = False
        self.open_hand_count = 0
        self.thumbs_up_count = 0
        self.bottle_inspect_votes = 0

        # --- State Machine ---
        # FOLLOW_LINE -> BACKING_UP -> INSPECT_OBJECT ->
        #   (no bottle) STOPPED_FOR_OBJECT
        #   (bottle)    CREEP_FORWARD -> GRABBING -> TURNING_TO_PLACE -> RELEASING -> POST_DROP_RETREAT -> TURNING_BACK -> FOLLOW_LINE
        self.state = "FOLLOW_LINE"
        self.state_start_time = self.now_sec()
        self.arm_command_sent = False

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("MissionController initialized.")

    def now_sec(self):
        return self.get_clock().now().nanoseconds / 1e9

    def gesture_callback(self, msg):
        self.current_gesture = msg.data.strip()

    def line_error_callback(self, msg):
        self.line_error = float(msg.data)

    def line_detected_callback(self, msg):
        self.line_detected = bool(msg.data)

    def object_callback(self, msg):
        if msg.data.strip():
            self.current_objects = [item.strip() for item in msg.data.split(",") if item.strip()]
        else:
            self.current_objects = []

    def arm_status_callback(self, msg):
        self.arm_status = msg.data.strip()

    def publish_twist(self, linear_x=0.0, angular_z=0.0):
        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(twist)

    def publish_stop(self):
        self.publish_twist(0.0, 0.0)

    def send_arm_command(self, cmd_text):
        msg = String()
        msg.data = cmd_text
        self.arm_cmd_pub.publish(msg)
        self.get_logger().info(f"Sent /arm/command: '{cmd_text}'")

    def transition_to(self, new_state):
        self.state = new_state
        self.state_start_time = self.now_sec()
        self.arm_command_sent = False
        if new_state == "FOLLOW_LINE":
            self.line_lost_count = 0
        if new_state == "INSPECT_OBJECT":
            self.bottle_inspect_votes = 0
        self.get_logger().info(f"State -> {new_state}")

    def handle_gestures(self):
        if self.current_gesture == "open_hand":
            self.open_hand_count = min(10, self.open_hand_count + 1)
            self.thumbs_up_count = 0
        elif self.current_gesture == "thumbs_up":
            self.thumbs_up_count = min(10, self.thumbs_up_count + 1)
            self.open_hand_count = 0
        else:
            self.open_hand_count = max(0, self.open_hand_count - 1)
            self.thumbs_up_count = max(0, self.thumbs_up_count - 1)

        if self.open_hand_count >= self.gesture_trigger_count and not self.navigation_paused:
            self.navigation_paused = True
            self.open_hand_count = 0
            self.publish_stop()
            self.get_logger().info("Open hand detected: paused.")
        elif self.thumbs_up_count >= self.gesture_trigger_count and self.navigation_paused:
            self.navigation_paused = False
            self.thumbs_up_count = 0
            self.get_logger().info("Thumbs up detected: resumed.")

    def follow_line_step(self):
        error = self.line_error - self.camera_offset_setpoint
        angular_z = -self.kp * error
        self.publish_twist(self.line_drive_speed, angular_z)

    def control_loop(self):
        self.handle_gestures()

        if self.navigation_paused:
            self.publish_stop()
            return

        elapsed = self.now_sec() - self.state_start_time

        if self.state == "FOLLOW_LINE":
            if not self.line_detected:
                self.publish_stop()
                self.line_lost_count += 1
                if self.line_lost_count >= self.line_lost_trigger_count:
                    self.transition_to("BACKING_UP")
                return
            self.line_lost_count = 0
            self.follow_line_step()

        elif self.state == "BACKING_UP":
            if elapsed < self.backup_duration_sec:
                self.publish_twist(self.backup_speed, 0.0)
            else:
                self.publish_stop()
                self.transition_to("INSPECT_OBJECT")

        elif self.state == "INSPECT_OBJECT":
            self.publish_stop()
            if "bottle" in self.current_objects:
                self.bottle_inspect_votes = min(self.bottle_votes_needed, self.bottle_inspect_votes + 1)
            else:
                self.bottle_inspect_votes = max(0, self.bottle_inspect_votes - 1)

            if elapsed >= self.inspect_duration_sec:
                if self.bottle_inspect_votes >= self.bottle_votes_needed:
                    self.get_logger().info("Bottle confirmed after backup.")
                    self.transition_to("CREEP_FORWARD")
                else:
                    self.get_logger().info("No bottle detected after backup: stopping for obstacle.")
                    self.transition_to("STOPPED_FOR_OBJECT")

        elif self.state == "STOPPED_FOR_OBJECT":
            self.publish_stop()

        elif self.state == "CREEP_FORWARD":
            if elapsed < self.creep_duration_sec:
                self.publish_twist(self.creep_speed, 0.0)
            else:
                self.publish_stop()
                self.transition_to("GRABBING")

        elif self.state == "GRABBING":
            self.publish_stop()
            if not self.arm_command_sent:
                self.send_arm_command("grab")
                self.arm_command_sent = True
            elif self.arm_status == "GRAB_COMPLETE":
                self.transition_to("TURNING_TO_PLACE")

        elif self.state == "TURNING_TO_PLACE":
            # Use the visible line to finish aligning after most of the turn.
            if elapsed >= self.min_turn_before_line_align_sec and self.line_detected:
                error = self.line_error - self.camera_offset_setpoint
                if abs(error) < 0.08:
                    self.publish_stop()
                    self.transition_to("RELEASING")
                    return
                align_angular_z = max(-0.45, min(0.45, -self.line_align_kp * error))
                self.publish_twist(0.0, align_angular_z)
                return

            if elapsed < self.turn_180_duration_sec:
                self.publish_twist(0.0, self.turn_speed)
            else:
                self.publish_stop()
                self.transition_to("RELEASING")

        elif self.state == "RELEASING":
            self.publish_stop()
            if not self.arm_command_sent:
                self.send_arm_command("release")
                self.arm_command_sent = True
            elif self.arm_status == "RELEASE_COMPLETE":
                self.transition_to("POST_DROP_RETREAT")

        elif self.state == "POST_DROP_RETREAT":
            if elapsed < self.post_drop_retreat_duration_sec:
                self.publish_twist(self.backup_speed, 0.0)
            else:
                self.publish_stop()
                self.transition_to("TURNING_BACK")

        elif self.state == "TURNING_BACK":
            # Once past 75% of the turn window and the line is visible, use line_error to finish aligning
            if elapsed >= self.min_turn_before_line_align_sec and self.line_detected:
                error = self.line_error - self.camera_offset_setpoint
                if abs(error) < 0.08:
                    self.publish_stop()
                    self.transition_to("FOLLOW_LINE")
                    return
                align_angular_z = max(-0.45, min(0.45, -self.line_align_kp * error))
                self.publish_twist(0.0, align_angular_z)
                return

            if elapsed < self.turn_180_duration_sec:
                self.publish_twist(0.0, -self.turn_speed)
            else:
                self.publish_stop()
                self.transition_to("FOLLOW_LINE")


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