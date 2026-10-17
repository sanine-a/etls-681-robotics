import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String


class TaskController(Node):
    def __init__(self):
        super().__init__('task_controller')

        # Subscribers
        self.gesture_sub = self.create_subscription(
            String, '/vision/gesture_labels', self.gesture_callback, 10
        )
        self.object_sub = self.create_subscription(
            String, '/vision/object_labels', self.object_callback, 10
        )

        # Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # State tracking variables
        self.current_gesture = "None"
        self.current_objects = []

        # Motion behavior states
        self.is_driving_forward = False
        self.is_spinning_360 = False
        self.is_running_away = False

        # 360 Spin tracking variables (~6.3 seconds for 360° at 1.0 rad/s)
        self.spin_start_time = None
        self.spin_duration = 6.3 

        # --- Noise Filtering (Debouncing) Counters ---
        self.debounce_threshold = 5  # Must see condition for 5 consecutive loops
        
        self.cellphone_counter = 0
        self.bottle_counter = 0
        self.thumbs_up_counter = 0
        self.open_hand_counter = 0

        # Control loop timer (10 Hz)
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("TaskController node initialized with priority adjustments.")

    def gesture_callback(self, msg):
        self.current_gesture = msg.data

    def object_callback(self, msg):
        if msg.data:
            self.current_objects = msg.data.split(",")
        else:
            self.current_objects = []

    def control_loop(self):
        twist = Twist()
        current_time = self.get_clock().now().nanoseconds / 1e9

        # ==========================================
        # 1. TOP PRIORITY: 360 Spin Behavior
        # ==========================================
        if self.is_spinning_360:
            elapsed_time = current_time - self.spin_start_time
            if elapsed_time < self.spin_duration:
                twist.angular.z = 1.0
                twist.linear.x = 0.0
                self.cmd_vel_pub.publish(twist)
                return
            else:
                self.get_logger().info("360 spin completed. Resuming normal operation.")
                self.is_spinning_360 = False

        # ==========================================
        # 2. UPDATE NOISE FILTERING COUNTERS
        # ==========================================
        cellphone_present = "cell phone" in self.current_objects
        bottle_present = "bottle" in self.current_objects

        if cellphone_present:
            self.cellphone_counter += 1
        else:
            self.cellphone_counter = max(0, self.cellphone_counter - 1)

        if bottle_present:
            self.bottle_counter += 1
        else:
            self.bottle_counter = max(0, self.bottle_counter - 1)

        if self.current_gesture == "thumbs_up":
            self.thumbs_up_counter += 1
        else:
            self.thumbs_up_counter = max(0, self.thumbs_up_counter - 1)

        if self.current_gesture == "open_hand":
            self.open_hand_counter += 1
        else:
            self.open_hand_counter = max(0, self.open_hand_counter - 1)

        # ==========================================
        # 3. CHECK CELLPHONE INTERRUPT TRIGGER
        # ==========================================
        if self.cellphone_counter >= self.debounce_threshold and not self.is_spinning_360:
            self.get_logger().info("Cell phone detected consistently! Initiating 360 spin.")
            self.is_spinning_360 = True
            self.spin_start_time = current_time
            self.cellphone_counter = 0
            self.is_driving_forward = False
            self.is_running_away = False
            twist.angular.z = 1.0
            self.cmd_vel_pub.publish(twist)
            return

        # ==========================================
        # 4. GESTURE TRIGGER EVALUATION
        # ==========================================
        if self.thumbs_up_counter >= self.debounce_threshold:
            if not self.is_driving_forward:
                self.get_logger().info("Thumbs Up confirmed: Latching forward drive (Overrides bottle retreat).")
            self.is_driving_forward = True
            self.is_running_away = False  # Thumbs up cancels running away

        if self.open_hand_counter >= self.debounce_threshold:
            if self.is_driving_forward:
                self.get_logger().info("Open Hand confirmed: Stopping robot.")
            self.is_driving_forward = False

        # ==========================================
        # 5. bottle RETREAT TRIGGER (Only if NOT driving forward)
        # ==========================================
        if not self.is_driving_forward:
            if self.bottle_counter >= self.debounce_threshold:
                self.is_running_away = True

        # ==========================================
        # 6. EXECUTE BEHAVIORS BASED ON PRIORITY FLAGS
        # ==========================================
        if self.is_driving_forward:
            # Thumbs up has higher priority than bottle view
            twist.linear.x = 0.15
            twist.angular.z = 0.0
            self.get_logger().info("Driving forward via Thumbs Up...", throttle_duration_sec=2.0)

        elif self.is_running_away:
            # Robot is stopped/idle and sees a bottle -> run away until it disappears
            if bottle_present:
                twist.linear.x = -0.35  # Reverse fast
                twist.angular.z = 0.0
                self.get_logger().info("bottle visible: Running away fast!", throttle_duration_sec=1.0)
            else:
                self.get_logger().info("bottle lost from view. Ending run-away state.")
                self.is_running_away = False
                self.bottle_counter = 0
                twist.linear.x = 0.0
                twist.angular.z = 0.0

        else:
            # Default state: Stopped
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        self.cmd_vel_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = TaskController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()