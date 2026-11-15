import rclpy
from rclpy.node import Node
from std_msgs.msg import String
# Note: Import your specific arm/gripper control messages here depending on your hardware setup


class ArmController(Node):
    def __init__(self):
        super().__init__('arm_controller')

        # Subscribe to arm command triggers from the mission controller
        self.arm_sub = self.create_subscription(
            String, '/arm/command', self.arm_callback, 10
        )

        self.get_logger().info("ArmController node initialized.")

    def arm_callback(self, msg):
        command = msg.data
        if command == "grab":
            self.get_logger().info("Executing GRAB sequence...")
            # TODO: Add publisher commands to servos/joints to grab the bottle
        elif command == "release":
            self.get_logger().info("Executing RELEASE sequence...")
            # TODO: Add publisher commands to servos/joints to release the bottle
        else:
            self.get_logger().warn(f"Unknown arm command received: {command}")


def main(args=None):
    rclpy.init(args=args)
    node = ArmController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()