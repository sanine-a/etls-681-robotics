import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from builtin_interfaces.msg import Duration as MsgDuration
from control_msgs.action import FollowJointTrajectory, GripperCommand
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectoryPoint


class BottleArmNode(Node):
    def __init__(self):
        super().__init__('bottle_arm_node')

        self.arm_joints = ['joint1', 'joint2', 'joint3', 'joint4']

        # --- Tunable Joint Postures [joint1, joint2, joint3, joint4] (radians) ---
        # Tune these gently while watching the robot on your first test!
        self.home_positions = [-0.0031, -1.0523, 1.0769, 0.0061]
        self.reach_forward_positions = [-0.0077, 1.129, -0.7363, -0.1718]
        self.lift_positions = [0.0031, 0.8173, -0.7424, 0.0046]

        # --- Tunable Gripper Positions (meters) ---
        self.gripper_open_pos = 0.019
        self.gripper_closed_pos = -0.005
        self.gripper_max_effort = 1.0

        self.move_time_sec = 1.5

        # --- Action Clients ---
        self.arm_client = ActionClient(
            self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory'
        )
        self.gripper_client = ActionClient(
            self, GripperCommand, '/gripper_controller/gripper_cmd'
        )

        # --- Command & Status ---
        self.create_subscription(String, '/arm/command', self.command_callback, 10)
        self.status_pub = self.create_publisher(String, '/arm/status', 10)

        self.active_sequence = None
        self.step = 0
        self.step_start_time = None

        self.timer = self.create_timer(0.1, self.control_loop)
        self.publish_status("IDLE")
        self.get_logger().info("BottleArmNode ready. Send 'grab' or 'release' to /arm/command.")

    def now_sec(self):
        return self.get_clock().now().nanoseconds / 1e9

    def publish_status(self, text):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

    def command_callback(self, msg):
        cmd = msg.data.strip().lower()
        if cmd == "grab":
            self.active_sequence = "GRAB"
            self.step = 0
            self.step_start_time = self.now_sec()
            self.publish_status("GRABBING")
            self.get_logger().info("Starting GRAB sequence.")
        elif cmd == "release":
            self.active_sequence = "RELEASE"
            self.step = 0
            self.step_start_time = self.now_sec()
            self.publish_status("RELEASING")
            self.get_logger().info("Starting RELEASE sequence.")
        else:
            self.get_logger().warn(f"Unknown /arm/command: '{msg.data}'")

    def send_arm_goal(self, positions, duration_sec=1.5):
        if not self.arm_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn("/arm_controller/follow_joint_trajectory action server unavailable.")
            return

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.arm_joints

        pt = JointTrajectoryPoint()
        pt.positions = [float(p) for p in positions]
        sec = int(duration_sec)
        nanosec = int((duration_sec - sec) * 1e9)
        pt.time_from_start = MsgDuration(sec=sec, nanosec=nanosec)

        goal.trajectory.points = [pt]
        self.arm_client.send_goal_async(goal)

    def send_gripper_goal(self, position):
        if not self.gripper_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn("/gripper_controller/gripper_cmd action server unavailable.")
            return

        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = float(self.gripper_max_effort)
        self.gripper_client.send_goal_async(goal)

    def control_loop(self):
        if self.active_sequence is None:
            return

        elapsed = self.now_sec() - self.step_start_time

        if self.active_sequence == "GRAB":
            if self.step == 0:
                # 1. Open gripper and reach straight forward
                self.send_gripper_goal(self.gripper_open_pos)
                self.send_arm_goal(self.reach_forward_positions, self.move_time_sec)
                self.step = 1
                self.step_start_time = self.now_sec()
            elif self.step == 1 and elapsed >= self.move_time_sec + 0.3:
                # 2. Close gripper on bottle
                self.send_gripper_goal(self.gripper_closed_pos)
                self.step = 2
                self.step_start_time = self.now_sec()
            elif self.step == 2 and elapsed >= 1.0:
                # 3. Lift up slightly
                self.send_arm_goal(self.lift_positions, self.move_time_sec)
                self.step = 3
                self.step_start_time = self.now_sec()
            elif self.step == 3 and elapsed >= self.move_time_sec + 0.3:
                self.active_sequence = None
                self.publish_status("GRAB_COMPLETE")
                self.get_logger().info("GRAB sequence complete.")

        elif self.active_sequence == "RELEASE":
            if self.step == 0:
                # 1. Lower back down to placement posture
                self.send_arm_goal(self.reach_forward_positions, self.move_time_sec)
                self.step = 1
                self.step_start_time = self.now_sec()
            elif self.step == 1 and elapsed >= self.move_time_sec + 0.3:
                # 2. Open gripper to release bottle
                self.send_gripper_goal(self.gripper_open_pos)
                self.step = 2
                self.step_start_time = self.now_sec()
            elif self.step == 2 and elapsed >= 1.0:
                # 3. Return arm to home/travel posture
                self.send_arm_goal(self.home_positions, self.move_time_sec)
                self.step = 3
                self.step_start_time = self.now_sec()
            elif self.step == 3 and elapsed >= self.move_time_sec + 0.3:
                self.active_sequence = None
                self.publish_status("RELEASE_COMPLETE")
                self.get_logger().info("RELEASE sequence complete.")


def main(args=None):
    rclpy.init(args=args)
    node = BottleArmNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()