import math
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


def normalize_angle_deg(angle_deg):
    while angle_deg > 180.0:
        angle_deg -= 360.0
    while angle_deg < -180.0:
        angle_deg += 360.0
    return angle_deg


class ScanGrabber(Node):
    def __init__(self):
        super().__init__('lidar_plot_grabber')
        self.latest_scan = None
        self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data
        )

    def scan_callback(self, msg):
        self.latest_scan = msg


def get_single_scan(timeout_sec=5.0):
    rclpy.init()
    node = ScanGrabber()
    end_time = node.get_clock().now().nanoseconds / 1e9 + timeout_sec
    try:
        while rclpy.ok() and node.latest_scan is None:
            if node.get_clock().now().nanoseconds / 1e9 > end_time:
                break
            rclpy.spin_once(node, timeout_sec=0.1)
        return node.latest_scan
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main():
    scan = get_single_scan(timeout_sec=5.0)
    if scan is None or not scan.ranges:
        print("No /scan message received.")
        return

    angles_deg = []
    distances_m = []
    angles_rad_valid = []
    distances_valid = []

    for i, r in enumerate(scan.ranges):
        angle_rad = scan.angle_min + i * scan.angle_increment
        angle_deg = normalize_angle_deg(math.degrees(angle_rad))
        angles_deg.append(angle_deg)

        if math.isnan(r) or math.isinf(r) or r <= scan.range_min or r >= scan.range_max:
            distances_m.append(float('nan'))
        else:
            distances_m.append(r)
            angles_rad_valid.append(angle_rad)
            distances_valid.append(r)

    # Sort by angle (-180 deg to +180 deg) for a clean line plot
    paired = sorted(zip(angles_deg, distances_m), key=lambda x: x[0])
    sorted_angles = [p[0] for p in paired]
    sorted_dists = [p[1] for p in paired]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # 1. Angle (deg) vs. Distance (m)
    ax1.plot(sorted_angles, sorted_dists, marker='.', markersize=2, linestyle='-')
    ax1.axvline(0.0, color='g', linestyle='--', linewidth=1, label='0 deg (Front)')
    ax1.axvline(90.0, color='b', linestyle=':', linewidth=1, label='+90 deg (Left)')
    ax1.axvline(-90.0, color='r', linestyle=':', linewidth=1, label='-90 deg (Right)')
    ax1.set_title('LiDAR Angle vs. Distance')
    ax1.set_xlabel('Angle (degrees)')
    ax1.set_ylabel('Distance (m)')
    ax1.set_xlim(-180, 180)
    ax1.grid(True)
    ax1.legend()

    # 2. Top-down polar view (0 rad at top / front)
    ax2 = plt.subplot(122, projection='polar')
    ax2.set_theta_zero_location('N')  # 0 deg at top (front of robot)
    ax2.set_theta_direction(1)        # Counter-clockwise (+angle = left)
    ax2.scatter(angles_rad_valid, distances_valid, s=6)
    ax2.set_title('Top-Down Scan (0 deg Top = Front)')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()