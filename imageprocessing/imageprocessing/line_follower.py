import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import math
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Bool


class LineFollower(Node):
    def __init__(self):
        super().__init__('line_follower')

        # Subscribe to camera feed
        self.image_sub = self.create_subscription(
            Image,
            '/camera_relay/image_raw',
            self.image_callback,
            10
        )

        # Publishers
        self.error_pub = self.create_publisher(Float32, '/vision/line_error', 10)
        self.line_detected_pub = self.create_publisher(Bool, '/vision/line_detected', 10)

        self.get_logger().info("LineFollower node initialized.")

    def image_callback(self, msg):
        frame = np.array(msg.data, dtype=np.uint8).reshape(
            msg.height, math.floor(msg.step / 3), 3
        )

        if frame is None:
            return

        height, width, _ = frame.shape

        # Focus on the bottom strip of the image where the line is close to the robot
        roi_height_start = int(height * 0.7)
        roi = frame[roi_height_start:height, 0:width]

        # Convert to grayscale and threshold to find the dark line
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

        # Find contours of potential lines
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        line_detected_msg = Bool()
        error_msg = Float32()

        if contours:
            # Find the largest contour in the region of interest
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            if area > 100:  # Minimum area threshold to ignore small noise
                M = cv2.moments(largest_contour)
                if M["m00"] > 0: # Avoid division by zero by checking if the area is greater than zero
                    cx = int(M["m10"] / M["m00"]) # x-centroid calculated by taking first-order horixaontal moment and dividing it by the total area
                    
                    # Calculate error: distance from frame center (-1.0 to 1.0 normalized)
                    frame_center_x = width / 2.0
                    error = (cx - frame_center_x) / frame_center_x
                    
                    error_msg.data = float(error)
                    line_detected_msg.data = True
                    
                    self.error_pub.publish(error_msg)
                    self.line_detected_pub.publish(line_detected_msg)
                    return

        # If no line is found
        line_detected_msg.data = False
        error_msg.data = 0.0
        self.error_pub.publish(error_msg)
        self.line_detected_pub.publish(line_detected_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LineFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()