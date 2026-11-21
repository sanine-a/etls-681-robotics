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

        self.declare_parameter('publish_debug_image', True)
        self.debug_image_pub = self.create_publisher(Image, '/vision/line_debug_image', 10)

        self.get_logger().info("LineFollower node initialized.")

    def image_callback(self, msg):
        frame = np.frombuffer(msg.data, dtype=np.uint8)
        frame = frame.reshape(msg.height, msg.step)
        frame = frame[:, :msg.width * 3]
        frame = frame.reshape(msg.height, msg.width, 3)

        if frame is None:
            self.get_logger().warn("Failed to decode image.")
            return

        height, width, _ = frame.shape

        # Focus on the bottom strip of the image where the line is close to the robot
        roi_height_start = int(height * 0.9) # Bottom 90% of the image
        roi = frame[roi_height_start:height, 0:width]

        # Convert to grayscale and threshold to find the dark line
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

        # Find contours of potential lines
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        line_detected_msg = Bool()
        error_msg = Float32()
        debug_frame = frame.copy()
        cx = None
        area = 0.0

        if contours:
            # Find the largest contour in the region of interest
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            if area > 200:  # Minimum area threshold to ignore small noise
                M = cv2.moments(largest_contour)
                if M["m00"] > 0: # Avoid division by zero by checking if the area is greater than zero
                    cx = int(M["m10"] / M["m00"]) # x-centroid calculated by taking first-order horixaontal moment and dividing it by the total area

                    # Calculate error: distance from frame center (-1.0 to 1.0 normalized)
                    frame_center_x = width / 2.0
                    error = (cx - frame_center_x) / frame_center_x

                    error_msg.data = float(error)
                    line_detected_msg.data = True

        if cx is None:
            line_detected_msg.data = False
            error_msg.data = 0.0

        # Draw debug information on the original frame.
        cv2.rectangle(debug_frame, (0, roi_height_start), (width - 1, height - 1), (255, 0, 255), 1)
        cv2.line(debug_frame, (int(width / 2), roi_height_start), (int(width / 2), height - 1), (255, 255, 0), 1)

        if line_detected_msg.data:
            contour = largest_contour[:, 0, :].copy()
            contour[:, 1] += roi_height_start
            cv2.drawContours(debug_frame, [contour], -1, (0, 255, 0), 2)
            cv2.line(debug_frame, (cx, roi_height_start), (cx, height - 1), (0, 255, 0), 2)
            cv2.circle(debug_frame, (cx, height - 10), 6, (0, 0, 255), -1)

        status = "DETECTED" if line_detected_msg.data else "NOT DETECTED"
        cv2.putText(debug_frame, f"Line: {status}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(debug_frame, f"Error: {error_msg.data:.3f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(debug_frame, f"Centroid X: {cx if cx is not None else 'N/A'}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(debug_frame, f"Contour area: {area:.1f}", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        self.error_pub.publish(error_msg)
        self.line_detected_pub.publish(line_detected_msg)

        if self.get_parameter('publish_debug_image').value:
            debug_msg = Image()
            debug_msg.height = debug_frame.shape[0]
            debug_msg.width = debug_frame.shape[1]
            debug_msg.encoding = 'bgr8'
            debug_msg.is_bigendian = msg.is_bigendian
            debug_msg.step = debug_frame.shape[1] * 3
            debug_msg.data = debug_frame.flatten().tolist()
            debug_msg.header = msg.header
            self.debug_image_pub.publish(debug_msg)


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