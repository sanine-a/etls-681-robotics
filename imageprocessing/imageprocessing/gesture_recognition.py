import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import mediapipe as mp

from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
import math

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class GestureRecognizer(Node):
    def __init__(self):
        super().__init__('gesture_recognizer')

        # Initialize MediaPipe HandLandmarker
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=1
        )
        self.detector = HandLandmarker.create_from_options(options)

        # Subscribe to TurtleBot3 camera
        self.image_sub = self.create_subscription(
            # CompressedImage,
            Image,
            # '/camera/image_raw/compressed',
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Publishers
        self.gesture_pub = self.create_publisher(String, '/vision/gesture_labels', 10)
        self.annotated_pub = self.create_publisher(Image, '/vision/gesture_annotated', 10)

        self.get_logger().info("GestureRecognizer node initialized.")

    def classify_gesture(self, landmarks):
        wrist = landmarks[0]
        thumb_mcp = landmarks[2]
        thumb_tip = landmarks[4]
        
        fingers = [
            (8, 6),   # Index: tip, pip
            (12, 10), # Middle: tip, pip
            (16, 14), # Ring:   tip, pip
            (20, 18)  # Pinky: tip, pip
        ]

        # Thumbs up logic:
        # 1. Thumb is pointing up relative to its knuckle and wrist
        # 2. Other fingers are curled: tip X is closer to the wrist X than the PIP X
        thumb_is_up = thumb_tip.y < thumb_mcp.y and thumb_tip.y < wrist.y
        others_are_curled = all(
            abs(landmarks[tip].x - wrist.x) < abs(landmarks[pip].x - wrist.x) 
            for tip, pip in fingers
        )

        if thumb_is_up and others_are_curled:
            return "thumbs_up"

    # Open hand logic:
    # All four fingers must have their tips higher than the wrist AND higher than their PIP joint
        open_hand_valid = all(
            landmarks[tip].y < wrist.y and landmarks[tip].y < landmarks[pip].y 
            for tip, pip in fingers
        )

        if open_hand_valid:
            return "open_hand"

        return None

    def image_callback(self, msg):
        # Decode image
        frame = np.array(msg.data, dtype=np.uint8).reshape(
			msg.height, math.floor(msg.step/3), 3
		)

        if frame is None:
            self.get_logger().warn("Failed to decode image.")
            return

        # Convert to MediaPipe Image format
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        result = self.detector.detect(mp_image)
        gesture_label = "None"

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            detected_gesture = self.classify_gesture(landmarks)

            if detected_gesture:
                gesture_label = detected_gesture

            # Draw landmarks and label on the frame
            for lm in landmarks:
                cx = int(lm.x * frame.shape[1])
                cy = int(lm.y * frame.shape[0])
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

            if gesture_label != "None":
                cv2.putText(frame, gesture_label, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Publish gesture label
        label_msg = String()
        label_msg.data = gesture_label
        self.gesture_pub.publish(label_msg)

        # Publish annotated image
        annotated_msg = Image()
        annotated_msg.height = frame.shape[0]
        annotated_msg.width = frame.shape[1]
        annotated_msg.encoding = "bgr8"
        annotated_msg.step = frame.shape[1] * 3
        annotated_msg.data = frame.flatten().tolist()
        self.annotated_pub.publish(annotated_msg)


def main(args=None):
    rclpy.init(args=args)
    node = GestureRecognizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()