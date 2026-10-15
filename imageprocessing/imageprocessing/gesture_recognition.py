#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import cv2
import numpy as np

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

import mediapipe as mp

# MediaPipe Tasks setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class GestureRecognitionNode(Node):
    def __init__(self):
        super().__init__('gesture_recognition')

        self.bridge = CvBridge()

        # Subscribers
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Publishers
        self.label_pub = self.create_publisher(String, '/vision/gesture_labels', 10)
        self.annotated_pub = self.create_publisher(Image, '/vision/gesture_annotated', 10)

        # Load MediaPipe model
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=1
        )
        self.detector = HandLandmarker.create_from_options(options)

        self.get_logger().info("GestureRecognitionNode initialized.")

    # -----------------------------
    # Gesture Classification Logic
    # -----------------------------
    def classify_gesture(self, landmarks):
        wrist = landmarks[0]

        # Thumb joints
        thumb_mcp = landmarks[2]
        thumb_tip = landmarks[4]

        # Other finger tips and PIP joints
        fingers = [
            (8, 6),   # Index
            (12, 10), # Middle
            (16, 14), # Ring
            (20, 18)  # Pinky
        ]

        # Thumbs up logic
        thumb_is_up = thumb_tip.y < thumb_mcp.y and thumb_tip.y < wrist.y
        others_are_curled = all(
            abs(landmarks[tip].x - wrist.x) < abs(landmarks[pip].x - wrist.x)
            for tip, pip in fingers
        )

        if thumb_is_up and others_are_curled:
            return "thumbs_up"

        # Open hand logic
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]

        if (
            index_tip.y < wrist.y and
            middle_tip.y < wrist.y and
            ring_tip.y < wrist.y and
            pinky_tip.y < wrist.y
        ):
            return "open_hand"

        return None

    # -----------------------------
    # Image Callback
    # -----------------------------
    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Optional preprocessing
        frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=20)
        frame = cv2.GaussianBlur(frame, (3, 3), 0)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.detector.detect(mp_image)

        gesture_label = None

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            gesture_label = self.classify_gesture(landmarks)

            # Draw landmarks
            for lm in landmarks:
                cx = int(lm.x * frame.shape[1])
                cy = int(lm.y * frame.shape[0])
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

            if gesture_label:
                cv2.putText(frame, gesture_label, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Publish gesture label
        label_msg = String()
        label_msg.data = gesture_label if gesture_label else ""
        self.label_pub.publish(label_msg)

        # Publish annotated image
        annotated_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.annotated_pub.publish(annotated_msg)


def main(args=None):
    rclpy.init(args=args)
    node = GestureRecognitionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
