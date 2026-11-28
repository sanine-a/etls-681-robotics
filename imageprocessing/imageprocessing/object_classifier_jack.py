import math
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
print("importing yolo...")
from ultralytics import YOLO
print("yolo imported")

from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32, Bool

TARGET_OBJECTS = ["bottle", "person"]
MIN_CONFIDENCE = 0.2


class ObjectClassifierJack(Node):
    def __init__(self):
        super().__init__('object_classifier_jack')

        # YOLO model
        self.get_logger().info("loading yolov8n model...")
        print("loading yolov8n model")
        self.model = YOLO("yolov8n.pt")
        self.get_logger().info("yolov8n model loaded.")
        print("yolov8n model loaded")

        # Subscribe to camera feed
        self.image_sub = self.create_subscription(
            Image,
            '/camera_relay/image_raw',
            self.image_callback,
            10
        )

        # Publishers (plain Bool / Float32 topics matching line_follower.py)
        self.bottle_detected_pub = self.create_publisher(Bool, '/vision/bottle_detected', 10)
        self.bottle_error_pub = self.create_publisher(Float32, '/vision/bottle_error', 10)
        self.bottle_bottom_pub = self.create_publisher(Float32, '/vision/bottle_bottom_norm', 10)

        self.person_detected_pub = self.create_publisher(Bool, '/vision/person_detected', 10)
        self.person_error_pub = self.create_publisher(Float32, '/vision/person_error', 10)

        self.label_pub = self.create_publisher(String, '/vision/object_labels', 10)
        self.annotated_pub = self.create_publisher(Image, '/vision/object_annotated', 10)

        self.get_logger().info("ObjectClassifierJack node initialized.")

    # ---------------------------------------------------------
    # Decode image -> run YOLO -> publish results
    # ---------------------------------------------------------
    def image_callback(self, msg):
        frame = np.frombuffer(msg.data, dtype=np.uint8)
        frame = frame.reshape(msg.height, msg.step)
        frame = frame[:, :msg.width * 3]
        frame = frame.reshape(msg.height, msg.width, 3).copy()

        if frame is None or frame.size == 0:
            self.get_logger().warn("Failed to decode image.")
            return

        height, width, _ = frame.shape
        frame_center_x = width / 2.0

        # Run YOLO inference
        results = self.model(frame, verbose=False)
        detected_labels = []
        bottles = []
        people = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = self.model.names[cls]
                conf = float(box.conf[0])

                if label in TARGET_OBJECTS and conf > MIN_CONFIDENCE:
                    detected_labels.append(label)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    det = {"conf": conf, "box": (x1, y1, x2, y2)}

                    if label == "bottle":
                        bottles.append(det)
                        color = (0, 255, 0)
                    else:
                        people.append(det)
                        color = (255, 180, 0)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"{label} {conf:.2f}",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )

        # --- Leftmost Water Bottle ---
        bottle_detected_msg = Bool()
        bottle_error_msg = Float32()
        bottle_bottom_msg = Float32()

        if bottles:
            leftmost = min(bottles, key=lambda b: (b["box"][0] + b["box"][2]) / 2.0)
            bx1, by1, bx2, by2 = leftmost["box"]
            b_cx = (bx1 + bx2) / 2.0
            b_cy = (by1 + by2) / 2.0

            bottle_detected_msg.data = True
            bottle_error_msg.data = float((b_cx - frame_center_x) / frame_center_x)
            bottle_bottom_msg.data = float(max(0.0, min(1.0, by2 / float(height))))

            cv2.circle(frame, (int(b_cx), int(b_cy)), 6, (0, 0, 255), -1)
            cv2.line(frame, (0, by2), (width, by2), (0, 0, 255), 1)
            cv2.putText(
                frame,
                f"Bottle err={bottle_error_msg.data:.2f} bot={bottle_bottom_msg.data:.2f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )
        else:
            bottle_detected_msg.data = False
            bottle_error_msg.data = 0.0
            bottle_bottom_msg.data = 0.0

        # --- Closest Person (largest bounding box) ---
        person_detected_msg = Bool()
        person_error_msg = Float32()

        if people:
            closest_person = max(
                people,
                key=lambda p: max(0, p["box"][2] - p["box"][0]) * max(0, p["box"][3] - p["box"][1])
            )
            px1, py1, px2, py2 = closest_person["box"]
            p_cx = (px1 + px2) / 2.0
            p_cy = (py1 + py2) / 2.0

            person_detected_msg.data = True
            person_error_msg.data = float((p_cx - frame_center_x) / frame_center_x)

            cv2.circle(frame, (int(p_cx), int(p_cy)), 6, (255, 0, 255), -1)
            cv2.putText(
                frame,
                f"Person err={person_error_msg.data:.2f}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2
            )
        else:
            person_detected_msg.data = False
            person_error_msg.data = 0.0

        # Publish detection signals
        self.bottle_detected_pub.publish(bottle_detected_msg)
        self.bottle_error_pub.publish(bottle_error_msg)
        self.bottle_bottom_pub.publish(bottle_bottom_msg)
        self.person_detected_pub.publish(person_detected_msg)
        self.person_error_pub.publish(person_error_msg)

        # Publish object labels
        label_msg = String()
        label_msg.data = ",".join(detected_labels)
        self.label_pub.publish(label_msg)

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
    node = ObjectClassifierJack()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()