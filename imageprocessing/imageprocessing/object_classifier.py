import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from ultralytics import YOLO

from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String

TARGET_OBJECTS = ["bottle", "cup", "cell phone", "backpack"]
MIN_CONFIDENCE = 0.2


class ObjectClassifier(Node):
    def __init__(self):
        super().__init__('object_classifier')

        # YOLO model
        self.model = YOLO("yolov8n.pt")

        # Subscribe to compressed camera feed
        self.image_sub = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.image_callback,
            10
        )

        # Publishers
        self.label_pub = self.create_publisher(String, '/vision/object_labels', 10)
        self.annotated_pub = self.create_publisher(Image, '/vision/object_annotated', 10)

        self.get_logger().info("ObjectClassifier node initialized.")

    # ---------------------------------------------------------
    # Decode compressed image → run YOLO → publish results
    # ---------------------------------------------------------
    def image_callback(self, msg):
        # Decode compressed image
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            self.get_logger().warn("Failed to decode compressed image.")
            return

        # Run YOLO inference
        results = self.model(frame)
        detected_labels = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = self.model.names[cls]
                conf = float(box.conf[0])

                if label in TARGET_OBJECTS and conf > MIN_CONFIDENCE:
                    detected_labels.append(label)

                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Label text
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
    node = ObjectClassifier()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

