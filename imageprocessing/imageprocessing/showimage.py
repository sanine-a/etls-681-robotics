import rclpy
from rclpy.node import Node
import cv2
import numpy as np
# from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String
import math


class ImageDisplay(Node):
	def __init__(self):
		super().__init__('image_display')
		self.image_sub = self.create_subscription(
			Image,
			# '/camera/image_raw',
			'/camera/image_raw/compressed',
			self.image_callback,
			10
		)

	# def image_callback(self, msg):
	# 	# print(msg.height, msg.width, msg.encoding, msg.step, len(msg.data))
	# 	img = np.array(msg.data, dtype=np.uint8).reshape(
	# 		msg.height, math.floor(msg.step/3), 3
	# 	)
	# 	# print(img.shape)
	# 	cv2.imshow("image window", img)
	# 	cv2.waitKey(3)

	def image_callback(self, msg):
		print(msg.format)
		# img = cv2.imdecode(np.asarray(msg.data, dtype=np.uint8), 0)
		# cv2.imshow("image window", img)
		# cv2.waitKey(3)


def main(args=None):
	rclpy.init(args=args)
	node = ImageDisplay()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	cv2.distroyAllWindows()
	node.destroy_node()
	rclpy.shutdown()


if __name__ == "__main__":
	main(sys.argv)
