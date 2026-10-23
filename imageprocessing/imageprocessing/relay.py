import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage


class ImageRelay(Node):
	def __init__(self):
		super().__init__('image_display')
		self.image_sub = self.create_subscription(
			Image, '/camera/image_raw',
			# CompressedImage, '/camera/image_raw/compressed',
			self.image_callback,
			10
		)
		self.image_pub = self.create_publisher(Image, '/camera_relay/image_raw', 10)

	def image_callback(self, msg):
		self.image_pub.publish(msg)


def main(args=None):
	rclpy.init(args=args)
	node = ImageRelay()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	node.destroy_node()
	rclpy.shutdown()


if __name__ == "__main__":
	main(sys.argv)
