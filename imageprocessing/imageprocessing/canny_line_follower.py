import rclpy
from rclpy.node import Node
import cv2
import numpy as np
# from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
import math


W = 240
H = 160

CANNY_LOW = 70
CANNY_RATIO = 3
CANNY_KERNELSIZE = 3

HOUGH_RHO = 1
HOUGH_THETA = np.pi/180
HOUGH_THRESHOLD = 10
HOUGH_MIN_LEN = 1
HOUGH_MAX_GAP = 3


class ImageDisplay(Node):
	def __init__(self):
		super().__init__('image_display')
		self.image_sub = self.create_subscription(
	    Image, '/camera_relay/image_raw',
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
		frame = np.array(msg.data, dtype=np.uint8).reshape(
				msg.height, math.floor(msg.step/3), 3
			)
		if frame is None:
			self.get_logger().warn("Failed to decode image.")
			return
		frame = frame[:,:W-1]
		# canny
		canny = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		canny = cv2.blur(canny, (3,3))
		canny = cv2.Canny(canny, CANNY_LOW, CANNY_LOW*CANNY_RATIO, CANNY_KERNELSIZE)
		# select bottom few pixels
		lines = cv2.HoughLinesP(
			canny[int(H/2):,:], 
			HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD, 
			None, HOUGH_MIN_LEN, HOUGH_MAX_GAP
		)
		lines_img = np.copy(frame[int(H/2):,:])
		if lines is not None:
			for i in range(0,len(lines)):
				l = lines[i][0]
				angle = math.atan2(l[3]-l[1], l[3]-l[0])
				x = (angle + np.pi)/(2*np.pi)
				delta = 0.3
				p0 = -0.5
				p1 = 0.5
				if (p0-delta<x and x<p0+delta) or (p1-delta<x and x<p1+delta):
					cv2.line(
						lines_img, 
						(l[0], l[1]), (l[2], l[3]), 
						(0,255,0), 1, cv2.LINE_AA
					)
				else:
					cv2.line(
						lines_img, 
						(l[0], l[1]), (l[2], l[3]), 
						(0,0,255), 1, cv2.LINE_AA
					)


		#img = img[H-8:,:]
		cv2.imshow("canny window", canny)
		cv2.imshow("lines window", lines_img)
		cv2.waitKey(3)


def main(args=None):
	rclpy.init(args=args)
	node = ImageDisplay()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	cv2.destroyAllWindows()
	node.destroy_node()
	rclpy.shutdown()


if __name__ == "__main__":
	main(sys.argv)
