import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class VideoPublisher(Node):
    def __init__(self, video_path):
        super().__init__('video_publisher')
        self.publisher = self.create_publisher(Image, 'image_topic', 10)
        self.bridge = CvBridge()
        self.video_path = video_path
        self.timer = self.create_timer(1/30.0, self.publish_frame)  # Assuming 30 FPS video
        self.cap = cv2.VideoCapture(self.video_path)

        # Check if video opened successfully
        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open video file: {self.video_path}")
        else:
            self.get_logger().info(f"Publishing video frames from {self.video_path}")

    def publish_frame(self):
        # Capture frame-by-frame
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().info("End of video reached, looping to start")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop back to the beginning
            ret, frame = self.cap.read()

        # Convert frame to ROS Image message and publish
        if ret:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.publisher.publish(msg)

    def __del__(self):
        # Release video capture on cleanup
        if self.cap.isOpened():
            self.cap.release()
        self.get_logger().info("VideoPublisher node terminated")

def main(args=None):
    rclpy.init(args=args)
    video_path = '/home/risc/rov/src/image_processor_pkg/image_processor_pkg/GX012307-shortened.mp4'  # Update this path to your MP4 file
    video_publisher = VideoPublisher(video_path)
    rclpy.spin(video_publisher)
    video_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

