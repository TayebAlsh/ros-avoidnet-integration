import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
import torch
from cv_bridge import CvBridge
from image_processor_pkg.AvoidNet.avoid_net import get_model
from dataset import SUIM_grayscale
from PIL import Image as PILImage
from draw_obsticle import draw_red_squares
from trajectory import determain_trajectory

class ImageProcessor(Node):
    def __init__(self, arc, run_name, use_gpu=False, threshold=0.7):
        super().__init__('image_processor')
        self.bridge = CvBridge()
        self.threshold = threshold

        # Model setup
        self.model = get_model(arc)
        self.model.load_state_dict(torch.load(f"models/{arc}_{run_name}.pth", map_location=torch.device('cpu')))
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.model.to(self.device).eval()

        # Transformations
        self.image_transform = SUIM_grayscale.get_transform()

        # Subscription to the image topic
        self.subscription = self.create_subscription(
            Image, 'image_topic', self.image_callback, 10)
        self.get_logger().info(f"Node initialized with model: {arc} on {self.device}")

    def image_callback(self, msg):
        # Convert ROS image message to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Pre-process the frame
        frame_tensor = PILImage.fromarray(frame)
        frame_tensor = self.image_transform(frame_tensor).to(self.device).unsqueeze(0)

        # Run inference
        outputs = self.model(frame_tensor)
        outputs = outputs.detach().cpu().squeeze().permute(1, 2, 0).numpy()  # Convert to numpy array

        # Process model output
        frame = draw_red_squares(frame, outputs, self.threshold)
        obstacle, new_trej = determain_trajectory(outputs, threshold=self.threshold)

        # Display obstacle information
        if obstacle:
            cv2.putText(frame, "Obstacle!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Turn {new_trej}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            h, w = frame.shape[:2]
            if new_trej == "left":
                cv2.arrowedLine(frame, (w//2, h//2), (w//2 - 100, h//2), (0, 255, 255), 2)
            elif new_trej == "right":
                cv2.arrowedLine(frame, (w//2, h//2), (w//2 + 100, h//2), (0, 255, 255), 2)
            elif new_trej == "up":
                cv2.arrowedLine(frame, (w//2, h//2), (w//2, h//2 - 100), (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Path Clear!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the frame (for testing purposes, can be removed if not needed)
        cv2.imshow("Processed Frame", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessor(
        arc="ImageReducer_bounded_grayscale",
        run_name="run_2",
        use_gpu=False
    )
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

