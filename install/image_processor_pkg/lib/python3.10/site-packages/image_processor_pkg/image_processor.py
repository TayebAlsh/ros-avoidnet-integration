import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image, Joy
import cv2
import numpy as np
import torch
from cv_bridge import CvBridge
from PIL import Image as PILImage
from ament_index_python.packages import get_package_share_directory
import os

# Import custom modules
from image_processor_pkg.AvoidNet.avoid_net import get_model
from image_processor_pkg.AvoidNet.dataset import SUIM_grayscale
from image_processor_pkg.AvoidNet.draw_obsticle import draw_red_squares
from image_processor_pkg.AvoidNet.trajectory import determain_trajectory

obstacle = 0

class ImageProcessor(Node):
    def __init__(self, arc, run_name, use_gpu=False, threshold=0.9):
        super().__init__('image_processor')
        self.bridge = CvBridge()
        self.threshold = threshold

        # Dynamically get the model path using get_package_share_directory
        model_path = os.path.join(
            get_package_share_directory('image_processor_pkg'),
            'models',
            f"{arc}_{run_name}.pth"
        )

        # Model setup
        self.model = get_model(arc)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.model.to(self.device).eval()

        # Transformation setup
        self.image_transform = SUIM_grayscale.get_transform()

        # Subscription to the compressed image topic
        self.subscription = self.create_subscription(
            CompressedImage, '/cam1/camera/image_raw/compressed', self.image_callback, 10)
        
        self.subscription_joy = self.create_subscription(
            Joy, '/joy', self.joy_callback, 10)

        # Publisher for the processed image
        self.processed_image_publisher = self.create_publisher(Image, 'processed_image_topic', 80)
        self.joy_manipulation = self.create_publisher(Joy, '/joy', 80)

        
        self.get_logger().info(f"Node initialized with model: {arc} on {self.device}")

    def image_callback(self, msg):
        # Convert ROS CompressedImage message to OpenCV format
        frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # downscale to 320x240
        #frame = cv2.resize(frame, (640, 480))

        # Pre-process the frame
        frame_tensor = PILImage.fromarray(frame)
        frame_tensor = self.image_transform(frame_tensor).to(self.device).unsqueeze(0)

        # Run inference
        outputs = self.model(frame_tensor)
        outputs = outputs.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()  # Convert to numpy array

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
            
            obstacle = 1


        else:
            cv2.putText(frame, "Path Clear!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            obstacle = 0

        # Resize the frame for display
        #frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

        # Display the processed frame (for testing purposes, can be removed if not needed)
        # cv2.imshow("Processed Frame", frame)
        # cv2.waitKey(1)

        # Convert the processed frame to a ROS Image message and publish it
        processed_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.processed_image_publisher.publish(processed_msg)
        # print("going on")


    def joy_callback(self, msg):
        # remove 
        #self.joy_msg =Joy()
        self.joy_msg = msg
        
        if msg.buttons[4] == 1 and obstacle == 1:
            self.joy_msg.axes[1] = 1 
            self.joy_manipulation.publish(self.joy_msg)




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
