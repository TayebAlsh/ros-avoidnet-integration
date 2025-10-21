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

class ImageProcessor(Node):
    def __init__(self, arc, run_name, use_gpu=False, threshold=0.5):
        super().__init__('image_processor')
        self.bridge = CvBridge()
        self.threshold = threshold
        self.obstacle = 0

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

        # Subscription to the image topic
        self.subscription = self.create_subscription(
            Image, '/cam1/camera/image_raw', self.image_callback, 10)
        
        self.get_logger().info("Subscribed to /cam1/camera/image_raw")
        
        # Add a counter to track received images
        self.image_count = 0
        
        self.subscription_joy = self.create_subscription(
            Joy, '/joy', self.joy_callback, 10)

        # Publisher for the processed image
        self.processed_image_publisher = self.create_publisher(Image, 'processed_image_topic', 80)
        self.joy_manipulation = self.create_publisher(Joy, '/joy', 80)

        
        self.get_logger().info(f"Node initialized with model: {arc} on {self.device} - updated!!!")

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV format (not compressed!)
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.image_count += 1
            
            # Log every 10th image to avoid spam
            if self.image_count % 10 == 0:
                self.get_logger().info(f"Processed {self.image_count} images. Current: {frame.shape[1]}x{frame.shape[0]} pixels")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {str(e)}")
            return

        # downscale to 640x480
        # frame = cv2.resize(frame, (640, 480))

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

            # TODO: send signal to the controller here! send the `new_trej` variable
            
            self.obstacle = 1


        else:
            cv2.putText(frame, "Path Clear!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            self.obstacle = 0

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
        # Print the received joystick message
        print(msg)

        # # Check if button 4 is pressed (indexing starts at 0)
        # if len(msg.buttons) > 4 and msg.buttons[4]:
        #     # Create a copy of the message and modify axes to pitch up
        #     modified_msg = Joy()
        #     modified_msg.header = msg.header
        #     modified_msg.axes = list(msg.axes)
        #     modified_msg.buttons = list(msg.buttons)

        #     # Ensure there is a second axis for pitch
        #     if len(modified_msg.axes) < 2:
        #         modified_msg.axes += [0.0] * (2 - len(modified_msg.axes))

        #     # Set pitch up command
        #     modified_msg.axes[1] = 1.0

        #     # Publish the modified message
        #     self.joy_manipulation.publish(modified_msg)
        return



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
