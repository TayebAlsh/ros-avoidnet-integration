import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32, Bool
import cv2
import numpy as np
import torch
from cv_bridge import CvBridge
from PIL import Image as PILImage
from ament_index_python.packages import get_package_share_directory
import os
import time

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
        # Track previous obstacle state to detect transitions (obstacle -> clear)
        self._prev_obstacle = 0

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

        # Subscription to the image topic (compressed transport)
        # The topic name ends with '/compressed' so the message type is CompressedImage
        self.subscription = self.create_subscription(
            CompressedImage, '/cam1/camera/image_raw/compressed', self.image_callback, 30)
        
        self.get_logger().info(f"Subscribed to {self.subscription}")
        
        # Add a counter to track received images
        self.image_count = 0
        
        # Publisher for the processed image
        self.processed_image_publisher = self.create_publisher(Image, 'processed_image_topic', 80)

        # Publisher to control depth when an obstacle is detected
        # Sends a Float32 value on the '/set_depth' topic (value 0.0 requested)
        self.set_depth_publisher = self.create_publisher(Float32, '/set_depth', 10)
        # Track last desired depth (restored from values/desired_depth_latest.txt when needed)
        self.last_desired_depth = None
        self._last_depth_publish_time = 0.0

        # Subscribe to AI enable topic so external UI can enable/disable automatic depth control
        # Default to disabled until UI explicitly enables it
        self.ai_enabled = False
        self.ai_enable_sub = self.create_subscription(Bool, '/ai_depth_enable', self.ai_enable_callback, 10)

        self.get_logger().info(f"Node initialized with model: {arc} on {self.device} - updated!!!")

    def image_callback(self, msg):
        try:
            # Convert ROS CompressedImage message to OpenCV format
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.image_count += 1
            
            # Log every 10th image to avoid spam
            # if self.image_count % 10 == 0:
                # self.get_logger().info(f"Processed {self.image_count} images. Current: {frame.shape[1]}x{frame.shape[0]} pixels")
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
        # obstacle is truthy when obstacle detected; convert to int 0/1 for state tracking
        obs_flag = 1 if obstacle else 0
        prev_obs = getattr(self, '_prev_obstacle', 0)

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

            # Send a depth command of 0 when obstacle detected, but only if AI depth control is enabled
            if self.ai_enabled:
                depth_msg = Float32()
                depth_msg.data = 0.0
                self.set_depth_publisher.publish(depth_msg)
                # Reset last_desired_depth so we will reload/publish the saved desired depth
                # when the obstacle clears
                self.last_desired_depth = None
                self.get_logger().info("Published /set_depth = 0.0 due to obstacle (AI enabled)")
            else:
                self.get_logger().debug("Obstacle detected but AI depth control is disabled; not publishing /set_depth")

            self.obstacle = 1


        else:
            cv2.putText(frame, "Path Clear!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # If we transitioned from obstacle -> clear, restore the saved
            # desired depth once so the vehicle returns to its operating depth.
            if self.ai_enabled and prev_obs == 1 and obs_flag == 0:
                desired_depth = self._get_last_desired_depth()
                if desired_depth is not None:
                    try:
                        depth_msg = Float32()
                        depth_msg.data = desired_depth
                        self.set_depth_publisher.publish(depth_msg)
                        self.get_logger().info(f"Published /set_depth = {desired_depth} (restored after obstacle)")
                        # cache it so we don't re-read the file repeatedly
                        self.last_desired_depth = desired_depth
                    except Exception as e:
                        self.get_logger().error(f"Failed to publish restored desired depth: {e}")
            self.obstacle = 0

        # Update previous obstacle flag for next frame
        self._prev_obstacle = obs_flag

        # Resize the frame for display
        #frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

        # Display the processed frame (for testing purposes, can be removed if not needed)
        # cv2.imshow("Processed Frame", frame)
        # cv2.waitKey(1)

        # Convert the processed frame to a ROS Image message and publish it
        processed_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.processed_image_publisher.publish(processed_msg)
        # print("going on")

    def ai_enable_callback(self, msg):
        try:
            self.ai_enabled = bool(msg.data)
            self.get_logger().info(f"AI depth control enabled: {self.ai_enabled}")
        except Exception as e:
            self.get_logger().error(f"Error handling AI enable message: {e}")

    def _get_last_desired_depth(self):
        """Return the last desired depth saved by the vehicle widget or None.

        This reads the file `values/desired_depth_latest.txt` which is written by
        the C++ widget. The result is cached in self.last_desired_depth so we
        don't repeatedly hit the filesystem.
        """
        if self.last_desired_depth is not None:
            return self.last_desired_depth

        # Try to read the saved desired depth from the values directory
        paths_to_try = [
            os.path.join(os.getcwd(), 'values', 'desired_depth_latest.txt'),
            os.path.join('values', 'desired_depth_latest.txt')
        ]
        for path in paths_to_try:
            try:
                if os.path.isfile(path):
                    with open(path, 'r') as f:
                        text = f.read().strip()
                        if not text:
                            continue
                        # file may contain a single float or extra whitespace
                        parts = text.split()
                        val = float(parts[0])
                        self.last_desired_depth = val
                        return val
            except Exception as e:
                # Non-fatal; we'll try the next location or return None
                self.get_logger().debug(f"Could not read desired depth file '{path}': {e}")

        return None


    # joystick functionality removed — depth commands are published to '/set_depth' instead



def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessor(
        arc="ImageReducer_bounded_grayscale",
        run_name="run_2_1",
        use_gpu=False
    )
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
