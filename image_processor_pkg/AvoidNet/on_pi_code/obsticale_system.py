import cv2
import numpy as np
from avoid_net import get_model
import torch
from PIL import Image
from torchvision import transforms


class ObstacleSystem:
    def __init__(self, arc, run_name, threshold=0.5, fake=False, record=False):
        # Disable MKL-DNN backend
        torch.backends.mkldnn.enabled = False
        # torch.set_num_threads(1)
        self.record = record
        self.arc = arc
        self.run_name = run_name
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    [155, 155], interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        self.model = self.load_model()
        self.fake = fake
        self.cap = self.open_camera()
        self.device = torch.device("cpu")
        self.threshold = threshold

    def load_model(self):
        model = get_model(self.arc)
        model.load_state_dict(
            torch.load(
                f"models/{self.arc}_{self.run_name}.pth",
                map_location=torch.device("cpu"),
            )
        )
        device = torch.device("cpu")
        print(f"Running model on: {device}")
        model.to(device)
        model.eval()
        return model

    def open_camera(self):
        # Create a VideoCapture object
        if self.fake:
            cap = cv2.VideoCapture("videos/fake_video.mp4")
        else:
            cap = cv2.VideoCapture(0)
        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error opening video stream or file")
        else:
            print("Camera is open")
            if self.record:
                print("Preparing to record")
                # Define the codec and create VideoWriter object, use mp4
                import time

                date_time = time.strftime("%Y-%m-%d_%H-%M-%S")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.out = cv2.VideoWriter(
                    f"videos/output_{date_time}.mp4", fourcc, 5.0, (640, 480)
                )
        return cap

    def process_frame(self, frame):
        # prepare the frame for inference
        frame_tensor = Image.fromarray(frame)
        frame_tensor = self.transform(frame_tensor).to(self.device).unsqueeze(0)
        outputs = self.model(frame_tensor)
        # move the output tensors to cpu for visualization
        outputs = outputs.detach().cpu()
        outputs = outputs[0].permute(1, 2, 0)
        outputs = np.array(outputs)  # Convert to numpy array
        # show the output
        # frame = draw_red_squares(frame, outputs, 0.5)
        return frame

    def find_and_avoid(self, grid):
        found = False
        search_area = 5

        center = (grid.shape[1] // 2, grid.shape[0] // 2)

        # see if any of the squares in the center of the grid plus search area are above the threshold
        center_area = grid[
            center[1] - search_area : center[1] + search_area,
            center[0] - search_area : center[0] + search_area,
        ]
        if np.any(center_area > self.threshold):
            found = True

        # search up, left and right of the center for a clear path and set the new trajectory
        if found:
            # get the average of all values of the grid above the centerr area
            up = np.mean(
                grid[: center[1], center[0] - search_area : center[0] + search_area]
            )
            # get the average of all values of the grid to the left of the centerr area
            left = np.mean(
                grid[center[1] - search_area : center[1] + search_area, : center[0]]
            )
            # get the average of all values of the grid to the right of the centerr area
            right = np.mean(
                grid[center[1] - search_area : center[1] + search_area, center[0] :]
            )

            # find the lowest average and set the new trajectory
            if up < left and up < right:
                direction = "up"
            elif left < up and left < right:
                direction = "left"
            elif right < up and right < left:
                direction = "right"
            else:
                direction = "up"

            return found, direction
        else:
            return found, None

    def avoid_obsticale(self):
        # Read frame from camera
        if self.cap.isOpened():
            # self.capture frame-by-frame
            ret, frame = self.cap.read()
            if self.record:
                # save the frame to a video file
                self.out.write(frame)
            if ret:
                # Process the frame
                output = self.process_frame(frame)
                found, direction = self.find_and_avoid(output)
                return found, direction
            else:
                print("there has been an error getting a frame from camera")
                return None, None
        else:
            # print("camera is not open")
            return None, None

    def cleanup(self):
        # When everything is done, release the video self.capture object
        print("Releasing camera")
        self.cap.release()
        # close the video file
        if self.record:
            self.out.release()


# example usage
# arc = "resnet18"
# run_name = "run_1"
# obstacle_system = ObstacleSystem(arc, run_name)
# obstacle_system.run()
