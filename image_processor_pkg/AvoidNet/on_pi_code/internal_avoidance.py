import json
import time
import cv2
from obsticale_system import ObstacleSystem


class InternalAvoidanceSystem(ObstacleSystem):
    """
    An extension of ObstacleSystem that allows for camera index configuration
    without modifying the original class.
    """

    def __init__(
        self, arc, run_name, camera_index=0, threshold=0.5, fake=False, record=False
    ):
        self.camera_index = camera_index
        # We call the parent __init__ after setting camera_index, because it calls open_camera()
        super().__init__(arc, run_name, threshold, fake, record)

    def open_camera(self):
        """
        Overrides the parent open_camera method to use the configured camera index.
        """
        if self.fake:
            cap = cv2.VideoCapture("videos/fake_video.mp4")
        else:
            cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            print(f"Error opening video stream for camera index {self.camera_index}")
        else:
            print(f"Camera {self.camera_index} is open")
            if self.record:
                print("Preparing to record")
                date_time = time.strftime("%Y-%m-%d_%H-%M-%S")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.out = cv2.VideoWriter(
                    f"videos/output_{date_time}.mp4", fourcc, 5.0, (640, 480)
                )
        return cap


def main():
    """
    Main function to run the internal avoidance system.
    """
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Error: config.json not found. Please create it.")
        return

    camera_index = config.get("camera_index", 0)

    # Following the pattern from main_frame_parallel.py
    model_name = "ImageReducer_bounded_grayscale"
    model_version = "run_2"
    threshold = 0.4

    print(f"Initializing Avoidance System...")
    print(f"Model: {model_name}, Version: {model_version}, Camera: {camera_index}")

    avoidance_system = None
    try:
        avoidance_system = InternalAvoidanceSystem(
            arc=model_name,
            run_name=model_version,
            camera_index=camera_index,
            threshold=threshold,
        )

        print("\n--- Starting Obstacle Detection ---")
        while True:
            found, direction = avoidance_system.avoid_obsticale()

            if found is None:
                print("Could not get frame from camera. Exiting.")
                break

            if found:
                print(
                    f"Obstacle Detected: YES | Suggested Direction: {direction.upper()}     ",
                    end="\r",
                )
                # TODO: Add the signal processing here, send to the Teensy! use `direction` variable
            else:
                print(
                    "Obstacle Detected: NO  | ---                                       ",
                    end="\r",
                )

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n--- User interrupted. Shutting down. ---")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if avoidance_system:
            avoidance_system.cleanup()
            print("System cleaned up.")


if __name__ == "__main__":
    main()
