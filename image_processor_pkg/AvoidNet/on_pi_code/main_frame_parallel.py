# Import necessary modules
print("loading imu")
from IMU import IMU
print("loading obstacle_system")
from obsticale_system import ObstacleSystem
print("loading servo_controller")
from servo import ServoController
import time
from multiprocessing import Process, Pipe
import sys
from average_fps import FPSCounter
import argparse

class MainFrame:
    def __init__(self, fake=False, record=False):
        print("initializing main frame")
        self.fake = fake
        self.record = record
        
    def process_data(self, conn):
        detection_fps = FPSCounter()
        # Initialize AvoidNet
        model_name = "ImageReducer_bounded_grayscale"
        model_version = "run_2"
        threshold = 0.4
        print(f"initializing avoid net with model {model_name} version {model_version} and threshold {threshold}")
        avoid_net = ObstacleSystem(model_name, model_version, threshold, self.fake, self.record)
        print("\nstarted visual guidance\n")
        while True:
            detection_fps.start()
            obstacle_data = avoid_net.avoid_obsticale()
            detection_fps.end()
            conn.send([obstacle_data, detection_fps.get_average_fps()])
                
    
    def react(self, conn):
        # React to data
        data_back = [[False,None], 0]
        # Initialize IMU
        ser_port = '/dev/ttyACM0'
        print(f"initializing imu on serial port {ser_port}")
        imu = IMU('/dev/ttyACM0')

        # Initialize Servo Controller
        servo_ports = [18,23]
        print(f"initializing servo controller with ports {servo_ports}")
        servo_controller = ServoController(servo_ports)
        locking_angle = 0
        pitch = 0
        roll = 0
        
        reaction_fps = FPSCounter()
        while True:
            reaction_fps.start()
            # Get IMU data
            imu_data = imu.get_current_data()
            # decontruct data
            if imu_data is not None:
                roll, pitch = imu_data[0], imu_data[1]
            # found, angel = obsticale_data[0], obsticale_data[1]
            # if found:
            #     # adjust the servo angels based on the angel of the obstacle
            #     servo_controller.change_servo_position(0, angel)
            #     servo_controller.change_servo_position(1, angel)
            # adjust the servo angels based on the pitch and roll to keep the camera level
            left_collective = pitch + roll
            right_collective = pitch - roll
            servo1 = servo_controller.change_servo_position(0, left_collective)
            servo2 = servo_controller.change_servo_position(1, right_collective)
            if conn.poll():
                data_back = conn.recv()
            
            avoid_fps = data_back[1]
            obstacle = data_back[0]
            # TODO: implement the reaction to the obstacle here
            
            reaction_fps.end()
            print(f"reaction fps: {reaction_fps.get_average_fps()}, obstacle time: {avoid_fps}, obstacle interrupt: {obstacle} || servo 1 and 2 : {round(servo1, 3), round(servo2, 3)}              ", end="\r")
def main(fake=False, record=False):
    main_frame = MainFrame(fake, record)
    parent_conn, child_conn = Pipe()
    process1 = Process(target=main_frame.process_data, args=(child_conn,))
    process2 = Process(target=main_frame.react, args=(parent_conn,))
    
    
    
    # if keyboard interrupt, stop the processes and clean up
    try:
        # Start the processes
        process1.start()
        process2.start()
    except KeyboardInterrupt:
        print("\n===cleaning up...===\n")
        process1.terminate()
        process2.terminate()
        # process2.terminate()
        main_frame.servo_controller.cleanup()
        print("servos cleaned")
        main_frame.imu.cleanup()
        print("imu cleaned")
        main_frame.avoid_net.cleanup()
        print("avoid net cleaned")
        print("===All good!===\n")
        sys.exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--fake",
        type=bool,
        default=False,
        help="Make the system react to a video instead of real camera feed, video should be placed in videos folder and names fake_video.mp4"
    )
    parser.add_argument(
        "--record",
        type=bool,
        default=False,
        help="Record the video feed from the camera, video will be saved in the videos folder with the name video_feed.mp4"
    )
    args = parser.parse_args()
    
    main(args.fake, args.record)