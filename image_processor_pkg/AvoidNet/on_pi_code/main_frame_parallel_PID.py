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
import os
from average_fps import FPSCounter
import argparse
from PID_control import PID
from info_displayer import DynamicDictDisplay_bare
from depth_reader import DepthReader

class MainFrame:
    def __init__(self, fake=False, record=False, depth=2):
        print("initializing main frame")
        self.fake = fake
        self.record = record
        self.stat = {}
        self.display = DynamicDictDisplay_bare(self.stat)
        self.depth = depth
        if self.record:
            # open a txt file inside the videos folder and write stats to it
            date_time = time.strftime("%Y-%m-%d_%H-%M-%S")
            # if the file does not exist, create it
            if not os.path.exists("videos"):
                os.makedirs("videos")
            self.file = open(f"videos/{date_time}_stats.txt", "w")
        
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
        
        # initialize depth reader
        print("initializing depth reader")
        depth_reader = DepthReader()

        # Initialize Servo Controller
        servo_ports = [18,23]
        print(f"initializing servo controller with ports {servo_ports}")
        servo_controller = ServoController(servo_ports)
        print("Initializing PID controller...")
        # Example usage with pitch and roll control
        pitch_pid = PID(kp=1, ki=0, kd=0, target=-2, limits=[-75, 75])
        roll_pid = PID(kp=1, ki=0, kd=0, target=0, limits=[-75, 75])
        depth_pid = PID(kp=1, ki=0, kd=0, target=self.depth, limits=[-75, 75])
        pitch = 0
        roll = 0
        depth = 0
        temps = 0
        avoidance_angel = 30
        
        reaction_fps = FPSCounter()
        while True:
            reaction_fps.start()
            # Get IMU data
            imu_data = imu.get_current_data()
            # decontruct data
            if imu_data is not None:
                roll, pitch = imu_data[0], imu_data[1]
            
            # get depth and temps from depth reader
            depth, temps = depth_reader.get_depth_and_temp()
            
            pitch_output = pitch_pid.update(pitch)
            roll_output = roll_pid.update(roll)
            depth_output = depth_pid.update(depth)
            
            # Control signals for servos
            servo1_output = pitch_output + roll_output
            servo2_output = pitch_output - roll_output
            # control signals for depth
            servo1_output += depth_output
            servo2_output += depth_output
            
            # update the self.stat dictionary
            self.stat["pitch"] = pitch
            self.stat["roll"] = roll
            self.stat["pitch_output"] = pitch_output
            self.stat["roll_output"] = roll_output
            self.stat["servo1_output"] = servo1_output
            self.stat["servo2_output"] = servo2_output
            self.stat["depth"] = depth
            self.stat["depth_output"] = depth_output
            self.stat["temps"] = temps
            if self.record:
                # write stats to file
                self.file.write(f"pitch: {pitch}, roll: {roll}, pitch_output: {pitch_output}, roll_output: {roll_output}, servo1_output: {servo1_output}, servo2_output: {servo2_output}, depth: {depth}, depth_output: {depth_output}, temps: {temps}\n")
            
            servo1 = servo_controller.change_servo_position(0, servo1_output)
            servo2 = servo_controller.change_servo_position(1, servo2_output)
            if conn.poll():
                data_back = conn.recv()
            
            avoid_fps = data_back[1]
            obstacle = data_back[0]

            # Change the target pitch and roll based on the direction given by the obstacle system
            if obstacle[0]:
                if obstacle[1] == "left":
                    roll_pid.target = -avoidance_angel
                    pitch_pid.target = 0
                elif obstacle[1] == "right":
                    roll_pid.target = avoidance_angel
                    pitch_pid.target = 0
                elif obstacle[1] == "up":
                    pitch_pid.target = avoidance_angel
            else:
                pitch_pid.target = -2
                roll_pid.target = 0
            
            reaction_fps.end()
            self.stat["reaction_fps"] = reaction_fps.get_average_fps()
            self.stat["obstacle_time"] = avoid_fps
            self.stat["obstacle_interrupt"] = obstacle
            self.stat["servo1"] = round(servo1, 3)
            self.stat["servo2"] = round(servo2, 3)
            self.display.update_dict(self.stat)
            # print(f"reaction fps: {self.stat['reaction_fps']}, obstacle time: {self.stat['obstacle_time']}, obstacle interrupt: {self.stat['obstacle_interrupt']} || servo 1 and 2 : {self.stat['servo1'], self.stat['servo2']}              ", end="\r")
def main(fake=False, record=False, depth=2):
    main_frame = MainFrame(fake, record, depth)
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
        main_frame.display.stop_display()
        print("display cleaned")
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
    # depth argument
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="Set depth, default is 2m"
    )
    args = parser.parse_args()
    
    main(args.fake, args.record, args.depth)