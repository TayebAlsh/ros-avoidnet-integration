# Import necessary modules
print("loading imu")
from IMU import IMU
print("loading obstacle_system")
from obsticale_system import ObstacleSystem
print("loading servo_controller")
from servo import ServoController
import time

class MainFrame:
    def __init__(self):
        # Initialize IMU
        ser_port = '/dev/ttyACM0'
        print(f"initializing imu on serial port {ser_port}")
        self.imu = IMU('/dev/ttyACM0')

        # Initialize AvoidNet
        model_name = "ImageReducer_bounded_grayscale"
        model_version = "run_2"
        threshold = 0.4
        print(f"initializing avoid net with model {model_name} version {model_version} and threshold {threshold}")
        self.avoid_net = ObstacleSystem(model_name, model_version, threshold)
        # Initialize Servo Controller
        servo_ports = [18,23]
        print(f"initializing servo controller with ports {servo_ports}")
        self.servo_controller = ServoController(servo_ports)
        self.locking_angle = 0
        self.pitch = 0
        self.roll = 0

    def process_data(self):
        # Get IMU data
        imu_data = self.imu.get_current_data()
        # # Get obstacle data
        obstacle_data = self.avoid_net.avoid_obsticale()
        # print(obstacle_data)
        
        return imu_data, None
    
    def react(self, imu_data, obsticale_data):
        # React to data
        # decontruct data
        if imu_data is not None:
            self.roll, self.pitch = imu_data[0], imu_data[1]
        # found, angel = obsticale_data[0], obsticale_data[1]
        # if found:
        #     # adjust the servo angels based on the angel of the obstacle
        #     self.servo_controller.change_servo_position(0, angel)
        #     self.servo_controller.change_servo_position(1, angel)
        # adjust the servo angels based on the pitch and roll to keep the camera level
        left_collective = self.pitch + self.roll
        right_collective = self.pitch - self.roll
        self.servo_controller.change_servo_position(0, left_collective)
        self.servo_controller.change_servo_position(1, right_collective)

def main():
    main_frame = MainFrame()
    while True:
        tic = time.time()
        imu_data, obsticale_data = main_frame.process_data()
        main_frame.react(imu_data, obsticale_data)
        toc = time.time()
        reaction_time = round(toc - tic, 4)
        print(f"reaction time: {reaction_time} s", end='\r')
if __name__ == "__main__":
    main()