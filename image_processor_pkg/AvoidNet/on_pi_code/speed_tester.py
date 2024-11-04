from IMU import IMU
from average_fps import FPSCounter

imu_timer = FPSCounter()

while True:
    ser_port = '/dev/ttyACM0'
    imu = IMU(ser_port)
    imu_timer.start()
    data = imu.get_current_data()
    if data is None:
        continue
    pitch , roll = data[0], data[1]
    imu_timer.end()
    print(pitch, roll, imu_timer.get_average_fps(), end='\r')