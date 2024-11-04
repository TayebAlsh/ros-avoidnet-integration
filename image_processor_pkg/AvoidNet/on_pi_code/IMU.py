import serial
import serial.tools.list_ports
import time

class IMU:
    def __init__(self, serial_port):
        self.serial_port = serial_port
        self.ser = serial.Serial(serial_port, baudrate=115200, timeout=0.08)

    def get_current_data(self):
        data = self.ser.readline().decode('utf-8').strip()
        if data:
            data = data.split(',')
            roll = float(data[2])
            pitch = float(data[3])
            return roll, pitch
        else:
            return None

    def cleanup(self):
        self.ser.close()