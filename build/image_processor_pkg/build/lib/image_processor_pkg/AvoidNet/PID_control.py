# this is a PID controller which recieves pitch and roll from the IMU at a rate of 30Hz and outputs the PWM values for the motors
import numpy as np


class PID:
    def __init__(self, kp, ki, kd, setpoint, output_limits):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self.integral = 0
        self.prev_error = 0

    def update(self, measured_value):
        error = self.setpoint - measured_value
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        print(output)
        output = np.clip(output, self.output_limits[0], self.output_limits[1])
        return output