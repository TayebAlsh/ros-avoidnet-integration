import numpy as np

class PID:
    def __init__(self, kp, ki, kd, target, limits):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target = target
        self.limits = limits
        self.integral = 0
        self.prev_error = 0
        
    def update(self, current):
        error = self.target - current
        self.integral += error
        derivative = error - self.prev_error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        output = np.clip(output, self.limits[0], self.limits[1])
        self.prev_error = error
        return output
    
    def set_target(self, target):
        self.target = target
