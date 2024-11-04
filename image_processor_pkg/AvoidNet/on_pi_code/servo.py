import RPi.GPIO as GPIO
import time
import curses

class ServoController:
    def __init__(self, servo_pins, frequency=50, neutral_duty_cycle=7.5, max_duty_cycle=12.5, min_duty_cycle=2.5):
        self.servo_pins = servo_pins
        self.frequency = frequency
        self.neutral_duty_cycle = neutral_duty_cycle
        self.max_duty_cycle = max_duty_cycle
        self.min_duty_cycle = min_duty_cycle
        self.pwms = []
        self.max_angle = 75

        # Set the GPIO mode
        GPIO.setmode(GPIO.BCM)

        # Initialize the GPIO pins for the servos
        for pin in self.servo_pins:
            GPIO.setup(pin, GPIO.OUT)
            pwm = GPIO.PWM(pin, self.frequency)
            pwm.start(self.neutral_duty_cycle)
            self.pwms.append(pwm)


    def change_servo_position(self, servo_index, degree):
        
        # chekc if the degree is within the range
        if degree > self.max_angle:
            degree = self.max_angle
        elif degree < -self.max_angle:
            degree = -self.max_angle

        # Calculate the duty cycle for the given degree
        duty_cycle = self.neutral_duty_cycle + (degree / 180) * (self.max_duty_cycle - self.min_duty_cycle)
        if duty_cycle > self.max_duty_cycle:
            duty_cycle = self.max_duty_cycle
        elif duty_cycle < self.min_duty_cycle:
            duty_cycle = self.min_duty_cycle
        
        # Change the duty cycle
        self.pwms[servo_index].ChangeDutyCycle(duty_cycle)
        
        return duty_cycle
        
        
    def cleanup(self):
        for pwm in self.pwms:
            pwm.stop()
        GPIO.cleanup()

# # Usage example
# servo_pins = [18, 19]  # Example servo pins
# controller = ServoController(servo_pins)
# change_servo_position(0, 10)