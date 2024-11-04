from servo import ServoController

# Initialize Servo Controller
# keep the servo at the zero position
servo_ports = [18,23]
print(f"initializing servo controller with ports {servo_ports}")
servo_controller = ServoController(servo_ports)
counter = 0
while counter < 50000:
    counter += 1
    servo_controller.change_servo_position(0, 0)
    servo_controller.change_servo_position(1, 0)
    print(f"counter: {counter}", end="\r")
print(f"counter: {counter}")