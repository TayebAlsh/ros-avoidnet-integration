# Obstacle Avoidance Integeration

## What is this document
✅ *top pc* Implementation of sending control signals to the vehicle
✅ New implementation for running the model onboard the vehicle (avoiding visual delay)
✅ Explanation of how each code functions and where it should be used to integrate with the controller.

## Resources
### git repo with new modified code
I had no way to push to the current repo so I made a new one and pushed it here:
https://github.com/Alooi/ros-avoidnet-integration.git
*currently public, will private it later*
## Files to pay attention to

- *modified* `image_processor_updated.py`
- *modified* `AvoidNet/trajectory/py`
- *New file* `on_pi_code/internal_avoidance.py`

### Where to receive signals

Inside `image_processor_updated.py` there is a snippet of code which utilizes `AvoidNet/trajectory.py` to find a simple exit route  *how it does it is explained in later section*.

```python
obstacle, new_trej = determain_trajectory(outputs, threshold=self.threshold)

# Display obstacle information
if obstacle:
    cv2.putText(frame, "Obstacle!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Turn {new_trej}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    h, w = frame.shape[:2]
    if new_trej == "left":
        cv2.arrowedLine(frame, (w//2, h//2), (w//2 - 100, h//2), (0, 255, 255), 2)
    elif new_trej == "right":
        cv2.arrowedLine(frame, (w//2, h//2), (w//2 + 100, h//2), (0, 255, 255), 2)
    elif new_trej == "up":
        cv2.arrowedLine(frame, (w//2, h//2), (w//2, h//2 - 100), (0, 255, 255), 2)
        # SEND SIGNAL HERE! to climb 1 or 2 meters to avoid the obstacle.
    
    self.obstacle = 1


else:
    cv2.putText(frame, "Path Clear!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    self.obstacle = 0
```

> [!NOTE]
> The function currently only draws an arrow indicator on the frame without sending any signals

In our case we would like to utilize the up function to send a signal to the controller to climb to the surface or a couple of meters

### How the trajectory finding works

  If an obstacle is present, it divides the grid into three areas: above, to the left, and to the
  right of the center. It then calculates the average value for each area. The direction with the
  lowest average value is chosen as the clearest path, effectively steering away from the densest
  regions of detected obstacles.

## How to send the signal

### Method 1: Emulate a joystick input *easier*

We wouldn't prefer this method, as it is very preemptive and might fail or confuse our manual control. But the idea here is to hijack the controller input and insert a pitch up signal, simple as that.

### Method 2: Using proper controller *proper*

We have a controller ready which would be able to maintain depth, by using the pitch. With this setup we can send an "Avoid" Signal and that will tell the controller to pitch up uncontrollably to a set depth or fully jump out then dive back in.

### Optionally

We can we can go left and right using the other conditions in `trajectory.py` if the controller allows us to move slightly to one side and maintain that until the obstacle is out of view.

## Bypassing the communication delay and running *directly* on the pi

There is code provided in `image_processor_pkg/on_pi_code`, specifically `image_processor_pkg/on_pi_code/internal_avoidance.py` which is code optimized to run on the pi, with similar functions, as described before.
### How to run the code on the PI
1. Copy the `on_pi_code` folder to the pi
2. ssh into the pi
3. spin up a new python/conda environment
4. install any required packages the code might tell you to install
	1. NOTE! you might run into a problem with installing opencv, just install the headless one
	2. NOTE! you can use the `requirements.txt` with pip 
5. run `internal_avoidance.py`

#### Running at startup on the pi
create a new service on the pi which would load the environment and start `internal_avoidance.py` file

#### What is provided in `internal_avoidance.py`
This will run locally on the pi, and has a similar function to how it calculates the clearest path. a clearly indicated **TODO** item is in the code, to indicate where and what to send as signal to the controller/Teensy.

### General improvements (in the works)
Since we are using the top PC which is a lot more capable than the PI, we do not have to stick with a small efficient model. Therefore, I will be trying a **new larger and more capable** model. We are still going to be using the smaller model if we need to run the model on the PI.
