import time

class FPSCounter:
    def __init__(self):
        self.fps_values = []

    def start(self):
        self.start_time = time.time()

    def end(self):
        end_time = time.time()
        self.fps_values.append(1 / (end_time - self.start_time))
        if len(self.fps_values) > 100:
            self.fps_values.pop(0)

    def get_average_fps(self):
        return round(sum(self.fps_values) / len(self.fps_values), 2) if self.fps_values else 0