import cv2
import numpy as np
import time
from avoid_net import get_model
import argparse
import torch
from dataset import SUIM, SUIM_grayscale
from PIL import Image
import numpy as np
from draw_obsticle import draw_red_squares
from average_fps import FPSCounter
import threading
import queue
import time
total_fps = FPSCounter()

# Disable MKL-DNN backend
torch.backends.mkldnn.enabled = False

torch.set_num_threads(4)

print(f"MKL-DNN status: {torch.backends.mkldnn.enabled}, number of threads: {torch.get_num_threads()}")

limiter = 50
dataset = SUIM_grayscale("TEST")
image_transform = dataset.get_transform()

def process_frame(frame, model, device):
    # prepare the frame for inference
    frame_tensor = Image.fromarray(frame)
    frame_tensor = image_transform(frame_tensor).to(device).unsqueeze(0)
    outputs = model(frame_tensor)
    # move the output tensors to cpu for visualization
    outputs = outputs.detach().cpu()
    outputs = outputs[0].permute(1, 2, 0)
    outputs = np.array(outputs)  # Convert to numpy array
    # show the output
    frame = draw_red_squares(frame, outputs, 0.5)
    # remove model from memory
    return frame

def run_model(arc, run_name, source, video_path=None, use_gpu=False, save_video=False):
    model = get_model(arc)
    model.load_state_dict(
        torch.load(f"models/{arc}_{run_name}.pth", map_location=torch.device("cpu"))
    )
    device = torch.device("cpu")
    print(f"Testing on: {device}")
    model.to(device)
    model.eval()
    
    if save_video:
        frames_array = queue.Queue()
    
    # Create a Queue with a maximum size of 4
    q = queue.Queue(maxsize=1)
    
    def worker():
        while True:
            item = q.get()
            if item is None:
                break
            frames_array.put(process_frame(item, model, device))
            q.task_done()
            
    # Start worker threads
    threads = []
    num_worker_threads = 6
    for i in range(num_worker_threads):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)

    # read from source using opencv
    if source == "webcam":
        cap = cv2.VideoCapture(0)
    elif source == "video":
        cap = cv2.VideoCapture(video_path)
        
    
    tic = time.time()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        if q.full():
            # Remove the oldest frame from the queue
            q.get()
        q.put(frame)
        if int(frames_array.qsize()) == limiter:
            break
        print(f"\r{frames_array.qsize()}/{limiter} || queue size: {q.qsize()}", end="\r")

    for i in range(num_worker_threads):
        q.put(None)
    for t in threads:
        t.join()
    
    toc = time.time()
    fps = 1 / ((toc - tic) / frames_array.qsize())
    print(f"FPS: {fps}")
    
    if save_video:
        print("Saving the video...")
        out = cv2.VideoWriter(f"output/{arc}_{run_name}_{source}.avi", cv2.VideoWriter_fourcc(*"DIVX"), int(fps), (frame.shape[1], frame.shape[0]))
        for i in range(frames_array.qsize()):
            out.write(frames_array.get())
        out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arc",
        type=str,
        default="unet",
        help="Architecture to be used for training",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="unet_1",
        help="Name of the run to be used for training",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="webcam",
        help="Source to be used for inference",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="",
        help="Path to the video to be used for inference",
    )
    parser.add_argument(
        "--use_gpu",
        type=bool,
        default=False,
        help="Use GPU for inference",
    )
    parser.add_argument(
        "--save_video",
        type=bool,
        default=False,
        help="Save the output video",
    )
    
    args = parser.parse_args()

    run_model(args.arc, args.run_name, args.source, args.video_path, args.use_gpu, args.save_video)

# example usage:
# python run_model.py --arc ImageReducer_bounded_grayscale --run_name run_2 --source video --video_path samples/underwater_drone_sample.mp4 --use_gpu True --save_video True