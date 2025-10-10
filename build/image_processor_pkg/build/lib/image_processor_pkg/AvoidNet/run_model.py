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
from trajectory import determain_trajectory


def run_model(arc, run_name, source, video_path=None, use_gpu=False, save_video=False):
    threshold = 0.7
    model = get_model(arc)
    model.load_state_dict(torch.load(f"models/{arc}_{run_name}.pth", map_location=torch.device('cpu')))


    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    print(f"Running on: {device}...")
    model.to(device).eval()

    dataset = SUIM_grayscale("/media/ali/New Volume/Datasets/TEST")
    image_transform = dataset.get_transform()
    mask_transform = dataset.get_mask_transform()

    # read from source using opencv
    if source == "webcam":
        cap = cv2.VideoCapture(0)
        output_name = "webcam"

    elif source == "video":
        cap = cv2.VideoCapture(video_path)
        output_name = video_path.split("/")[-1].split(".")[0]
    size = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    save_size = (1920, 1080)
    print(f"Video size: {int(size[0])}x{int(size[1])}")
    print(f"Video save_size: {save_size}")
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    if save_video:
        out = cv2.VideoWriter(
            f"results/{output_name}.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            frame_rate,
            (int(save_size[0]), int(save_size[1])),
        )

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        # prepare the frame for inference
        if size > save_size:
            frame = cv2.resize(frame, (int(save_size[0]), int(save_size[1])))
        frame_tensor = Image.fromarray(frame)
        frame_tensor = image_transform(frame_tensor).to(device).unsqueeze(0)
        outputs = model(frame_tensor)

        # move the output tensors to cpu for visualization
        outputs = outputs.detach().cpu()
        outputs = outputs[0].permute(1, 2, 0)
        outputs = np.array(outputs)  # Convert to numpy array
        # show the output
        frame = draw_red_squares(frame, outputs, threshold)
        obstacle, new_trej = determain_trajectory(outputs, threshold=threshold)
        # put text on the to pleft corner of the frame
        if obstacle:
            cv2.putText(frame, f"Obstacle!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Turn {new_trej}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            # draw an arrow in the middle of the frame to indicate the direction
            if new_trej == "left":
                cv2.arrowedLine(frame, (int(size[0] / 2), int(size[1] / 2)), (int(size[0] / 2) - 100, int(size[1] / 2)), (0, 255, 255), 24)
            elif new_trej == "right":
                cv2.arrowedLine(frame, (int(size[0] / 2), int(size[1] / 2)), (int(size[0] / 2) + 100, int(size[1] / 2)), (0, 255, 255), 24)
            elif new_trej == "up":
                cv2.arrowedLine(frame, (int(size[0] / 2), int(size[1] / 2)), (int(size[0] / 2), int(size[1] / 2) - 100), (0, 255, 255), 24)
        else:
            cv2.putText(frame, "Path Clear!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("frame", frame)
        if save_video:
            # frame = cv2.resize(frame, (int(save_size[0]), int(save_size[1])))
            out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            out.release()
            break
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
# python run_model.py --arc ImageReducer_bounded_grayscale --run_name run_2 --source video --video_path vlogs/output_2024-05-27_19-49-32.mp4 --use_gpu True --save_video True
