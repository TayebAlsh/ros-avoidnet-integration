import os
from PIL import Image
import argparse


def progress_bar(current, total, barLength=20):
    percent = float(current) * 100 / total
    arrow = "-" * int(percent / 100 * barLength - 1) + ">"
    spaces = " " * (barLength - len(arrow))
    print(f"Progress: [{arrow + spaces}] {percent:.2f} %", end="\r")


def generate_grid(grid_size, threshold):
    masks_folder = "/media/ali/New Volume/Datasets/train_val/masks"
    output_folder = f"/media/ali/New Volume/Datasets/train_val/masks_grided_{grid_size}_threshold_{threshold}"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # read all the masks
    masks = os.listdir(masks_folder)
    masks_proccessed = 0
    for mask in masks:
        # only process .bmp files
        if mask.split(".")[-1] != "bmp":
            continue
        mask_path = os.path.join(masks_folder, mask)
        mask = Image.open(mask_path)

        mask = mask.convert("L")
        mask = mask.point(lambda x: 255 if x != 0 else 0)

        # resize the mask to grid_size x grid_size
        mask = mask.resize((grid_size, grid_size), Image.NEAREST)

        # save the mask
        mask.save(os.path.join(output_folder, mask_path.split("/")[-1]))

        # print a progress bar
        progress_bar(masks_proccessed, len(masks))

        masks_proccessed += 1


# main function
if __name__ == "__main__":
    # get args from comando line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grid_size",
        type=int,
        default=32,
        help="Size of the grid to be generated",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold to be used to generate the grid",
    )
    args = parser.parse_args()
    
    # generate the grid
    generate_grid(args.grid_size, args.threshold)