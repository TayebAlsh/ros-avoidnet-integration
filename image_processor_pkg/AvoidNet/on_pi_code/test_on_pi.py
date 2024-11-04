import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
from torch import nn, optim
import matplotlib.pyplot as plt
from dataset import SUIM, SUIM_grayscale
import time
from avoid_net import get_model
import argparse
from torch.utils import mkldnn as mkldnn

# Disable MKL-DNN backend
torch.backends.mkldnn.enabled = False

torch.set_num_threads(4)

print(f"MKL-DNN status: {torch.backends.mkldnn.enabled}, number of threads: {torch.get_num_threads()}")

def test(batch_size, num_avg, arc, run_name):
    model = get_model(arc)
    model.load_state_dict(
        torch.load(f"models/{arc}_{run_name}.pth", map_location=torch.device("cpu"))
    )
    


    # Prepare your own dataset
    dataset = SUIM_grayscale("TEST")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cpu")
    print(f"Testing on: {device}")
    model.to(device)
    model.eval()

    # load images and masks
    images, masks = next(iter(dataloader))
    images = images.to(device)
    print(f"images shapes {images.shape}")

    # forward pass
    tic = time.time()
    for i in range(num_avg):
        outputs = model(images)
        i += 1
    toc = time.time()
    inf_time = ((toc - tic) / num_avg) / batch_size
    print(f"-- Inference time: {inf_time:.4f} s/image || {1/inf_time:.4f} fps")

    # forward pass
    outputs = model(images)

    # move the output tensors to cpu for visualization
    outputs = outputs.detach().cpu()
    images = images.cpu()
    print("outputs shape ", outputs.shape)

    # denormalize the images and masks
    images = images * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images = images + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)

    # calculate mask predictions vs ground truth
    MSE = nn.MSELoss()
    MSE_loss = MSE(outputs, masks)
    print(f"-- MSE loss: {MSE_loss}")

    # # show the first batch of images and masks and predictions side by side on the same figure

    # show all of the batch images and masks on the same figure
    fig, ax = plt.subplots(batch_size, 3, figsize=(15, 15))
    for i in range(batch_size):
        image_np = images[i].permute(1, 2, 0).numpy()
        mask_np = masks[i].permute(1, 2, 0).numpy()
        output_np = outputs[i].permute(1, 2, 0).numpy()

        ax[i, 0].imshow(image_np)
        ax[i, 0].set_title("Image")
        plt.imsave(f"results/{arc}_{run_name}_{i}_image.png", image_np)
        ax[i, 1].imshow(mask_np)
        ax[i, 1].set_title("Mask")
        plt.imsave(
            f"results/{arc}_{run_name}_{i}_mask.png", mask_np.squeeze(), cmap="gray"
        )
        ax[i, 2].imshow(output_np)
        ax[i, 2].set_title("Prediction")
        plt.imsave(
            f"results/{arc}_{run_name}_{i}.png", output_np.squeeze(), cmap="gray"
        )
    plt.savefig(f"results/figure_{arc}_{run_name}.png")


if __name__ == "__main__":
    # get args from command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--num_average",
        type=int,
        default=100,
        help="Number of average runs for speed calculation",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="test",
        help="Name of the run",
    )
    parser.add_argument(
        "--arc",
        type=str,
        default="ImageReducer",
        help="Name of the model architecture",
    )
    args = parser.parse_args()

    # test the model
    test(args.batch_size, args.num_average, args.arc, args.run_name)

# example usage
# python test_on_pi.py --batch_size 4 --num_average 100 --run_name run_2 --arc ImageReducer_bounded_grayscale