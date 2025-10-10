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


def test(batch_size, num_avg, arc, run_name, use_gpu=False):

    model = get_model(arc)
    model.load_state_dict(torch.load(f"models/{arc}_{run_name}.pth"))

    # Prepare your own dataset
    dataset = SUIM_grayscale("/media/ali/New Volume/Datasets/TEST")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    print(f"Testing on: {device}")
    model.to(device)
    model.eval()

    # load images and masks
    images, masks = next(iter(dataloader))
    images = images.to(device)

    # forward pass
    tic = time.time()
    for i in range(num_avg):
        outputs = model(images)
        i += 1
    toc = time.time()
    print(f"-- Inference time: ", ((toc - tic) / num_avg) / batch_size)

    # forward pass
    outputs = model(images)

    # move the output tensors to cpu for visualization
    outputs = outputs.detach().cpu()
    images = images.cpu()
    # print(outputs)
    print("outputs shape ", outputs.shape)

    # denormalize the images and masks
    images = images * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images = images + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)

    # apply threshold to output to get binary mask
    # indices above threshold are 1 and below are 0
    threshold = 0.1
    best_TP = 99999
    best_FN = 99999
    MSE = nn.MSELoss()
    print("length of outputs ", len(outputs))
    for threshold in range(1, 9, 1):
        # any values inside each output mask should be 1 if it is above the threshold
        threshold = threshold / 10
        temp_outputs = outputs.clone()
        for i in range(len(temp_outputs)):
            temp_outputs[i] = torch.where(temp_outputs[i] > threshold, torch.tensor(1.0), torch.tensor(0.0))

        # calculate the false positive and false negative
        # Calculate true positives (TP): Both predicted and actual are 1
        TP = (temp_outputs * masks).sum().item()
        
        # Calculate false positives (FP): Actual is 0 but predicted is 1
        FP = ((1 - masks) * temp_outputs).sum().item()
        
        # Calculate false negatives (FN): Actual is 1 but predicted is 0
        FN = ((1 - temp_outputs) * masks).sum().item()
        
        # Calculate true negatives (TN): Both predicted and actual are 0
        TN = ((1 - temp_outputs) * (1 - masks)).sum().item()

        # Calculate the total number of positives in the true tensor
        total_positives = masks.sum().item()

        # Calculate the total number of negatives in the true tensor
        total_negatives = (1 - masks).sum().item()
        
        # Calculate the percentage of true positives
        TP_percentage = (TP / total_positives) * 100
        
        # Calculate the percentage of false negatives
        FN_percentage = (FN / total_positives) * 100
        
        # Calculate the percentage of false positives
        FP_percentage = (FP / total_negatives) * 100
        
        # Calculate the percentage of true negatives
        TN_percentage = (TN / total_negatives) * 100
        
        
        # print the statistics
        print(f"=== Threshold: {threshold} ===")
        print(f"-- True Positive: ({TP_percentage}%)")
        print(f"-- False Positive: ({FP_percentage}%)")
        print(f"-- False Negative: ({FN_percentage}%)")
        print(f"-- True Negative: ({TN_percentage}%)")
        print(f"-- MSE: {MSE(temp_outputs, masks)}")
        print(f"-- F1 Score: {2 * TP / (2 * TP + FP + FN)}")
        print(f"-- Precision: {TP / (TP + FP)}")
        print(f"-- Recall: {TP / (TP + FN)}")
        print(f"-- Accuracy: {(TP + TN) / (TP + TN + FP + FN)}")
        print(f"-- Sensitivity: {TP / (TP + FN)}")


    # print("=== General Statistics ===")
    # # calculate the false positive and false negative
    # TP = torch.sum(outputs * masks)
    # FP = torch.sum(outputs * (1 - masks))
    # FN = torch.sum((1 - outputs) * masks)
    # TN = torch.sum((1 - outputs) * (1 - masks))

    # # convvert to precentage
    # TP = TP / (TP + FP + FN + TN)
    # FP = FP / (TP + FP + FN + TN)
    # FN = FN / (TP + FP + FN + TN)
    # TN = TN / (TP + FP + FN + TN)

    # print(f"-- True Positive: {TP}")
    # print(f"-- False Positive: {FP}")
    # print(f"-- False Negative: {FN}")
    # print(f"-- True Negative: {TN}")

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
        plt.imsave(f"results/{arc}_{run_name}_{i}_mask.png", mask_np.squeeze(), cmap='gray')
        ax[i, 2].imshow(output_np)
        ax[i, 2].set_title("Prediction")
        plt.imsave(f"results/{arc}_{run_name}_{i}.png", output_np.squeeze(), cmap='gray')
    # plt.show()


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
    parser.add_argument(
        "--use_gpu",
        type=bool,
        default=False,
        help="Use GPU for testing",
    )
    args = parser.parse_args()

    # test the model
    test(args.batch_size, args.num_average, args.arc, args.run_name, args.use_gpu)

# example usage
# python test.py --batch_size 300 --num_average 1 --run_name run_2 --arc ImageReducer_bounded_grayscale --use_gpu True
