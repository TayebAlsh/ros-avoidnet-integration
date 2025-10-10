import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from dataset import SUIM, SUIM_grayscale
import argparse
from avoid_net import get_model
from torchvision.models.segmentation import lraspp_mobilenet_v3_large


def progress_bar(curr_epoch, loss, num_epochs, data_len, batch_size, bar_length):
    percent = curr_epoch / num_epochs * 100
    arrow = "-" * int(percent / 100 * bar_length - 1) + ">"
    spaces = " " * (bar_length - len(arrow))
    print(
        f"Epoch: [{curr_epoch + 1}/{num_epochs}], Loss: {loss:.4f}, Progress: [{arrow + spaces}] {percent:.2f} %",
        end="\r",
    )


def train_model(batch_size, num_epochs, arc, run_name):
    print(
        f"Training {arc}_{run_name} for {num_epochs} epochs with batch size {batch_size}"
    )
    # import the model
    model = lraspp_mobilenet_v3_large(num_classes=1)
    
    
    class custom_backbone(nn.Module):
        def __init__(self):
            super(custom_backbone, self).__init__()
            self.model = lraspp_mobilenet_v3_large(num_classes=1)
            # Modify the first conv layer to accept 1 channel instead of 3
            self.model.backbone.low_level_features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)

        def forward(self, x):
            return self.model(x)

    model = custom_backbone()
    
    # print the first layer of the model
    print(model)
    

    # Prepare your own dataset
    dataset = SUIM_grayscale("/media/ali/New Volume/Datasets/train_val")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define the loss function and the optimizer for segmentation
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    model.to(device)

    # Train the model
    current_loss = 100000000
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in dataloader:
            images, masks = batch
            # move tensors to device
            images = images.to(device)
            masks = masks.to(device)
            # Forward pass
            outputs = model(images)
            # normalize the output to be between 0 and 1
            # Calculate the loss
            outputs = nn.functional.interpolate(outputs["out"], size=(32, 32), mode="bilinear")
            loss = criterion(outputs, masks)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}", end="\r"
            )
            running_loss += loss.item()
            progress_bar(epoch, loss.item(), num_epochs, len(dataset), batch_size, 20)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], RUNNING Loss: {running_loss / len(dataloader):.4f}"
        )
        # make a checkpoint after each epoch
        if current_loss > running_loss / len(dataloader):
            current_loss = running_loss / len(dataloader)
            torch.save(model.state_dict(), f"models/{arc}_{run_name}.pth")
            # output running loss and epoch into a txt file
            with open(f"models/{run_name}.txt", "a") as f:
                f.write(
                    f"Epoch [{epoch + 1}/{num_epochs}], RUNNING Loss: {running_loss / len(dataloader):.4f}\n"
                )
            print("saving checkpoint, epoch: ", epoch + 1)


if __name__ == "__main__":
    # get args from comando line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size to be used for training",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of epochs to train the model",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="run_1",
        help="Name of the run to be used to save the model",
    )
    parser.add_argument(
        "--arc",
        type=str,
        default="ImageReducer",
        help="Architecture of the model",
    )
    args = parser.parse_args()

    # train the model
    train_model(args.batch_size, args.num_epochs, args.arc, args.run_name)

# example usage:
# python train_mobilenet.py --batch_size 64 --num_epochs 10 --run_name run_1 --arc lr_mobilenet_v3_large
