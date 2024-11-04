import torch
import torch.nn as nn


class ImageReducer(nn.Module):
    def __init__(self):
        super(ImageReducer, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))

        return x


class ImageReducer_bounded(nn.Module):
    def __init__(self):
        super(ImageReducer_bounded, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=0)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        self.conv5 = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))

        # Normalize between 0 and 1
        x = x / torch.max(x)
        return x


class ImageReducer_bounded_grayscale(nn.Module):
    def __init__(self):
        super(ImageReducer_bounded_grayscale, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=0)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        self.conv5 = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))

        # Normalize between 0 and 1
        x = x / torch.max(x)
        return x


def get_model(arc):
    if arc == "ImageReducer":
        return ImageReducer()
    elif arc == "ImageReducer_bounded":
        return ImageReducer_bounded()
    elif arc == "ImageReducer_bounded_grayscale":
        return ImageReducer_bounded_grayscale()
    else:
        raise NotImplementedError(f"{arc} is not implemented yet")
