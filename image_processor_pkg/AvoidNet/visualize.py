# import the model from avoid_net.py
from avoid_net import get_model
import torch
from dataset import SUIM, SUIM_grayscale
from torch.utils.data import DataLoader
batch_size = 1

model = get_model("ImageReducer_bounded_grayscale")

# create a dummy input image
images = torch.randn(batch_size, 1, 155, 155, requires_grad=True)

input_names = ['image']
output_names = ['binary obstacle map']
torch.onnx.export(model, images, 'cnn.onnx', input_names=input_names, output_names=output_names)

