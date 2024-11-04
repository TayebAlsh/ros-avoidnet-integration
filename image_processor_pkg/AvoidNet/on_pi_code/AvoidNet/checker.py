import torch
torch.backends.mkldnn.enabled = False
print(torch.__config__.show())