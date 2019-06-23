import torch


import os, glob
import numpy as np
from torchvision import models
from PIL import Image
def main():
    net = models.vgg16()
    net.load_state_dict(torch.load("weights/vloss0.5725-vacc0.8790-loss0.1197-acc0.9661.pth"))
    net = net.eval()
    img = Image.open("00002.jpg")
    img = img.numpy()


if __name__ == "__main__":
    main()