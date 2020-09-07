import torch
from torch.nn import (
    Linear,
    ReLU,
    Sequential,
    Conv2d,
    MaxPool2d,
    Module,
    BatchNorm2d,
    Dropout,
)
import numpy as np
import cv2


def crop(img, c1, c2):
    return img[c1[0] : c2[0], c1[1] : c2[1]]


def crop_dbox1(img, args=[0]):
    n = args[0]
    firstbox = np.array([[98, 728], [126, 768]])
    offset = np.array([[28, 0], [28, 0]])
    c = firstbox + (n * offset)
    box = crop(img, c[0], c[1])

    return box


class Dbox1(Module):
    shape = [1, 40, 28]

    def __init__(self, n_labels=43):
        super(Dbox1, self).__init__()

        self.cnn_layers = Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            BatchNorm2d(32),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            BatchNorm2d(64),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            BatchNorm2d(128),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(1920, 256), ReLU(inplace=True), Dropout(0.2), Linear(256, n_labels),
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

    @staticmethod
    def crop(img, args=[0]):
        return crop_dbox1(img, args)

    @staticmethod
    def convert_np(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype("float32")
        img /= 255.0
        img = img.reshape(1, 1, 40, 28)
        return img

    @staticmethod
    def convert(img):
        img = Dbox1.convert_np(img)
        return torch.from_numpy(img)

    @staticmethod
    def prepare(img, args=[0]):
        return Dbox1.convert(Dbox1.crop(img, args=args))
