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
import torch
import cv2


def crop_224(img):
    img = cv2.resize(img, (224, 224))
    return img


class Test1(Module):
    shape = [3, 224, 224]

    def __init__(self, n_labels=43):
        super(Test1, self).__init__()

        self.cnn_layers = Sequential(
            Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
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
            Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            BatchNorm2d(256),
            Dropout(0.1),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            Dropout(0.1),
            BatchNorm2d(512),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(25088, 1024),
            ReLU(inplace=True),
            Dropout(0.2),
            # Linear(2048, 1024),
            # ReLU(inplace=True),
            # Dropout(0.2),
            Linear(1024, 512),
            ReLU(inplace=True),
            Dropout(0.5),
            Linear(512, n_labels),
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

    @staticmethod
    def crop(img, args=None):
        return crop_224(img)

    @staticmethod
    def convert_np(img):
        img = img.astype("float32")
        img /= 255.0
        img = img.reshape(1, 3, 224, 224)
        return img

    @staticmethod
    def convert(img):
        img = Test1.convert_np(img)
        return torch.from_numpy(img)

    @staticmethod
    def prepare(img, args=[]):
        return Test1.convert(Test1.crop(img))
