import os
import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import Dataset
from skimage.io import imread
from skimage.color import rgb2gray
from tqdm import tqdm


class TensorDataset(Dataset):
    def __init__(self, x_tensor, y_tensor, labels=None, df=None, path=None):
        self.labels = labels
        self.df = df
        self.x = x_tensor
        self.y = y_tensor
        self.path = path

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)


def preprocess(path, fileset, cvt_gray=False):
    images = []
    for img_name in tqdm(fileset["image"]):
        image_path = os.path.join(path, img_name)
        # img = imread(image_path)
        img = cv2.imread(image_path)
        if img is None:
            print("error")
        if cvt_gray:
            # img = rgb2gray(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype("float32")
        img /= 255.0
        images.append(img)

    X = np.array(images)
    y = fileset["label"].values
    return X, y


def load_data_directory(path, cvt_gray=False):
    train_df = pd.read_csv(os.path.join(path, "train.csv"))
    test_df = pd.read_csv(os.path.join(path, "test.csv"))

    train_df["label_name"] = pd.Categorical(train_df["label"])
    train_df["label"] = train_df.label_name.cat.codes

    test_df["label_name"] = pd.Categorical(test_df["label"])
    test_df["label"] = test_df.label_name.cat.codes

    labels = list(train_df.label_name.cat.categories)

    train_x, train_y = preprocess(path, train_df, cvt_gray=cvt_gray)
    val_x, val_y = preprocess(path, test_df, cvt_gray=cvt_gray)

    dims = train_x.shape[1:]
    if len(dims) == 2:
        dims = list(dims) + [1]

    train_x = train_x.reshape(len(train_x), dims[2], dims[1], dims[0])
    train_x = torch.from_numpy(train_x)

    train_y = train_y.astype(int)
    train_y = torch.from_numpy(train_y)

    val_x = val_x.reshape(len(val_x), dims[2], dims[1], dims[0])
    val_x = torch.from_numpy(val_x)

    val_y = val_y.astype(int)
    val_y = torch.from_numpy(val_y)

    train_data = TensorDataset(train_x, train_y, labels=labels, df=train_df, path=path)
    val_data = TensorDataset(val_x, val_y, labels=labels, df=test_df, path=path)

    return train_data, val_data, labels
