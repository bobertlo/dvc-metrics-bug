#!/usr/bin/env python3
import json
import os
import sys
import yaml

import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
import torch
from torch.utils.data import Dataset

from cnn.architectures import Architectures
from cnn.training import Trainer


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


def preprocess(path, fileset, arch):
    images = []
    for img_name in tqdm(fileset["image"]):
        image_path = os.path.join(path, img_name)
        img = cv2.imread(image_path)
        if img is None:
            print("error")
        img = arch.convert_np(img)
        images.append(img)

    X = np.array(images)
    y = fileset["label"].values
    return X, y


def load_data_directory(path, arch):
    train_df = pd.read_csv(os.path.join(path, "train.csv"))
    test_df = pd.read_csv(os.path.join(path, "test.csv"))

    train_df["label_name"] = pd.Categorical(train_df["label"])
    train_df["label"] = train_df.label_name.cat.codes

    test_df["label_name"] = pd.Categorical(test_df["label"])
    test_df["label"] = test_df.label_name.cat.codes

    labels = list(train_df.label_name.cat.categories)

    train_x, train_y = preprocess(path, train_df, arch)
    val_x, val_y = preprocess(path, test_df, arch)

    train_x = train_x.reshape(len(train_x), arch.shape[0], arch.shape[1], arch.shape[2])
    train_x = torch.from_numpy(train_x)

    train_y = train_y.astype(int)
    train_y = torch.from_numpy(train_y)

    val_x = val_x.reshape(len(val_x), arch.shape[0], arch.shape[1], arch.shape[2])
    val_x = torch.from_numpy(val_x)

    val_y = val_y.astype(int)
    val_y = torch.from_numpy(val_y)

    train_data = TensorDataset(train_x, train_y, labels=labels, df=train_df, path=path)
    val_data = TensorDataset(val_x, val_y, labels=labels, df=test_df, path=path)

    return train_data, val_data, labels


def train_model(datapath, arch, bs, stages):
    last_lr = 0
    lr_set = False

    print("Loading Training Data:")
    train_data, val_data, _ = load_data_directory(datapath, arch)
    trainer = Trainer(train_data, val_data, arch, batchsize=bs)

    for s in stages:
        if not lr_set:
            lr = s.get("lr", 0.0003)
            last_lr = lr
            lr_set = True
            trainer.set_learningrate(lr)
        else:
            lr = s.get("lr", last_lr)
            if lr != last_lr:
                trainer.set_learningrate(lr)
                last_lr = lr
        epochs = s.get("n", None)
        print("Training", epochs, "epochs @", lr)
        if epochs is None:
            sys.stderr.write("Invalid epoch count. Aborting\n")
            sys.exit(1)
        trainer.train_epochs(epochs)

    return trainer


def make_sets(set_labels, class_labels):
    labelmap = {}
    for i, l in enumerate(class_labels):
        labelmap[l] = i
    sets = {}

    for k in set_labels:
        sets[k] = [labelmap[i] for i in set_labels[k]]

    return sets


def save_model(trainer, outpath, sets):
    os.makedirs(outpath, exist_ok=True)

    weightpath = os.path.join(outpath, "weights.pt")
    print("saving", weightpath)
    torch.save(trainer.model.state_dict(), weightpath)

    labelpath = os.path.join(outpath, "labels.json")
    print("saving", labelpath)
    with open(labelpath, "w") as json_file:
        json.dump(trainer.labels, json_file, sort_keys=True, indent=1)

    setpath = os.path.join(outpath, "sets.json")
    print("saving", setpath)
    with open(setpath, "w") as json_file:
        json.dump(sets, json_file, sort_keys=True, indent=1)


def run(model_name):
    ydata = yaml.safe_load(open("params.yaml"))

    m = ydata.get(model_name)
    if not m:
        sys.stderr.write("Invalid model: " + model_name + " - aborting\n")
        sys.exit(1)

    m_datapath = m["data"]["genpath"]
    m_stages = m["training"]["stages"]
    m_outpath = m["training"]["outpath"]
    m_set_labels = m["training"].get("sets", {})
    m_cvt_gray = m["training"].get("cvt_gray", "False")
    m_bs = m["training"].get("batchsize", 250)
    m_model = Architectures[m["arch"]]

    if m_cvt_gray in ["False", "false", "No", "no"]:
        m_cvt_gray = False
    else:
        m_cvt_gray = True

    trained_model = train_model(m_datapath, m_model, m_bs, m_stages)
    sets = make_sets(m_set_labels, trained_model.labels)

    save_model(trained_model, m_outpath, sets)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython prepare.py model-repro.yml\n")
        sys.exit(1)

    run(sys.argv[1])
