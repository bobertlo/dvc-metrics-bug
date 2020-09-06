import json
import os

import numpy as np
import torch


class Network:
    def __init__(
        self, model, path, loadweights=True, map_location="cpu",
    ):
        self.path = path
        self.__labelpath = os.path.join(path, "labels.json")

        with open(self.__labelpath, "r") as json_file:
            self.labels = json.load(json_file)

        self.lmap = {}
        for i, l in enumerate(self.labels):
            self.lmap[l] = i

        self.model = model(n_labels=len(self.labels))
        self.model.eval()

        if loadweights:
            self.__weightpath = os.path.join(path, "weights.pt")
            self.model.load_state_dict(
                torch.load(self.__weightpath, map_location=map_location)
            )

        self.sets = {}
        self.__setspath = os.path.join(path, "sets.json")
        if os.path.exists(self.__setspath):
            with open(self.__setspath, "r") as json_file:
                self.sets = json.load(json_file)

    def predict_verbose_data(self, dat):
        output = self.model(dat).detach()
        out_class = np.argmax(output)
        out_label = self.labels[out_class]
        return out_label, out_class, output

    def predict_verbose(self, frame, args=[0]):
        return self.predict_verbose_data(self.prepare(frame, args=args))

    def predict_data(self, dat):
        return np.argmax(self.model(dat).detach()).item()

    def predict(self, frame, args=[0]):
        return self.predict_data(self.prepare(frame, args=args))

    def prepare(self, img, args=[0]):
        return self.model.prepare(img, args=args)

    def convert(self, data):
        return self.model.convert(data)
