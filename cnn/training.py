import os

from matplotlib import pyplot as plt
import numpy as np
import pylab as pl
from sklearn.metrics import confusion_matrix
import torch
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from skimage.io import imread

# from IPython import display
from tqdm import tqdm

import itertools


class Trainer:
    def __init__(self, train_data, val_data, model, lr=0.003, batchsize=100):
        self.batchsize = batchsize
        self.train_data = train_data
        self.val_data = val_data
        self.labels = train_data.labels

        self.train_loader = DataLoader(
            dataset=train_data, batch_size=batchsize, shuffle=True
        )
        self.val_loader = DataLoader(
            dataset=val_data, batch_size=batchsize, shuffle=True
        )

        self.model = model(len(train_data.labels))

        self.clear_training_logs()
        self.set_learningrate(lr)
        self.criterion = CrossEntropyLoss()

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

    def set_learningrate(self, lr):
        self.lr = lr
        self.optimizer = Adam(self.model.parameters(), lr=lr)

    def clear_training_logs(self):
        # empty list to store training losses
        self.train_losses = []
        self.train_accuracies = []
        # empty list to store validation losses
        self.val_losses = []
        self.val_accuracies = []

    def train_epochs(self, n_epochs, lr=None):
        if lr is None:
            lr = self.lr

        for epoch in range(n_epochs):

            # training
            self.model.train()
            status = tqdm(total=len(self.train_loader), desc="Epoch: " + str(epoch))
            self.model.train()
            batch_losses = []
            batch_accuracies = []
            for batch_idx, (x_train, y_train) in enumerate(self.train_loader):
                if torch.cuda.is_available():
                    x_train = x_train.cuda()
                    y_train = y_train.cuda()
                x_train = Variable(x_train)
                y_train = Variable(y_train)

                # clearing the Gradients of the model parameters
                self.optimizer.zero_grad()
                # prediction for training and validation set
                output_train = self.model(x_train)
                # computing the training and validation loss
                loss_train = self.criterion(output_train, y_train)

                results_train = torch.softmax(output_train, dim=1).argmax(dim=1)
                batch_accuracy = (results_train == y_train).sum().float() / float(
                    y_train.size(0)
                )
                batch_accuracies.append(batch_accuracy.item())

                # computing the updated weights of all the model parameters
                loss_train.backward()
                self.optimizer.step()
                batch_losses.append(loss_train.item())
                status.update(1)
                status.set_postfix_str(loss_train.item())

            train_mean = np.mean(batch_losses)
            self.train_losses.extend(batch_losses)
            train_accuracy_mean = np.mean(batch_accuracies)
            self.train_accuracies.extend(batch_accuracies)

            # testing
            self.model.eval()
            status.reset(total=len(self.val_loader))
            batch_losses = []
            batch_accuracies = []
            with torch.no_grad():
                for x_val, y_val in self.val_loader:
                    if torch.cuda.is_available():
                        x_val = x_val.cuda()
                        y_val = y_val.cuda()

                    x_val = Variable(x_val)
                    y_val = Variable(y_val)

                    self.model.eval()
                    output_val = self.model(x_val)
                    loss_val = self.criterion(output_val, y_val)
                    batch_losses.append(loss_val.item())

                    results_val = torch.softmax(output_val, dim=1).argmax(dim=1)
                    batch_accuracy = (results_val == y_val).sum().float() / float(
                        y_val.size(0)
                    )
                    batch_accuracies.append(batch_accuracy.item())
                    status.update(1)
                    status.set_postfix_str(loss_val.item())

            status.close()
            self.val_losses.extend(batch_losses)
            self.val_accuracies.extend(batch_accuracies)
            val_mean = np.mean(batch_losses)
            val_accuracy_mean = np.mean(batch_accuracies)

            # pl.plot(self.train_losses)
            # pl.xlabel("batches")
            # pl.ylabel("training loss")
            # display.clear_output(wait=True)
            # display.display(pl.gcf())

            print(train_mean, val_mean, train_accuracy_mean, val_accuracy_mean)

        # display.clear_output(wait=True)
        # print(train_mean, val_mean, train_accuracy_mean, val_accuracy_mean)

    def eval(self):
        return Eval(self)


class Eval:
    def __init__(self, trainer):
        self.trainer = trainer
        eval_loader = DataLoader(
            dataset=trainer.val_data, batch_size=trainer.batchsize, shuffle=False
        )
        values = []
        outputs = []
        with torch.no_grad():
            for x_val, y_val in tqdm(eval_loader):
                if torch.cuda.is_available():
                    x_val = x_val.cuda()
                    y_val = y_val.cuda()

                x_val = Variable(x_val)
                y_val = Variable(y_val)

                trainer.model.eval()
                output_val = trainer.model(x_val)

                results_val = torch.softmax(output_val, dim=1).argmax(dim=1)
                outputs.extend(results_val)
                values.extend(y_val)

        self.values = [x.item() for x in values]
        self.outputs = [x.item() for x in outputs]

        self.failures = []
        for i in range(len(values)):
            if values[i] != outputs[i]:
                self.failures.append((i, values[i].item(), outputs[i].item()))

    def confusion_matrix(self):
        return confusion_matrix(self.values, self.outputs)

    def plot_confusion_matrix(
        self, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
    ):
        cm = confusion_matrix(self.values, self.outputs)
        classes = self.trainer.labels

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        plt.imshow(cm, interpolation="nearest", cmap=cmap)
        plt.title(title)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = ".2f" if normalize else "d"
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")

    def preview(self, failure):
        img = imread(
            os.path.join(
                self.trainer.val_data.path, self.trainer.val_data.df.image[failure[0]]
            )
        )
        plt.imshow(img)
        plt.title(
            self.trainer.labels[failure[2]]
            + " (is:"
            + self.trainer.labels[failure[1]]
            + ")"
        )
