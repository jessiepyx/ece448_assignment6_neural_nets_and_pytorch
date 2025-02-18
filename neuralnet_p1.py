# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019

"""
You should only modify code within this file for part 1 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class NeuralNet(nn.Module):
    def __init__(self, lrate, in_size, out_size):
        """
        Initialize the layers of your neural network
        @param lrate: The learning rate for the model.
        @param loss_fn: The loss function
        @param in_size: Dimension of input
        @param out_size: Dimension of output
        The network should have the following architecture (in terms of hidden units):
        in_size -> 128 ->  out_size
        """
        super(NeuralNet, self).__init__()
        self.lrate = lrate
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = 128

        self.features = nn.Sequential(
            nn.Linear(self.in_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.out_size),
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.features.parameters(), self.lrate)

    def get_parameters(self):
        """ Get the parameters of your network
        @return params: a list of tensors containing all parameters of the network
        """
        return self.features.parameters()

    def forward(self, x):
        """ A forward pass of your autoencoder
        @param x: an (N, in_size) torch tensor
        @return y: an (N, out_size) torch tensor of output from the network
        """
        return self.features(x)

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y
        @param x: an (N, in_size) torch tensor
        @param y: an (N,) torch tensor
        @return L: total empirical risk (mean of losses) at this time step as a float
        """
        self.optimizer.zero_grad()

        # forward
        y_hat = self.forward(x)

        # loss
        loss = self.criterion(y_hat, y)
        L = loss.item()

        # backward
        loss.backward()
        self.optimizer.step()

        return L


def fit(train_set, train_labels, dev_set, n_iter, batch_size=100):
    """ Fit a neural net.  Use the full batch size.
    @param train_set: an (N, in_size) torch tensor
    @param train_labels: an (N,) torch tensor
    @param dev_set: an (M, in_size) torch tensor
    @param n_iter: int, the number of batches to go through during training (not epoches)
                   when n_iter is small, only part of train_set will be used, which is OK,
                   meant to reduce runtime on autograder.
    @param batch_size: The size of each batch to train on.
    # return all of these:
    @return losses: list of total loss (as type float) at the beginning and after each iteration. Ensure len(losses) == n_iter
    @return yhats: an (M,) NumPy array of approximations to labels for dev_set
    @return net: A NeuralNet object
    # NOTE: This must work for arbitrary M and N
    """
    # standardize data, assuming std != 0
    train_mean = train_set.mean(dim=1, keepdim=True)
    train_std = train_set.std(dim=1, keepdim=True)
    train_set = (train_set - train_mean) / train_std

    dev_mean = dev_set.mean(dim=1, keepdim=True)
    dev_std = dev_set.std(dim=1, keepdim=True)
    dev_set = (dev_set - dev_mean) / dev_std

    # train
    lrate = 0.01
    N = train_set.shape[0]
    in_size = train_set.shape[1]
    net = NeuralNet(lrate, in_size, 3)
    losses = []
    for batch in range(n_iter):
        # get batch data
        idx = torch.randperm(N)
        x_batch = train_set[idx[:batch_size]]
        y_batch = train_labels[idx[:batch_size]]

        loss = net.step(x_batch, y_batch)
        losses.append(loss)

    # develop
    yhats = np.argmax(net.forward(dev_set).detach().numpy(), axis=1)

    return losses, yhats, net
