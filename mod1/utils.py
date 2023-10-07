
# this file contains utilities to train models

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime


# receives the model, the number of epochs, optimizer and loss function and returns the average loss
""" Trains a model

:param model: model to train
:param epochs: how many epochs to run
:param optimizer: optimizer algorithm to use. can be stochastic gradient descent, adam, etc.
:param loss_fn: loss function which computes how far is the model from the expected results
:param training_dataloader: dataloader to use during training  
:param validation_dataloader: dataloader to use during validation
:param logging_interval: how many batches to run between tensorboard loss logs
"""
def train_model(model: torch.nn.Module, epochs: int, optimizer: torch.optim.optimizer, loss_fn: torch.nn.Module,
                training_dataloader: torch.utils.data.DataLoader, validation_dataloader: torch.utils.data.DataLoader,
                logging_interval: int = 100) -> float:

    # create the tensorboard writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter("runs/mnist_mlp_{}".format(timestamp))

    for epoch in range(epochs):

        run_single_epoch(model, writer, logging_interval)

    return 0.


""" Runs a single training epoch

:param model: model to train
:param writer: tensorboard writer to run logs with
:param logging_interval: how many batches to run between tensorboard logs
"""
def run_single_epoch(model: torch.nn.Module, writer: SummaryWriter, logging_interval: int = 100) -> float:
    pass
