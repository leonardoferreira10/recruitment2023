
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
:param training_loader: dataloader to use during training  
:param validation_loader: dataloader to use during validation
:param logging_interval: how many batches to run between tensorboard loss logs

:return best average validation loss
"""
def train_model(model: torch.nn.Module, epochs: int, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module,
                training_loader: torch.utils.data.DataLoader, validation_loader: torch.utils.data.DataLoader,
                logging_interval: int = 100) -> float:

    # detect if the device has cuda acceleration
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # move the model to the device
    model.to(device)

    # create the tensorboard writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter("runs/mnist_mlp_{}".format(timestamp))

    best_val_loss = 1000000000

    for epoch in range(epochs):

        epoch_train_loss, epoch_val_loss = run_single_epoch(epoch, model, optimizer, loss_fn, training_loader, validation_loader, writer,
                                      logging_interval, device)

        # compute the average validation loss
        avg_val_loss = epoch_val_loss / len(validation_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

    return best_val_loss


r""" Runs a single training epoch

:param: epoch: int
:param model: model to train
:param optim: optimizer to use
:param loss_fn: loss function to use
:param training_loader: dataloader to use during training
:param validation_loader: dataloader to use during validation 
:param writer: tensorboard writer to run logs with
:param logging_interval: how many batches to run between tensorboard logs

:returns a tuple containing the total training and average loss
"""
def run_single_epoch(epoch: int, model: torch.nn.Module, optim: torch.optim.Optimizer, loss_fn: torch.nn.Module,
                     training_loader: torch.utils.data.DataLoader, validation_loader: torch.utils.data.DataLoader,
                     writer: SummaryWriter, logging_interval: int = 100, device=torch.device("cpu")) -> (float, float):

    total_train_loss = 0.

    # set the model to training mode
    model.train()

    # iterate the training set batches
    for i, (inputs, labels) in enumerate(training_loader):

        # copy the inputs and labels to the device
        inputs, labels = inputs.to(device), labels.to(device)

        # transfer the data to the device
        inputs.to(device)
        labels.to(device)

        # reset the optimizer gradients
        optim.zero_grad()

        # do the forward pass
        outputs = model(inputs)

        # compute the loss comparing the outputs to the dataset labels
        loss = loss_fn(outputs, labels)
        loss.backward()

        # adjust the weights
        optim.step()

        # add to the total training loss
        total_train_loss += loss.item()

        # write the logs
        if (i+1) % logging_interval == 0:
            # compute the average training loss
            avg_train_loss = total_train_loss / i
            writer.add_scalar("training_loss", avg_train_loss, i)

            print("Epoch {} ({}/{}): training_loss = {}".format(epoch, i, len(training_loader), avg_train_loss))


    # -- EVALUATION --

    total_val_loss = 0.

    # set the model to evaluation mode (no backprop)
    model.eval()

    with torch.no_grad():

        # iterate the validation set
        for i, (vinputs, vlabels) in enumerate(validation_loader):
            vinputs, vlabels = vinputs.to(device), vlabels.to(device)
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)

            total_val_loss += vloss

            # write the logs
            if (i+1) % logging_interval == 0:
                # compute the average validation loss
                avg_val_loss = total_val_loss / i
                writer.add_scalar("validation_loss", avg_val_loss, i)

                print("Epoch {} ({}/{}): validation_loss = {}".format(epoch, i, len(validation_loader), avg_val_loss))

    return total_train_loss, total_val_loss
