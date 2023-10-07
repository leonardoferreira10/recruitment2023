from torch import nn, functional as F

class CNN(nn.Module):

    def __init__(self):

        super().__init__()

        raise NotImplementedError("Define the layers here!")

    def forward(self, x):
        raise NotImplementedError("Implement the forward pass here!")