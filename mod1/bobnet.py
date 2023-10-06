import torch.nn as nn
import torch.nn.functional as F

class BobNet(nn.Module):

    def __init__(self):

        super().__init__()

        # define the two fully-connected layers
        self.fc1 = nn.Linear(784, 800)
        self.fc2 = nn.Linear(800, 10)

    def forward(self, x):

        # do a forward pass and ReLU activation on the first layer
        x = self.fc1(x)
        x = F.relu(x)

        # do a forward pass and softmax activation on the output layer
        x = self.fc2(x)
        x = F.softmax(x, dim=1)

        return x
    