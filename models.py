import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################################################
#from https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
###############################################################################


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(9*20, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class NeuralNetworkBCEL(nn.Module):
    def __init__(self):
        super(NeuralNetworkBCEL, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(9*20, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return torch.sigmoid(logits)


class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()
        # Number of input features is 12.
        self.flatten = nn.Flatten()
        self.layer_1 = nn.Linear(9*20, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 2)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, inputs):
        x = self.relu(self.layer_1(self.flatten(inputs)))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x


class BinaryClassificationOneLayer(nn.Module):
    def __init__(self):
        super(BinaryClassificationOneLayer, self).__init__()
        # Number of input features is 12.
        self.flatten = nn.Flatten()
        self.layer_1 = nn.Linear(9*20, 64)
        self.layer_out = nn.Linear(64, 2)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, inputs):
        x = self.relu(self.layer_1(self.flatten(inputs)))
        x = self.batchnorm1(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x