import torch
import torch.nn as nn


class RegressionNeuralNet(nn.Module):
    def __init__(self, input_size):
        super(RegressionNeuralNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, 1)
        )

    def forward(self, x):

        return self.model(x)

    def find_device(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        return device
