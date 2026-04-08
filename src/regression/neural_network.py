import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from configs.config import PLOTS_PATH


class RegressionNeuralNet(nn.Module):
    def __init__(self, input_size=14):

        super(RegressionNeuralNet, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 25),
            nn.ReLU(),
            nn.Linear(25, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
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

    def plot_losses(self, losses, filename):

        filepath = PLOTS_PATH / 'regressions/neural-net' / filename

        plt.figure()
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.xticks(range(0, len(losses)+1, 5))
        plt.title("Loss Curve")
        plt.ylim(0)

        plt.savefig(filepath, bbox_inches="tight", dpi=300)
        plt.close()

    def plot_learning_curve(self, n_epochs, train_metrics, val_metrics):

        filepath = PLOTS_PATH / 'regressions/neural-net/learning_curve.png'

        plt.figure()
        plt.plot(np.arange(n_epochs) + 0.5, train_metrics, ".--",
                 label="Training")
        plt.plot(np.arange(n_epochs) + 1.0, val_metrics, ".-",
                 label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.grid()
        plt.title("Learning curves")
        plt.axis([0, 50, 0, 70])
        plt.legend()

        plt.savefig(filepath, bbox_inches="tight", dpi=300)
        plt.close()
