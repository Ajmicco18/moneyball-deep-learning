import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from configs.config import PLOTS_PATH


class ClassificationNeuralNet(nn.Module):
    def __init__(self, input_size=12, num_classes=1):

        super(ClassificationNeuralNet, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):

        return self.model(x)

     # ****FUNCTION TO PLOT AND SAVE TRAINING LOSSES****
    def plot_losses(self, losses):

        filepath = PLOTS_PATH / 'classifications/neural_net/training_loss.png'

        plt.figure()
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.xticks(range(0, len(losses)+1, 5))
        plt.title("Loss Curve")
        plt.ylim(0)

        plt.savefig(filepath, bbox_inches="tight", dpi=300)
        plt.close()

    # ****FUNCTION TO PLOT AND SAVE MODEL'S LEARNING CURVE****
    def plot_learning_curve(self, n_epochs, train_metrics, val_metrics):

        filepath = PLOTS_PATH / 'classifications/neural_net/learning_curve.png'

        plt.figure()
        plt.plot(np.arange(n_epochs) + 0.5, train_metrics, ".--",
                 label="Training")
        plt.plot(np.arange(n_epochs) + 1.0, val_metrics, ".-",
                 label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid()
        plt.title("Learning curves")
        plt.axis([0, 50, 0.5, 1.0])
        plt.legend()

        plt.savefig(filepath, bbox_inches="tight", dpi=300)
        plt.close()
