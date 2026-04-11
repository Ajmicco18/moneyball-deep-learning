import torch
import torch.nn as nn
from torchview import draw_graph
import numpy as np
import matplotlib.pyplot as plt
from configs.config import PLOTS_PATH


class WideAndDeepNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.hidden1a = nn.Sequential(
            nn.Linear(input_size, 50),
            nn.ReLU(),
            nn.Linear(50, 40),
            nn.ReLU(),
            nn.Linear(40, 30),
            nn.ReLU(),
        )

        self.hidden1b = nn.Sequential(
            nn.Linear(input_size, 50),
            nn.ReLU(),
            nn.Linear(50, 40),
            nn.ReLU(),
            nn.Linear(40, 30),
            nn.ReLU()
        )

        self.hidden2 = nn.Sequential(
            nn.Linear(60, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        self.hidden3a = nn.Sequential(
            nn.Linear(input_size, 50),
            nn.ReLU(),
            nn.Linear(50, 40),
            nn.ReLU(),
            nn.Linear(40, 30),
            nn.ReLU(),
        )

        self.hidden3b = nn.Sequential(
            nn.Linear(input_size, 50),
            nn.ReLU(),
            nn.Linear(50, 40),
            nn.ReLU(),
            nn.Linear(40, 30),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(76, 1)

    def forward(self, X):
        h1a = self.hidden1a(X)
        h1b = self.hidden1b(X)
        concat1 = torch.concat([h1a, h1b], dim=1)
        deep_output = self.hidden2(concat1)
        h3a = self.hidden3a(X)
        h3b = self.hidden3b(X)
        concat2 = torch.concat([h3a, h3b], dim=1)
        wide_and_deep = torch.concat([concat2, deep_output], dim=1)

        return self.output_layer(wide_and_deep)

    # ****FUNCTION TO PLOT AND SAVE TRAINING LOSSES****
    def plot_losses(self, losses):

        filepath = PLOTS_PATH / 'regressions/wide_deep_nn/training_loss.png'

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

        filepath = PLOTS_PATH / 'regressions/wide_deep_nn/learning_curve.png'

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

    # ****FUNCTION TO PLOT NEURAL NET ARCHITECTURE****
    def plot_neural_net(self, model):
        filepath = PLOTS_PATH / 'regressions/wide_deep_nn/'

        model_graph = draw_graph(model, input_size=(1, 14), device="cpu")
        model_graph.visual_graph.render(directory=filepath,
                                        filename="wide_and_deep_nn", format='png', cleanup=True)
