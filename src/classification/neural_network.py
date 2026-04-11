import torch
import torch.nn as nn
from torchview import draw_graph
import numpy as np
import matplotlib.pyplot as plt
from configs.config import PLOTS_PATH


class ClassificationNeuralNet(nn.Module):
    def __init__(self, input_size=12, num_classes=1):

        super().__init__()

        self.hidden1a = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU()
        )

        self.hidden1b = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU()
        )

        self.hidden2a = nn.Sequential(
            nn.Linear(32, 8),
            nn.ReLU()
        )

        self.hidden2b = nn.Sequential(
            nn.Linear(32, 8),
            nn.ReLU()
        )

        self.output = nn.Linear(16, num_classes)

    def forward(self, x):
        h1a = self.hidden1a(x)
        h1b = self.hidden1b(x)
        concat1 = torch.concat([h1a, h1b], dim=1)
        h2a = self.hidden2a(concat1)
        h2b = self.hidden2b(concat1)
        concat2 = torch.concat([h2a, h2b], dim=1)
        out = self.output(concat2)

        return out

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

    # ****FUNCTION TO PLOT NEURAL NET ARCHITECTURE****
    def plot_neural_net(self, model):
        filepath = PLOTS_PATH / 'classifications/neural_net/'

        model_graph = draw_graph(model, input_size=(1, 12), device="cpu")
        model_graph.visual_graph.render(directory=filepath,
                                        filename="class_neural_net", format='png', cleanup=True)
