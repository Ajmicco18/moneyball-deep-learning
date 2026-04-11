import torch
import torch.nn as nn
from torchview import draw_graph
import numpy as np
import matplotlib.pyplot as plt
from configs.config import PLOTS_PATH


class MultiInputNeuralNet(nn.Module):
    def __init__(self, input_size=7):

        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size-2, 50),
            nn.ReLU(),
            nn.Linear(50, 40),
            nn.ReLU(),
            nn.Linear(40, 30),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(30+7, 1)

    def forward(self, X_wide, X_deep):

        deep_output = self.model(X_deep)
        wide_and_deep = torch.concat([X_wide, deep_output], dim=1)

        return self.output_layer(wide_and_deep)

     # ****FUNCTION TO PLOT AND SAVE TRAINING LOSSES****
    def plot_losses(self, losses):

        filepath = PLOTS_PATH / 'classifications/multi_input_nn/training_loss.png'

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

        filepath = PLOTS_PATH / 'classifications/multi_input_nn/learning_curve.png'

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
        filepath = PLOTS_PATH / 'classifications/multi_input_nn/'

        model_graph = draw_graph(
            model, input_size=[(1, 7), (1, 5)], device="cpu")
        model_graph.visual_graph.render(directory=filepath,
                                        filename="multi_input_neural_net", format='png', cleanup=True)
