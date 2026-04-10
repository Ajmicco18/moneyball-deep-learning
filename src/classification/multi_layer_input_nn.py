import torch
import torch.nn as nn


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
