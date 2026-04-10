import torch
import torch.nn as nn


class WideAndDeepNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 50),
            nn.ReLU(),
            nn.Linear(50, 40),
            nn.ReLU(),
            nn.Linear(40, 30),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(30+input_size, 1)

    def forward(self, X):

        deep_output = self.model(X)
        wide_and_deep = torch.concat([X, deep_output], dim=1)

        return self.output_layer(wide_and_deep)

    # ADD PLOTTING FUNCTIONS
