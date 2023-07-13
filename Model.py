import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, input_features, number_of_nodes, output_features):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_features, out_features=number_of_nodes),
            nn.ReLU(),
            nn.Linear(in_features=number_of_nodes, out_features=output_features),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor):
        return self.sequential(x)
