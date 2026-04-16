import torch
import torch.nn as nn
from collections.abc import Iterable
from typing import Union

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim: Union[Iterable[int], int], output_dim):
        super(MLP, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        if isinstance(self.hidden_dim, Iterable):
            self.model = nn.Sequential()
            in_ind = self.input_dim
            for h_dim in self.hidden_dim:
                self.model.append(
                    nn.Linear(in_ind, h_dim, dtype=torch.float64))
                self.model.append(
                    nn.ReLU())
                in_ind = h_dim
            self.model.append(
                nn.Linear(self.hidden_dim[-1], self.output_dim, dtype=torch.float64)
            )
            """
            self.model.append(
                nn.Sigmoid())
            """
        else:
            self.model = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim, dtype=torch.float64),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.output_dim, dtype=torch.float64),
                #nn.Sigmoid()
            )


    def forward(self, x):
        
        return self.model(x)