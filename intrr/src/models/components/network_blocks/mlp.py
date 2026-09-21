from typing import List, Optional

import torch
from torch import nn


class MLP(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim_list: Optional[List[int]] = None,
        activation: nn.Module = nn.ReLU,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        """Initialize the MLP.

        Args:
            input_dim (int): The dimensionality of the input tensor.
            output_dim (int): The dimensionality of the output tensor.
            hidden_dim_list Optional(List[int]): A list of the dimensions of each hidden
                layer output. The number of layers in the MLP is the length of this list
                plus one.
            activation (nn.Module): The activation function to use between layers.
            bias (bool): Whether to include bias terms in the linear layers.
            dropout (float): The dropout rate to apply after each layer.
        """
        super().__init__()

        if hidden_dim_list is None:
            hidden_dim_list = []
        hidden_dim_list.append(output_dim)
        layers = [nn.Linear(input_dim, hidden_dim_list[0], bias=bias)]
        for i in range(1, len(hidden_dim_list)):
            layers.append(activation())
            layers.append(
                nn.Linear(hidden_dim_list[i - 1], hidden_dim_list[i], bias=bias)
            )
            layers.append(nn.Dropout(dropout))
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
