import torch.nn as nn
import torch.nn.functional as F


class NormalizeLayer(nn.Module):
    def __init__(self, dim=-1, p=2):
        """
        Initialize the NormalizeLayer.

        This is a wrapper around `torch.nn.functional.normalize` that enables using it
        as an nn.Module.

        Args:
            dim (int): The dimension along which to normalize. Default is -1 (last dimension).
            p (float): The norm degree. Default is 2 (L2 normalization).
        """
        super(NormalizeLayer, self).__init__()
        self.dim = dim
        self.p = p

    def forward(self, x):
        return F.normalize(x, dim=self.dim, p=self.p)
