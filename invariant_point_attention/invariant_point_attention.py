import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange

# helpers

def apply_transform(x, rot, trans):
    return x

def apply_inverse_transform(x, rot, trans):
    return x

# classes

class InvariantPointAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        scalar_key_dim = 16,
        scalar_value_dim = 16,
        point_key_dim = 4,
        point_value_dim = 4,
        eps = 1e-8
    ):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        single_repr,
        pairwise_repr,
        *,
        rotations,
        translations,
        mask = None
    ):
        return single_repr, pairwise_repr
