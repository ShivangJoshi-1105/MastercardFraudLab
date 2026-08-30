"""
Hand-rolled graph convolution, written in plain PyTorch over dense adjacency matrices rather than
importing PyTorch Geometric (see the plan's rationale: Python 3.13 is too new for PyG's prebuilt
wheels to install reliably, and our mule/ring/fan-in subgraphs are tiny - at most a dozen nodes -
so a dense N x N adjacency is not a scalability problem here the way it would be on a million-node
graph).

**The core idea, in one line:** a node's next-layer representation should depend on its own
current features *and* on what its neighbors currently look like. That's message passing.
Concretely, for adjacency `A` (N x N, `A[i, j]` = edge weight/strength from node i to node j) and
node features `X` (N x F):

    H = activation( A_norm @ X @ W_neighbor + X @ W_self )

`A_norm` is degree-normalized so a node with 10 neighbors doesn't get a proportionally huge
"neighbor message" compared to a node with 1 - without this, high-degree nodes (like a mule
chain's collector account) would dominate purely by virtue of connectivity, not by anything
meaningful. `W_self` (self-loop weight) is what lets a node retain its own identity instead of
being smoothed away into its neighborhood average as layers stack, which is a well-known failure
mode of pure neighbor-averaging GCNs on shallow, small graphs like ours.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def normalize_adjacency(adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Row-normalize adjacency by (masked) out-degree, so each node's incoming "message" is an
    average, not a sum, of its neighbors. `mask` (batch, N) marks which node slots are real vs.
    padding, since our graphs have variable size but are stored in fixed-size (max_nodes) tensors."""
    mask2d = mask.unsqueeze(1) * mask.unsqueeze(2)  # (batch, N, N): both endpoints must be real
    adj = adj * mask2d
    degree = adj.sum(dim=2, keepdim=True).clamp(min=1e-6)
    return adj / degree


class GraphConvLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.w_neighbor = nn.Linear(in_dim, out_dim, bias=False)
        self.w_self = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (batch, N, in_dim), adj_norm: (batch, N, N), mask: (batch, N)
        neighbor_msg = torch.bmm(adj_norm, self.w_neighbor(x))
        out = torch.relu(neighbor_msg + self.w_self(x))
        return out * mask.unsqueeze(-1)  # zero out padding slots so they can't leak signal


class MaskedGraphPool(nn.Module):
    """Graph-level embedding = mean of real (non-padding) node embeddings. Mean rather than sum
    so the discriminator's score doesn't just track "how many nodes does this graph have"."""

    def forward(self, node_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        summed = (node_embeddings * mask.unsqueeze(-1)).sum(dim=1)
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return summed / counts
