"""
GraphGenerator and GraphDiscriminator - the "novel technique" component of Pillar 2.

**Why a graph model at all, restated concretely:** take a 4-hop mule layering chain. Every
individual transaction in it — account X pays account Y some amount — is, on its own, a
completely unremarkable transfer. There is no row-level feature that flags it. What *is*
anomalous is the shape: X only ever appears as a sender, Y appears as both receiver and sender
within minutes, and the chain terminates at a brand-new "sink" account. A tabular model that
scores rows independently structurally cannot see that shape. A graph model, whose job is
explicitly "reason about a node using its neighbors," can.

**Generator design** — rather than having the generator hallucinate a fully-formed graph in one
shot from an MLP (which is what a naive from-scratch attempt does, and it shows: the resulting
edges/features don't cohere with each other), it does two things in sequence:
1. An MLP maps (noise, ring-type condition) to an initial guess at node features, a node
   existence mask, and a soft adjacency (edge probabilities/weights).
2. **One round of the same hand-rolled graph convolution the discriminator uses** (see
   `gnn_layers.py`) refines the node features using that just-generated adjacency - i.e. the
   generator "looks at" the graph it just proposed and adjusts node features to be consistent
   with it (a mule 3 hops in should look different from the initial source, and this refinement
   step is what actually produces that, rather than every node being an independent MLP output).

**Discriminator design** — a small stack of `GraphConvLayer`s builds node embeddings, a masked
mean-pool collapses them into one graph-level embedding, and a linear head scores it. This is a
real, if compact, GNN classifier — not a tabular model fed flattened adjacency values (which
would throw away the topology-invariance a GNN is supposed to provide: two isomorphic graphs
with relabeled nodes should get the same score, which only holds if the model treats node order
as arbitrary, exactly what message-passing + pooling guarantees and a flattened-adjacency MLP
does not).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .gnn_layers import GraphConvLayer, MaskedGraphPool, normalize_adjacency


class GraphGenerator(nn.Module):
    def __init__(self, noise_dim: int, cond_dim: int, max_nodes: int, node_feat_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.max_nodes = max_nodes
        self.node_feat_dim = node_feat_dim
        self.trunk = nn.Sequential(
            nn.Linear(noise_dim + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.node_head = nn.Linear(hidden_dim, max_nodes * node_feat_dim)
        self.mask_head = nn.Linear(hidden_dim, max_nodes)
        self.edge_head = nn.Linear(hidden_dim, max_nodes * max_nodes)
        self.refine = GraphConvLayer(node_feat_dim, node_feat_dim)
        self.register_buffer("no_self_loop", 1 - torch.eye(max_nodes))

    def forward(self, noise: torch.Tensor, cond: torch.Tensor):
        h = self.trunk(torch.cat([noise, cond], dim=1))
        batch = h.size(0)

        node_feats = torch.tanh(self.node_head(h)).view(batch, self.max_nodes, self.node_feat_dim)
        mask = torch.sigmoid(self.mask_head(h))  # soft existence prob; thresholded at sample time
        edge_raw = self.edge_head(h).view(batch, self.max_nodes, self.max_nodes)
        adj = torch.sigmoid(edge_raw) * self.no_self_loop

        adj_norm = normalize_adjacency(adj, mask)
        refined_feats = self.refine(node_feats, adj_norm, mask)
        return refined_feats, adj, mask


class GraphDiscriminator(nn.Module):
    def __init__(self, node_feat_dim: int, cond_dim: int, hidden_dim: int = 128, n_layers: int = 2):
        super().__init__()
        dims = [node_feat_dim] + [hidden_dim] * n_layers
        self.convs = nn.ModuleList([GraphConvLayer(dims[i], dims[i + 1]) for i in range(n_layers)])
        self.pool = MaskedGraphPool()
        self.out = nn.Sequential(nn.Linear(hidden_dim + cond_dim, hidden_dim), nn.LeakyReLU(0.2), nn.Linear(hidden_dim, 1))

    def forward(self, node_feats: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        adj_norm = normalize_adjacency(adj, mask)
        h = node_feats
        for conv in self.convs:
            h = conv(h, adj_norm, mask)
        graph_embedding = self.pool(h, mask)
        return self.out(torch.cat([graph_embedding, cond], dim=1))
