"""
MeshGNN: predicts, for every mesh edge, an "importance" score in [0, 1]
(0 = safe to collapse, 1 = must be preserved) from local geometric features.

Architecture (as specified):

    GraphConv -> GraphConv -> Graph Attention Layer -> MLP

The two GraphConv layers propagate coarse structural context across the
mesh graph. The GATConv layer (attention, edge-feature aware) lets each
vertex weigh its neighbors by how geometrically relevant they are before
the final per-edge MLP head combines both endpoint embeddings with the
raw edge features to produce the importance score.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GraphConv

from preprocessing.feature_extractor import EDGE_FEATURE_DIM, NODE_FEATURE_DIM

__all__ = ["MeshGNN", "NODE_FEATURE_DIM", "EDGE_FEATURE_DIM"]


class MeshGNN(nn.Module):
    def __init__(
        self,
        node_dim: int = NODE_FEATURE_DIM,
        edge_dim: int = EDGE_FEATURE_DIM,
        hidden_dim: int = 64,
        gat_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert hidden_dim % gat_heads == 0, "hidden_dim must be divisible by gat_heads"

        self.dropout = dropout

        self.conv1 = GraphConv(node_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)

        self.gat = GATConv(
            hidden_dim,
            hidden_dim // gat_heads,
            heads=gat_heads,
            edge_dim=edge_dim,
            dropout=dropout,
            add_self_loops=False,
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode_nodes(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = F.relu(self.conv2(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = F.elu(self.gat(h, edge_index, edge_attr))
        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        x          : (N, node_dim)
        edge_index : (2, E_directed) -- both directions of every undirected edge
        edge_attr  : (E_directed, edge_dim)

        Returns edge_importance : (E_directed,) in [0, 1], aligned with edge_index.
        """
        h = self.encode_nodes(x, edge_index, edge_attr)

        src, dst = edge_index
        edge_input = torch.cat([h[src], h[dst], edge_attr], dim=1)
        logits = self.edge_mlp(edge_input).squeeze(-1)
        return torch.sigmoid(logits)

    @staticmethod
    def to_undirected_scores(scores: torch.Tensor) -> torch.Tensor:
        """
        edge_index_directed is always built (see preprocessing/feature_extractor.py)
        as concat([unique_edges, reversed(unique_edges)]), so the first half and
        second half of `scores` correspond 1:1 to the same undirected edges.
        Average them to get one importance value per undirected edge.
        """
        assert scores.shape[0] % 2 == 0, "expected an even number of directed edges"
        half = scores.shape[0] // 2
        return 0.5 * (scores[:half] + scores[half:])
