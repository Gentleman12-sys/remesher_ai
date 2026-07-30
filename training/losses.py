"""
Combined training loss for MeshGNN:

    Total = 0.3 * BCE
          + 0.3 * Chamfer
          + 0.2 * Normal
          + 0.1 * Curvature
          + 0.1 * Edge Regularization

BCE is the primary supervised signal: predicted edge importance vs. the
QEM-derived ground-truth label (preprocessing/generate_pairs.py).

The other four terms are geometry-aware regularizers computed per mesh in
the batch (a PyG Batch is split back into its constituent graphs with
`to_data_list()`, which is cheap at batch_size=8):

  * Chamfer / Normal - a per-vertex "importance" weight is obtained by
    averaging incident edge scores, normalized to sum to 1 (softmax-free,
    just importance / importance.sum()). Minimizing the resulting *weighted*
    Chamfer / normal-mismatch loss can only be achieved by moving weight
    mass toward vertices that stay close (in position and normal) to the
    independently-generated target low-poly mesh -- i.e. vertices that are
    geometrically "safe" to treat as important survivors of simplification.
    Because the weights are constrained to sum to 1, the trivial "set every
    weight to zero" solution is not available to the optimizer.

  * Curvature - direct regression: the aggregated per-vertex importance
    should track local surface curvature magnitude (sharp features matter).

  * Edge Regularization - a graph-Laplacian-style smoothness penalty: an
    edge's importance should be close to the average importance of its two
    endpoints, discouraging noisy, spatially incoherent predictions.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from models.mesh_gnn import MeshGNN


def _vertex_importance(unique_scores: torch.Tensor, unique_edges: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Average incident-edge importance per vertex, shape (num_nodes,)."""
    device = unique_scores.device
    importance = torch.zeros(num_nodes, device=device, dtype=unique_scores.dtype)
    counts = torch.zeros(num_nodes, device=device, dtype=unique_scores.dtype)

    src, dst = unique_edges[:, 0], unique_edges[:, 1]
    importance.index_add_(0, src, unique_scores)
    importance.index_add_(0, dst, unique_scores)
    counts.index_add_(0, src, torch.ones_like(unique_scores))
    counts.index_add_(0, dst, torch.ones_like(unique_scores))

    return importance / counts.clamp_min(1.0)


def _weighted_chamfer_and_normal(
    pred_points: torch.Tensor,
    pred_normals: torch.Tensor,
    weights: torch.Tensor,
    target_points: torch.Tensor,
    target_normals: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if target_points.numel() == 0 or pred_points.numel() == 0:
        zero = pred_points.sum() * 0.0
        return zero, zero

    dists = torch.cdist(pred_points, target_points)  # (N, M)
    nn_dist, nn_idx = dists.min(dim=1)  # (N,)

    w = weights / weights.sum().clamp_min(1e-8)

    chamfer = torch.sum(w * nn_dist)

    nearest_target_normals = target_normals[nn_idx]
    cos_sim = F.cosine_similarity(pred_normals, nearest_target_normals, dim=1, eps=1e-6)
    normal_err = 1.0 - cos_sim
    normal_loss = torch.sum(w * normal_err)

    return chamfer, normal_loss


class CombinedLoss(nn.Module):
    def __init__(
        self,
        w_bce: float = 0.3,
        w_chamfer: float = 0.3,
        w_normal: float = 0.2,
        w_curvature: float = 0.1,
        w_edge_reg: float = 0.1,
    ):
        super().__init__()
        self.w_bce = w_bce
        self.w_chamfer = w_chamfer
        self.w_normal = w_normal
        self.w_curvature = w_curvature
        self.w_edge_reg = w_edge_reg

    def forward(self, scores_directed: torch.Tensor, batch: Batch) -> Tuple[torch.Tensor, Dict[str, float]]:
        bce = F.binary_cross_entropy(scores_directed, batch.y)

        chamfer_total = scores_directed.sum() * 0.0
        normal_total = scores_directed.sum() * 0.0
        curvature_total = scores_directed.sum() * 0.0
        edge_reg_total = scores_directed.sum() * 0.0

        graphs = batch.to_data_list()
        edge_offset = 0
        for graph in graphs:
            num_directed = graph.edge_index.shape[1]
            g_scores = scores_directed[edge_offset : edge_offset + num_directed]
            edge_offset += num_directed

            unique_scores = MeshGNN.to_undirected_scores(g_scores)  # (E,)
            unique_edges = graph.edge_index[:, : num_directed // 2].t()  # (E, 2)

            vertex_importance = _vertex_importance(unique_scores, unique_edges, graph.x.shape[0])

            pred_points = graph.x[:, :3]
            pred_normals = graph.x[:, 3:6]
            curvature = graph.x[:, 6]

            chamfer, normal_loss = _weighted_chamfer_and_normal(
                pred_points, pred_normals, vertex_importance, graph.target_vertices, graph.target_normals
            )
            chamfer_total = chamfer_total + chamfer
            normal_total = normal_total + normal_loss

            curvature_abs = curvature.abs()
            c_min, c_max = curvature_abs.min(), curvature_abs.max()
            curvature_target = (curvature_abs - c_min) / (c_max - c_min).clamp_min(1e-6)
            curvature_total = curvature_total + F.mse_loss(vertex_importance, curvature_target)

            src, dst = unique_edges[:, 0], unique_edges[:, 1]
            local_mean = 0.5 * (vertex_importance[src] + vertex_importance[dst])
            edge_reg_total = edge_reg_total + F.mse_loss(unique_scores, local_mean)

        n_graphs = max(1, len(graphs))
        chamfer_loss = chamfer_total / n_graphs
        normal_loss = normal_total / n_graphs
        curvature_loss = curvature_total / n_graphs
        edge_reg_loss = edge_reg_total / n_graphs

        total = (
            self.w_bce * bce
            + self.w_chamfer * chamfer_loss
            + self.w_normal * normal_loss
            + self.w_curvature * curvature_loss
            + self.w_edge_reg * edge_reg_loss
        )

        components = {
            "total": float(total.detach().cpu()),
            "bce": float(bce.detach().cpu()),
            "chamfer": float(chamfer_loss.detach().cpu()),
            "normal": float(normal_loss.detach().cpu()),
            "curvature": float(curvature_loss.detach().cpu()),
            "edge_reg": float(edge_reg_loss.detach().cpu()),
        }
        return total, components
