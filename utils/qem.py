"""
Feature-Weighted Quadric Error Metrics (QEM) mesh simplification.

Implements the classic Garland-Heckbert edge-collapse algorithm from scratch
(pure NumPy + heapq), extended so that every candidate edge can carry an
external "importance" weight in [0, 1] supplied by the GNN
(models/mesh_gnn.py). Edges the network marks as important are made
artificially expensive to collapse, so they survive simplification even if
their raw geometric QEM cost is low (e.g. a sharp but short edge).

This module has two public entry points:

    compute_vertex_quadrics(vertices, faces)
        Per-vertex 4x4 error quadrics, used both by the simplifier below and
        by preprocessing/generate_pairs.py to derive ground-truth edge
        importance labels for training.

    compute_edge_qem_scores(vertices, faces)
        Raw, percentile-normalized [0, 1] QEM importance per undirected edge.
        Used as the supervised training target for edge importance.

    WeightedQEMSimplifier
        Runs the actual edge-collapse decimation to a target face count,
        optionally biased by a per-edge importance dict.

No third-party decimation library is required: pyfqmr (used in
preprocessing/generate_pairs.py) does not support externally supplied
per-edge weights, so this hand-written simplifier is what actually powers
"Feature Weighted QEM Simplification" at inference time.
"""

from __future__ import annotations

import heapq
from typing import Dict, Optional, Tuple

import numpy as np


def _face_planes(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return one homogeneous plane [a, b, c, d] per face (unit normal, ax+by+cz+d=0)."""
    p0 = vertices[faces[:, 0]]
    p1 = vertices[faces[:, 1]]
    p2 = vertices[faces[:, 2]]
    normals = np.cross(p1 - p0, p2 - p0)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths < 1e-12] = 1.0
    normals = normals / lengths
    d = -np.einsum("ij,ij->i", normals, p0)
    return np.concatenate([normals, d[:, None]], axis=1)  # (F, 4)


def compute_vertex_quadrics(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Sum-of-plane-quadrics per vertex, shape (N, 4, 4)."""
    n = vertices.shape[0]
    quadrics = np.zeros((n, 4, 4), dtype=np.float64)
    planes = _face_planes(vertices, faces)  # (F, 4)
    face_quadrics = np.einsum("fi,fj->fij", planes, planes)  # (F, 4, 4)
    for corner in range(3):
        np.add.at(quadrics, faces[:, corner], face_quadrics)
    return quadrics


def _optimal_contraction(Q: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> Tuple[np.ndarray, float]:
    """Solve for the position minimizing v^T Q v; fall back to the cheapest of {v1, v2, midpoint}."""
    A = Q[:3, :3]
    b = -Q[:3, 3]
    try:
        if abs(np.linalg.det(A)) > 1e-10:
            v_bar = np.linalg.solve(A, b)
            cost = float(_quadric_error(Q, v_bar))
            return v_bar, cost
    except np.linalg.LinAlgError:
        pass

    candidates = [v1, v2, 0.5 * (v1 + v2)]
    costs = [float(_quadric_error(Q, c)) for c in candidates]
    best = int(np.argmin(costs))
    return candidates[best], costs[best]


def _quadric_error(Q: np.ndarray, v: np.ndarray) -> float:
    v_h = np.array([v[0], v[1], v[2], 1.0])
    return float(v_h @ Q @ v_h)


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


def compute_edge_qem_scores(vertices: np.ndarray, faces: np.ndarray) -> Dict[Tuple[int, int], float]:
    """
    Raw QEM-based edge importance in [0, 1], percentile-normalized so it can be
    used as a training target: 1.0 = geometrically critical (high collapse
    cost, should be preserved), 0.0 = safe to collapse.
    """
    quadrics = compute_vertex_quadrics(vertices, faces)
    edges = _unique_edges(faces)

    costs = np.empty(len(edges), dtype=np.float64)
    for idx, (i, j) in enumerate(edges):
        Q = quadrics[i] + quadrics[j]
        _, cost = _optimal_contraction(Q, vertices[i], vertices[j])
        costs[idx] = cost

    costs = np.clip(costs, 0.0, None)
    # Rank-based normalization is robust to the heavy-tailed distribution of
    # raw quadric costs (a handful of edges can dominate by orders of magnitude).
    ranks = np.argsort(np.argsort(costs))
    scores = ranks / max(1, len(costs) - 1)

    return {(int(i), int(j)): float(s) for (i, j), s in zip(edges, scores)}


class WeightedQEMSimplifier:
    """
    Half-edge-free, dictionary-based greedy edge-collapse simplifier.

    Usage:
        simplifier = WeightedQEMSimplifier(vertices, faces)
        new_vertices, new_faces = simplifier.simplify(
            target_faces=20000,
            edge_importance={(i, j): score, ...},  # optional, from the GNN
        )
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        self.vertices = {i: np.array(v, dtype=np.float64) for i, v in enumerate(vertices)}
        self.quadrics = {i: q for i, q in enumerate(compute_vertex_quadrics(vertices, faces))}
        # face id -> tuple of 3 current vertex ids (mutated in place on collapse)
        self.faces: Dict[int, Tuple[int, int, int]] = {
            fid: tuple(int(x) for x in f) for fid, f in enumerate(faces)
        }
        self.vertex_faces: Dict[int, set] = {i: set() for i in self.vertices}
        for fid, f in self.faces.items():
            for v in f:
                self.vertex_faces[v].add(fid)

        self.alive_vertices = set(self.vertices.keys())
        self.alive_faces = set(self.faces.keys())
        self.version = {i: 0 for i in self.vertices}

    def _edge_key(self, a: int, b: int) -> Tuple[int, int]:
        return (a, b) if a < b else (b, a)

    def _neighbors(self, v: int) -> set:
        nbrs = set()
        for fid in self.vertex_faces[v]:
            for u in self.faces[fid]:
                if u != v:
                    nbrs.add(u)
        return nbrs

    def _edge_cost(self, i: int, j: int, edge_importance: Optional[Dict], strength: float):
        Q = self.quadrics[i] + self.quadrics[j]
        v_bar, cost = _optimal_contraction(Q, self.vertices[i], self.vertices[j])
        if edge_importance:
            importance = edge_importance.get(self._edge_key(i, j), 0.0)
            cost = cost * (1.0 + strength * importance)
        return cost, v_bar

    def simplify(
        self,
        target_faces: int,
        edge_importance: Optional[Dict[Tuple[int, int], float]] = None,
        importance_strength: float = 8.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        heap = []
        seen_edges = set()
        for v in self.alive_vertices:
            for u in self._neighbors(v):
                key = self._edge_key(v, u)
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                cost, v_bar = self._edge_cost(key[0], key[1], edge_importance, importance_strength)
                heapq.heappush(
                    heap,
                    (cost, key[0], key[1], self.version[key[0]], self.version[key[1]], v_bar),
                )

        current_faces = len(self.alive_faces)

        while current_faces > target_faces and heap:
            cost, i, j, vi, vj, v_bar = heapq.heappop(heap)

            if i not in self.alive_vertices or j not in self.alive_vertices:
                continue
            if self.version[i] != vi or self.version[j] != vj:
                continue  # stale entry, one of the endpoints changed since this was pushed

            current_faces -= self._collapse_edge(i, j, v_bar)

            # push refreshed costs for edges now touching the merged vertex i
            for u in self._neighbors(i):
                key = self._edge_key(i, u)
                cost, v_bar_new = self._edge_cost(key[0], key[1], edge_importance, importance_strength)
                heapq.heappush(
                    heap,
                    (cost, key[0], key[1], self.version[key[0]], self.version[key[1]], v_bar_new),
                )

        return self._export()

    def _collapse_edge(self, i: int, j: int, v_bar: np.ndarray) -> int:
        """Merge j into i at position v_bar. Returns the number of faces removed."""
        self.vertices[i] = v_bar
        self.quadrics[i] = self.quadrics[i] + self.quadrics[j]

        removed = 0
        affected_faces = list(self.vertex_faces[j])
        for fid in affected_faces:
            face = self.faces[fid]
            new_face = tuple(i if v == j else v for v in face)
            if len(set(new_face)) < 3:
                # face degenerates (both i and j were already in it) -> drop it
                self.alive_faces.discard(fid)
                for v in face:
                    self.vertex_faces[v].discard(fid)
                removed += 1
            else:
                self.faces[fid] = new_face
                self.vertex_faces[i].add(fid)
                self.vertex_faces[j].discard(fid)

        self.alive_vertices.discard(j)
        self.version[i] += 1
        self.version[j] += 1
        del self.vertex_faces[j]
        return removed

    def _export(self) -> Tuple[np.ndarray, np.ndarray]:
        old_to_new = {}
        new_vertices = []
        for old_id in sorted(self.alive_vertices):
            old_to_new[old_id] = len(new_vertices)
            new_vertices.append(self.vertices[old_id])

        new_faces = []
        for fid in self.alive_faces:
            face = self.faces[fid]
            if len(set(face)) == 3:
                new_faces.append([old_to_new[v] for v in face])

        return np.array(new_vertices, dtype=np.float64), np.array(new_faces, dtype=np.int64)
