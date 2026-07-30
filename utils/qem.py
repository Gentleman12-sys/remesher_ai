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

    simplify_mesh_by_component(vertices, faces, target_faces, ...)
        Wraps WeightedQEMSimplifier to handle multi-object scenes correctly:
        splits the mesh into connected components first and simplifies each
        one to its OWN proportional target face count, instead of running one
        global collapse queue across the whole scene. Global collapse is
        scale-blind -- absolute quadric-error cost, not relative importance,
        decides what gets collapsed first, so a single global run tends to
        gut large/flat objects while leaving small/detailed ones completely
        untouched (their few edges are never the cheapest in the whole scene)
        and can fuse separate objects together at contact points. Always
        prefer this over calling WeightedQEMSimplifier directly on a mesh
        that might contain more than one disconnected object -- see
        inference/remesh.py, which uses it exclusively.

No third-party decimation library is required: pyfqmr (used in
preprocessing/generate_pairs.py) does not support externally supplied
per-edge weights, so this hand-written simplifier is what actually powers
"Feature Weighted QEM Simplification" at inference time.
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components as _scipy_connected_components


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

    def _link_condition_ok(self, i: int, j: int) -> bool:
        """
        Standard mesh-simplification "link condition" (Dey et al.): collapsing
        edge (i, j) is only topology-safe if the vertices adjacent to BOTH i
        and j are exactly the (at most two) apex vertices of the triangles
        that contain edge (i, j) -- no more, no fewer. If some other vertex k
        is adjacent to both i and j without (i, j, k) being a face, i and j
        are only "close" through some other part of the mesh (e.g. opposite
        sides of a thin neck/handle), and collapsing them would pinch or tear
        the surface -- silently disconnecting it into extra pieces, or fusing
        parts that should stay separate. Skipping such collapses is what
        keeps a thin handle/leg/spout attached instead of getting severed.
        """
        common_neighbors = (self._neighbors(i) & self._neighbors(j)) - {i, j}

        expected_apexes = set()
        for fid in self.vertex_faces[i] & self.vertex_faces[j]:
            face = self.faces[fid]
            if i in face and j in face:
                expected_apexes.update(v for v in face if v != i and v != j)

        return common_neighbors == expected_apexes

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
            if not self._link_condition_ok(i, j):
                continue  # collapsing here would tear/pinch the mesh -- leave it alone

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


def _split_into_components(vertices: np.ndarray, faces: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Split a mesh into separate objects, using FACE adjacency (two faces are
    connected iff they share a full edge, i.e. 2 vertices) -- NOT vertex
    adjacency. A vertex-based split over-merges: two genuinely separate
    objects that merely touch at a single shared vertex (a "bowtie" contact
    point -- common in multi-object scenes/scans, e.g. a fork resting against
    a plate) would be treated as one component, and the QEM collapse queue
    would then happily eat the smaller touching object from inside what it
    thinks is a single blob. Face/edge adjacency matches how a 3D viewer (and
    trimesh.Trimesh.split()) actually defines "separate object".

    Returns a list of (global_vertex_indices, local_faces) pairs: for each
    component, the indices of its vertices into the ORIGINAL `vertices` array,
    and its faces re-indexed to be local to that component (0..len(component)-1).
    """
    n_vertices = len(vertices)
    n_faces = len(faces)
    if n_faces == 0:
        return []

    edges = np.sort(
        np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0), axis=1
    )
    face_ids = np.tile(np.arange(n_faces), 3)

    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges_sorted = edges[order]
    face_ids_sorted = face_ids[order]

    # consecutive rows with an identical edge -> their owning faces share that edge
    shares_edge = np.all(edges_sorted[:-1] == edges_sorted[1:], axis=1)
    face_rows = face_ids_sorted[:-1][shares_edge]
    face_cols = face_ids_sorted[1:][shares_edge]

    if len(face_rows) > 0:
        face_graph = coo_matrix(
            (np.ones(len(face_rows), dtype=np.int8), (face_rows, face_cols)), shape=(n_faces, n_faces)
        )
        _, face_labels = _scipy_connected_components(face_graph, directed=False)
    else:
        face_labels = np.arange(n_faces)  # every face is its own component (no shared edges at all)

    components = []
    for comp_id in np.unique(face_labels):
        comp_faces_global = faces[face_labels == comp_id]
        vertex_indices = np.unique(comp_faces_global)
        local_index = np.full(n_vertices, -1, dtype=np.int64)
        local_index[vertex_indices] = np.arange(len(vertex_indices))
        comp_faces_local = local_index[comp_faces_global]
        components.append((vertex_indices, comp_faces_local))

    return components


def _drop_tiny_fragments(
    vertices: np.ndarray, faces: np.ndarray, min_fragment_faces: int = 4
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Defensive cleanup: the link-condition check in WeightedQEMSimplifier
    prevents almost all topology-tearing collapses, but on messy/near-degenerate
    real-world geometry (thin necks, near-coincident vertices) a handful of
    tiny (often single-face) debris shards can still split off. A stray
    floating triangle reads as visual noise, not detail worth keeping, so drop
    any post-simplification fragment smaller than `min_fragment_faces` --
    unless that would remove everything, in which case keep the largest one.

    Returns (vertices, faces, n_fragments_dropped).
    """
    if len(faces) == 0:
        return vertices, faces, 0

    subcomponents = _split_into_components(vertices, faces)
    if len(subcomponents) <= 1:
        return vertices, faces, 0

    kept = [vertex_indices[local_faces] for vertex_indices, local_faces in subcomponents if len(local_faces) >= min_fragment_faces]
    n_dropped = len(subcomponents) - len(kept)

    if not kept:
        largest = max(subcomponents, key=lambda c: len(c[1]))
        kept = [largest[0][largest[1]]]
        n_dropped = len(subcomponents) - 1

    new_faces = np.concatenate(kept, axis=0)
    used_vertices = np.unique(new_faces)
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used_vertices] = np.arange(len(used_vertices))
    return vertices[used_vertices], remap[new_faces], n_dropped


def simplify_mesh_by_component(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
    edge_importance_edges: Optional[np.ndarray] = None,
    edge_importance_scores: Optional[np.ndarray] = None,
    importance_strength: float = 8.0,
    min_component_faces: int = 50,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Simplify a (possibly multi-object) mesh to `target_faces`, splitting into
    connected components first and giving each one the SAME proportional
    reduction (component_target = component_faces * target_faces / total_faces),
    so a big flat object and a small detailed one lose the same relative
    fraction of detail instead of one dominating a global collapse queue.

    edge_importance_edges/_scores: optional (E, 2) / (E,) arrays of GNN-predicted
    per-edge importance, keyed by GLOBAL vertex indices (as returned by
    inference/remesh.py's predict_edge_importance) -- remapped to each
    component's local indices internally.

    Components smaller than `min_component_faces` are left untouched: below
    that size a component is "a small detail", not "a simplification target",
    and forcing it down to a proportional target would just mangle it.

    Returns (new_vertices, new_faces, stats) where stats reports how many
    components were simplified vs. kept as-is, for transparency.
    """
    components = _split_into_components(vertices, faces)
    if not components:
        return vertices.copy(), faces.copy(), {"components": 0, "simplified": 0, "kept_small": 0}

    total_faces = sum(len(f) for _, f in components)
    global_ratio = target_faces / max(1, total_faces)

    has_importance = edge_importance_edges is not None and len(edge_importance_edges) > 0

    out_vertices, out_faces = [], []
    vertex_offset = 0
    simplified_count, kept_small_count, debris_dropped = 0, 0, 0

    for vertex_indices, comp_faces_local in components:
        comp_vertices = vertices[vertex_indices]
        comp_face_count = len(comp_faces_local)
        comp_target = max(4, round(comp_face_count * global_ratio))

        if comp_face_count < min_component_faces or comp_target >= comp_face_count:
            new_v, new_f = comp_vertices, comp_faces_local
            kept_small_count += 1
        else:
            local_importance = None
            if has_importance:
                vertex_mask = np.zeros(len(vertices), dtype=bool)
                vertex_mask[vertex_indices] = True
                edge_mask = vertex_mask[edge_importance_edges[:, 0]] & vertex_mask[edge_importance_edges[:, 1]]
                if edge_mask.any():
                    global_to_local = np.full(len(vertices), -1, dtype=np.int64)
                    global_to_local[vertex_indices] = np.arange(len(vertex_indices))
                    comp_edges_local = global_to_local[edge_importance_edges[edge_mask]]
                    comp_scores = edge_importance_scores[edge_mask]
                    local_importance = {
                        (int(i), int(j)) if i < j else (int(j), int(i)): float(s)
                        for (i, j), s in zip(comp_edges_local, comp_scores)
                    }

            simplifier = WeightedQEMSimplifier(comp_vertices, comp_faces_local)
            new_v, new_f = simplifier.simplify(
                target_faces=comp_target, edge_importance=local_importance, importance_strength=importance_strength
            )
            new_v, new_f, n_dropped = _drop_tiny_fragments(new_v, new_f)
            debris_dropped += n_dropped
            simplified_count += 1

        out_vertices.append(new_v)
        out_faces.append(new_f + vertex_offset)
        vertex_offset += len(new_v)

    combined_vertices = np.concatenate(out_vertices, axis=0)
    combined_faces = np.concatenate(out_faces, axis=0)
    stats = {
        "components": len(components),
        "simplified": simplified_count,
        "kept_small": kept_small_count,
        "debris_fragments_dropped": debris_dropped,
    }
    return combined_vertices, combined_faces, stats
