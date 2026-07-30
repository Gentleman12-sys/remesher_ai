"""
Geometric feature extraction for the mesh graph fed to the GNN
(models/mesh_gnn.py).

Per-vertex (node) features, 8-dim:
    [x, y, z, nx, ny, nz, curvature, valence]
    - x,y,z         : position, centered and scaled to a unit bounding sphere
    - nx,ny,nz      : vertex normal
    - curvature     : discrete Gaussian curvature (angle-defect), tanh-normalized to [-1, 1]
    - valence       : vertex degree / 6 (6 = expected valence on a regular
                      triangulation), clipped to [0, 3] -- 1.0 means "typical"

Per-edge features, 8-dim, ALL normalized to comparable [0, ~1]-ish scales so
they're consistent across meshes of very different density/category (a raw,
un-normalized dihedral angle or face area varies by an order of magnitude
between e.g. a coarse box and a dense knot mesh purely from face count, not
geometry -- that heterogeneity actively hurts training across categories):
    [length, dihedral_angle, curvature, valence, normal_difference,
     boundary, uv_seam, face_area]
    - length            : edge length / mesh scale (same normalized space as node xyz)
    - dihedral_angle    : angle between the two adjacent face normals, / pi -> [0, 1]
    - curvature         : mean vertex curvature of the two endpoints
    - valence           : mean vertex valence (see above) of the two endpoints
    - normal_difference : (1 - cos(dihedral_angle)) / 2 -> [0, 1]
    - boundary          : 1.0 if the edge belongs to a single face
    - uv_seam           : 1.0 if the edge lies on a UV island boundary
    - face_area         : mean adjacent face area / this mesh's mean face area,
                          clipped to [0, 5] -- 1.0 means "typical" for this mesh
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import trimesh
from scipy.spatial import cKDTree

NODE_FEATURE_NAMES = ["x", "y", "z", "nx", "ny", "nz", "curvature", "valence"]
EDGE_FEATURE_NAMES = [
    "length",
    "dihedral_angle",
    "curvature",
    "valence",
    "normal_difference",
    "boundary",
    "uv_seam",
    "face_area",
]

NODE_FEATURE_DIM = len(NODE_FEATURE_NAMES)
EDGE_FEATURE_DIM = len(EDGE_FEATURE_NAMES)


def _vertex_curvature(mesh: trimesh.Trimesh) -> np.ndarray:
    """Discrete Gaussian curvature via angle defect (2*pi - sum of incident angles)."""
    defects = trimesh.curvature.vertex_defects(mesh)
    # Normalize with tanh so a handful of very sharp vertices don't dominate the scale.
    return np.tanh(defects / np.pi).astype(np.float32)


_REGULAR_VALENCE = 6.0  # expected vertex valence on a regular triangulated 2-manifold


def _vertex_valence(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Valence normalized against the REGULAR triangulation valence (6), not the
    mesh's own max: dividing by a per-mesh max collapses almost every vertex
    to ~1.0 (most vertices sit near 6, only rare poles reach the max), which
    destroys the signal. Dividing by a fixed reference keeps 1.0 meaning
    "typical interior vertex" and lets poles/boundaries show up as real
    deviations, comparable across meshes of any density.
    """
    n = len(mesh.vertices)
    valence = np.bincount(mesh.edges_unique.ravel(), minlength=n).astype(np.float32)
    return np.clip(valence / _REGULAR_VALENCE, 0.0, 3.0)


def _uv_seam_edges(mesh: trimesh.Trimesh, boundary_mask: np.ndarray, unique_edges: np.ndarray) -> np.ndarray:
    """
    Detect edges that are boundary-in-the-graph-sense but are actually interior
    UV-island cuts: the edge has only one incident face, yet both endpoints
    have a near-duplicate vertex elsewhere in the mesh at (almost) the same
    3D position (the duplicate created by an OBJ UV/normal seam).
    """
    seam = np.zeros(len(unique_edges), dtype=np.float32)
    uv = getattr(mesh.visual, "uv", None)
    if uv is None or not boundary_mask.any():
        return seam

    tree = cKDTree(mesh.vertices)
    tol = max(1e-8, 1e-5 * mesh.scale)

    for idx in np.nonzero(boundary_mask)[0]:
        i, j = unique_edges[idx]
        dup_i = [k for k in tree.query_ball_point(mesh.vertices[i], tol) if k != i]
        dup_j = [k for k in tree.query_ball_point(mesh.vertices[j], tol) if k != j]
        if dup_i and dup_j:
            seam[idx] = 1.0
    return seam


def mesh_normalization_stats(mesh: trimesh.Trimesh) -> Tuple[np.ndarray, float]:
    """Center (mean position) and scale (max distance from center) used to fit the mesh to a unit sphere."""
    center = mesh.vertices.mean(axis=0)
    scale = max(1e-8, float(np.linalg.norm(mesh.vertices - center, axis=1).max()))
    return center, scale


def extract_node_features(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return (N, NODE_FEATURE_DIM) float32 array."""
    center, scale = mesh_normalization_stats(mesh)
    positions = ((mesh.vertices - center) / scale).astype(np.float32)

    normals = mesh.vertex_normals.astype(np.float32)
    curvature = _vertex_curvature(mesh)
    valence = _vertex_valence(mesh)

    return np.concatenate(
        [positions, normals, curvature[:, None], valence[:, None]], axis=1
    ).astype(np.float32)


def extract_edge_features(mesh: trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        edge_index_directed : (2, 2E) int64, both directions, for GNN message passing
        edge_attr_directed  : (2E, EDGE_FEATURE_DIM) float32, aligned with edge_index_directed
        unique_edges        : (E, 2) int64, canonical (i < j) undirected edges
        edge_attr_unique    : (E, EDGE_FEATURE_DIM) float32, aligned with unique_edges
    """
    unique_edges = mesh.edges_unique.astype(np.int64)  # (E, 2), already i < j... trimesh guarantees sorted
    n_edges = len(unique_edges)

    _, scale = mesh_normalization_stats(mesh)
    lengths = (mesh.edges_unique_length / scale).astype(np.float32)

    curvature = _vertex_curvature(mesh)
    valence = _vertex_valence(mesh)
    edge_curvature = curvature[unique_edges].mean(axis=1)
    edge_valence = valence[unique_edges].mean(axis=1)

    # face_adjacency gives, per INTERNAL manifold edge, the pair of faces sharing it.
    # face_adjacency_edges tells us which (unsorted) vertex pair that corresponds to.
    dihedral = np.zeros(n_edges, dtype=np.float32)
    normal_diff = np.zeros(n_edges, dtype=np.float32)
    face_area = np.zeros(n_edges, dtype=np.float32)
    boundary = np.ones(n_edges, dtype=np.float32)

    edge_lookup = {(int(a), int(b)): idx for idx, (a, b) in enumerate(unique_edges)}

    adj_edges = np.sort(mesh.face_adjacency_edges, axis=1)
    adj_angles = mesh.face_adjacency_angles
    adj_faces = mesh.face_adjacency
    face_areas = mesh.area_faces

    for k in range(len(adj_edges)):
        key = (int(adj_edges[k, 0]), int(adj_edges[k, 1]))
        idx = edge_lookup.get(key)
        if idx is None:
            continue
        boundary[idx] = 0.0
        dihedral[idx] = adj_angles[k]
        normal_diff[idx] = 1.0 - np.cos(adj_angles[k])
        f1, f2 = adj_faces[k]
        face_area[idx] = 0.5 * (face_areas[f1] + face_areas[f2])

    # boundary edges: still assign the area of their single incident face
    if boundary.any():
        face_of_edge = {}
        for fidx, face in enumerate(mesh.faces):
            for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                key = (int(a), int(b)) if a < b else (int(b), int(a))
                face_of_edge.setdefault(key, []).append(fidx)
        for idx in np.nonzero(boundary)[0]:
            key = (int(unique_edges[idx, 0]), int(unique_edges[idx, 1]))
            faces_here = face_of_edge.get(key, [])
            if faces_here:
                face_area[idx] = float(np.mean(face_areas[faces_here]))

    uv_seam = _uv_seam_edges(mesh, boundary.astype(bool), unique_edges)

    # Normalize scale-heterogeneous features so they're comparable across
    # meshes of very different density/category, not just within one mesh:
    #   - dihedral_angle lives in [0, pi] radians -> rescale to [0, 1]
    #   - normal_difference lives in [0, 2] (1 - cos) -> rescale to [0, 1]
    #   - face_area has no natural bound and varies ~10x in mean scale between
    #     a coarse box and a dense knot mesh purely from face count, not
    #     geometry -> express it relative to this mesh's own mean face area
    #     (1.0 = "typical" face for this mesh), clipped against slivers/outliers
    dihedral_norm = dihedral / np.pi
    normal_diff_norm = normal_diff / 2.0
    mean_face_area = max(1e-12, float(face_areas.mean()))
    face_area_norm = np.clip(face_area / mean_face_area, 0.0, 5.0)

    edge_attr_unique = np.stack(
        [lengths, dihedral_norm, edge_curvature, edge_valence, normal_diff_norm, boundary, uv_seam, face_area_norm],
        axis=1,
    ).astype(np.float32)

    edge_index_directed = np.concatenate(
        [unique_edges.T, unique_edges[:, ::-1].T], axis=1
    ).astype(np.int64)
    edge_attr_directed = np.concatenate([edge_attr_unique, edge_attr_unique], axis=0)

    return edge_index_directed, edge_attr_directed, unique_edges, edge_attr_unique
