"""
Quality metrics for comparing an original (high-poly) mesh against a
simplified (low-poly) mesh: Hausdorff distance, Chamfer distance, normal
deviation, triangle quality and face reduction percentage.

Used by inference/remesh.py (--compare flag) and training/train.py
(validation-time sanity checks).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def _sample_points(mesh: trimesh.Trimesh, n_points: int, seed: int = 0) -> np.ndarray:
    n_points = min(n_points, max(1, len(mesh.faces) * 10))
    points, _ = trimesh.sample.sample_surface(mesh, n_points, seed=seed)
    return points


def hausdorff_distance(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh, n_points: int = 20000) -> float:
    """Symmetric (two-sided) Hausdorff distance between the surfaces, approximated by dense sampling."""
    pa = _sample_points(mesh_a, n_points, seed=0)
    pb = _sample_points(mesh_b, n_points, seed=1)

    tree_b = cKDTree(pb)
    d_ab = tree_b.query(pa)[0].max()

    tree_a = cKDTree(pa)
    d_ba = tree_a.query(pb)[0].max()

    return float(max(d_ab, d_ba))


def chamfer_distance(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh, n_points: int = 20000) -> float:
    """Symmetric mean-squared nearest-neighbor distance between sampled point clouds."""
    pa = _sample_points(mesh_a, n_points, seed=0)
    pb = _sample_points(mesh_b, n_points, seed=1)

    tree_b = cKDTree(pb)
    d_ab = tree_b.query(pa)[0]

    tree_a = cKDTree(pa)
    d_ba = tree_a.query(pb)[0]

    return float(np.mean(d_ab**2) + np.mean(d_ba**2))


def normal_deviation(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh, n_points: int = 10000) -> float:
    """Mean angular deviation (radians) between surface normals at nearest-neighbor sample points."""
    pa, fid_a = trimesh.sample.sample_surface(mesh_a, min(n_points, len(mesh_a.faces) * 10), seed=0)
    pb, fid_b = trimesh.sample.sample_surface(mesh_b, min(n_points, len(mesh_b.faces) * 10), seed=1)

    na = mesh_a.face_normals[fid_a]
    nb = mesh_b.face_normals[fid_b]

    tree_b = cKDTree(pb)
    _, nn_idx = tree_b.query(pa)

    cos_sim = np.clip(np.einsum("ij,ij->i", na, nb[nn_idx]), -1.0, 1.0)
    angles = np.arccos(cos_sim)
    return float(np.mean(angles))


def triangle_quality(mesh: trimesh.Trimesh) -> Dict[str, float]:
    """
    Mean/min normalized triangle quality (1.0 = equilateral, 0.0 = degenerate),
    using Q = 4*sqrt(3)*Area / (l0^2 + l1^2 + l2^2).
    """
    tri = mesh.vertices[mesh.faces]
    edges = np.stack(
        [
            np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
            np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
            np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1),
        ],
        axis=1,
    )
    sq_sum = np.sum(edges**2, axis=1)
    sq_sum[sq_sum < 1e-12] = 1e-12
    quality = (4.0 * np.sqrt(3.0) * mesh.area_faces) / sq_sum
    quality = np.clip(quality, 0.0, 1.0)
    return {
        "mean_triangle_quality": float(np.mean(quality)),
        "min_triangle_quality": float(np.min(quality)),
    }


def face_reduction_percentage(original: trimesh.Trimesh, simplified: trimesh.Trimesh) -> float:
    return float(100.0 * (1.0 - len(simplified.faces) / max(1, len(original.faces))))


def quality_report(original: trimesh.Trimesh, simplified: trimesh.Trimesh) -> Dict[str, float]:
    """Full comparison report used by inference/remesh.py --compare."""
    tq = triangle_quality(simplified)
    return {
        "original_faces": len(original.faces),
        "simplified_faces": len(simplified.faces),
        "face_reduction_percent": face_reduction_percentage(original, simplified),
        "hausdorff_distance": hausdorff_distance(original, simplified),
        "chamfer_distance": chamfer_distance(original, simplified),
        "mean_normal_deviation_deg": float(np.degrees(normal_deviation(original, simplified))),
        **tq,
    }
