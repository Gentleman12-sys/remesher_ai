"""
Generates a small, self-contained, license-clean sample dataset for
AI-MeshOptimizer: two "complex" meshes (many polygons, intricate curvature /
twisting topology) and two "simple" meshes (few polygons, basic shapes), each
paired with an AI-MeshOptimizer-generated optimized ("low poly") counterpart.

All meshes are generated procedurally with trimesh + numpy -- no external
downloads, no licensing questions, fully reproducible. Useful as:
  - a ready-to-open demo of "high poly in / low poly out" (dataset/raw, dataset/pairs)
  - a tiny starter training set (dataset/processed), built with the exact
    same feature extraction + QEM-labeling code as preprocessing/generate_pairs.py

For serious training, replace/augment this with real scans from ABC Dataset /
Objaverse / Thingi10K -- see README.md > Dataset. This script's meshes are
deliberately modest in size (thousands, not millions, of faces) so the whole
thing runs in well under a minute on a laptop CPU.

Usage:
    python dataset/make_sample_dataset.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
from torch_geometric.data import Data

sys.path.append(str(Path(__file__).resolve().parent.parent))

from preprocessing.feature_extractor import extract_edge_features, extract_node_features, mesh_normalization_stats
from preprocessing.mesh_loader import export_mesh, mesh_from_arrays, validate_mesh
from utils.mesh_quality import quality_report
from utils.qem import WeightedQEMSimplifier, compute_edge_qem_scores

RAW_DIR = Path(__file__).resolve().parent / "raw"
PAIRS_DIR = Path(__file__).resolve().parent / "pairs"
PROCESSED_DIR = Path(__file__).resolve().parent / "processed"


def make_torus_knot(p: int = 3, q: int = 2, tube_radius: float = 0.3, u_res: int = 260, v_res: int = 36) -> trimesh.Trimesh:
    """Complex example: a (p,q) torus-knot tube -- twisting topology, high curvature variation, many faces."""
    u = np.linspace(0, 2 * np.pi, u_res, endpoint=False)
    r = np.cos(q * u) + 2.0
    centerline = np.stack([r * np.cos(p * u), r * np.sin(p * u), -np.sin(q * u)], axis=1)

    tangent = np.gradient(centerline, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    up = np.array([0.0, 0.0, 1.0])
    normal = np.cross(tangent, up)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    binormal = np.cross(tangent, normal)

    v = np.linspace(0, 2 * np.pi, v_res, endpoint=False)
    cos_v, sin_v = np.cos(v), np.sin(v)

    verts = np.zeros((u_res, v_res, 3))
    for i in range(u_res):
        verts[i] = centerline[i] + tube_radius * (np.outer(cos_v, normal[i]) + np.outer(sin_v, binormal[i]))
    verts = verts.reshape(-1, 3)

    faces = []
    for i in range(u_res):
        i_next = (i + 1) % u_res
        for j in range(v_res):
            j_next = (j + 1) % v_res
            a, b = i * v_res + j, i * v_res + j_next
            c, d = i_next * v_res + j, i_next * v_res + j_next
            faces.append([a, b, c])
            faces.append([b, d, c])

    return trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=True)


def make_organic_blob(subdivisions: int = 5, seed: int = 0) -> trimesh.Trimesh:
    """Complex example: a displaced icosphere -- organic sculpt-like surface, many faces."""
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions)
    rng = np.random.default_rng(seed)
    directions = mesh.vertices / np.linalg.norm(mesh.vertices, axis=1, keepdims=True)

    displacement = np.zeros(len(mesh.vertices))
    for freq, amp in [(3.1, 0.06), (5.7, 0.035), (11.3, 0.02), (23.0, 0.01)]:
        phase = rng.uniform(0, 2 * np.pi, size=3)
        displacement += amp * np.sin(
            freq * directions[:, 0] + phase[0] + freq * directions[:, 1] + phase[1] + freq * directions[:, 2] + phase[2]
        )

    new_vertices = mesh.vertices + directions * displacement[:, None]
    return trimesh.Trimesh(vertices=new_vertices, faces=mesh.faces, process=True)


def make_simple_box(subdivide_iters: int = 2) -> trimesh.Trimesh:
    """Simple example: a box, lightly subdivided -- flat, few features, few faces."""
    mesh = trimesh.creation.box(extents=[2.0, 1.2, 1.0])
    for _ in range(subdivide_iters):
        mesh = mesh.subdivide()
    return mesh


def make_simple_sphere(subdivisions: int = 1) -> trimesh.Trimesh:
    """Simple example: a low-subdivision icosphere -- rounded, very few faces."""
    return trimesh.creation.icosphere(subdivisions=subdivisions)


# (name, category, mesh factory, target face count for the "optimized" pair)
SAMPLES = [
    ("torus_knot_complex", "complex", make_torus_knot, 1500),
    ("organic_blob_complex", "complex", make_organic_blob, 1200),
    ("box_simple", "simple", make_simple_box, 48),
    ("icosphere_simple", "simple", make_simple_sphere, 40),
]


def build_graph(mesh: trimesh.Trimesh) -> Data:
    node_features = extract_node_features(mesh)
    edge_index_d, edge_attr_d, unique_edges, _ = extract_edge_features(mesh)
    qem_scores = compute_edge_qem_scores(mesh.vertices, mesh.faces)
    labels_unique = np.array([qem_scores[(int(i), int(j))] for i, j in unique_edges], dtype=np.float32)
    labels_directed = np.concatenate([labels_unique, labels_unique])
    return Data(
        x=torch.from_numpy(node_features),
        edge_index=torch.from_numpy(edge_index_d),
        edge_attr=torch.from_numpy(edge_attr_d),
        y=torch.from_numpy(labels_directed),
    )


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PAIRS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, category, factory, target_faces in SAMPLES:
        t0 = time.time()
        print(f"Generating {name} ({category}) ...")
        raw_mesh = factory()
        mesh = validate_mesh(mesh_from_arrays(raw_mesh.vertices, raw_mesh.faces))

        export_mesh(mesh, str(RAW_DIR / f"{name}.obj"))
        export_mesh(mesh, str(PAIRS_DIR / f"{name}_high.obj"))

        graph = build_graph(mesh)
        center, scale = mesh_normalization_stats(mesh)

        simplifier = WeightedQEMSimplifier(mesh.vertices, mesh.faces)
        low_v, low_f = simplifier.simplify(target_faces=target_faces, edge_importance=None)
        low_mesh = mesh_from_arrays(low_v, low_f)
        export_mesh(low_mesh, str(PAIRS_DIR / f"{name}_low.obj"))

        graph.target_vertices = torch.from_numpy(((low_mesh.vertices - center) / scale).astype(np.float32))
        graph.target_normals = torch.from_numpy(low_mesh.vertex_normals.astype(np.float32))
        torch.save(graph, PROCESSED_DIR / f"{name}.pt")

        report = quality_report(mesh, low_mesh)
        elapsed = time.time() - t0
        rows.append((name, category, len(mesh.faces), len(low_mesh.faces), report, elapsed))
        print(f"  {len(mesh.faces)} -> {len(low_mesh.faces)} faces in {elapsed:.1f}s")

    print("\n=== Sample dataset summary ===")
    header = f"{'name':24s} {'category':9s} {'faces_high':>11s} {'faces_low':>10s} {'reduction%':>11s} {'hausdorff':>10s} {'chamfer':>10s}"
    print(header)
    for name, category, fh, fl, report, elapsed in rows:
        print(
            f"{name:24s} {category:9s} {fh:11d} {fl:10d} "
            f"{report['face_reduction_percent']:11.1f} {report['hausdorff_distance']:10.4f} {report['chamfer_distance']:10.5f}"
        )

    print(f"\nRaw meshes       -> {RAW_DIR}")
    print(f"High/low pairs   -> {PAIRS_DIR}")
    print(f"Training graphs  -> {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
