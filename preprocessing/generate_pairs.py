"""
Build the training set: for every raw high-poly mesh in dataset/raw/, generate

  1. A mesh graph with node/edge features             -> dataset/processed/{name}.pt
     (a torch_geometric.data.Data object with x, edge_index, edge_attr, y)
  2. A {name}_high.obj / {name}_low.obj pair           -> dataset/pairs/
     used for Chamfer/normal supervision and benchmarking.

The "ground truth" edge-importance label (y) comes from the closed-form QEM
collapse cost of every edge on the ORIGINAL high-poly mesh (utils/qem.py):
high cost -> geometrically critical -> label close to 1 ("preserve").
This is what the GNN is trained to predict, and at inference time its
prediction is used to *re-weight* QEM costs on unseen meshes
(inference/remesh.py), which is the actual "Feature Weighted QEM
Simplification" step.

The target_low_poly mesh is produced independently with pyfqmr (a fast,
well-tested C++ quadric decimator) as a stand-in for Instant Meshes /
QuadriFlow, which are not pip-installable/cross-platform. If pyfqmr is not
available in the environment, this script falls back to our own
WeightedQEMSimplifier (utils/qem.py, unweighted) so the pipeline still runs
end-to-end without any third-party mesh decimation binary.

Usage:
    python preprocessing/generate_pairs.py \
        --input_dir dataset/raw \
        --reduction_ratio 0.1 \
        --limit 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from preprocessing.feature_extractor import extract_edge_features, extract_node_features, mesh_normalization_stats
from preprocessing.mesh_loader import SUPPORTED_EXTENSIONS, export_mesh, load_mesh, mesh_from_arrays
from utils.qem import WeightedQEMSimplifier, compute_edge_qem_scores


def _simplify_target(vertices: np.ndarray, faces: np.ndarray, target_faces: int):
    """Generate the target low-poly mesh, preferring pyfqmr, falling back to our own QEM."""
    try:
        import pyfqmr

        simplifier = pyfqmr.Simplify()
        simplifier.setMesh(vertices, faces)
        simplifier.simplify_mesh(target_count=target_faces, aggressiveness=7, preserve_border=True, verbose=False)
        new_vertices, new_faces, _ = simplifier.getMesh()
        return np.asarray(new_vertices), np.asarray(new_faces)
    except Exception:
        simplifier = WeightedQEMSimplifier(vertices, faces)
        return simplifier.simplify(target_faces=target_faces, edge_importance=None)


def process_mesh(path: Path, reduction_ratio: float) -> dict:
    mesh = load_mesh(str(path))

    node_features = extract_node_features(mesh)
    edge_index_directed, edge_attr_directed, unique_edges, _ = extract_edge_features(mesh)

    qem_scores = compute_edge_qem_scores(mesh.vertices, mesh.faces)
    labels_unique = np.array(
        [qem_scores[(int(i), int(j))] for i, j in unique_edges], dtype=np.float32
    )
    # duplicated to match edge_index_directed's [forward; reverse] layout
    labels_directed = np.concatenate([labels_unique, labels_unique])

    graph = Data(
        x=torch.from_numpy(node_features),
        edge_index=torch.from_numpy(edge_index_directed),
        edge_attr=torch.from_numpy(edge_attr_directed),
        y=torch.from_numpy(labels_directed),
    )
    graph.num_original_faces = len(mesh.faces)

    target_faces = max(4, int(len(mesh.faces) * reduction_ratio))
    low_vertices, low_faces = _simplify_target(mesh.vertices, mesh.faces, target_faces)
    low_mesh = mesh_from_arrays(low_vertices, low_faces)

    # Target mesh, normalized with the SAME center/scale as node_features, so
    # Chamfer/normal losses (training/losses.py) compare like-for-like coordinates.
    center, scale = mesh_normalization_stats(mesh)
    graph.target_vertices = torch.from_numpy(((low_mesh.vertices - center) / scale).astype(np.float32))
    graph.target_normals = torch.from_numpy(low_mesh.vertex_normals.astype(np.float32))

    return {"graph": graph, "high_mesh": mesh, "low_mesh": low_mesh}


def main():
    parser = argparse.ArgumentParser(description="Generate AI-MeshOptimizer training pairs.")
    parser.add_argument("--input_dir", default="dataset/raw", help="Directory of raw high-poly meshes.")
    parser.add_argument("--processed_dir", default="dataset/processed", help="Output dir for graph .pt files.")
    parser.add_argument("--pairs_dir", default="dataset/pairs", help="Output dir for high/low mesh pairs.")
    parser.add_argument("--reduction_ratio", type=float, default=0.1, help="target_faces = ratio * original_faces.")
    parser.add_argument("--min_faces", type=int, default=100, help="Skip meshes smaller than this.")
    parser.add_argument("--limit", type=int, default=0, help="Max number of meshes to process (0 = no limit).")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    processed_dir = Path(args.processed_dir)
    pairs_dir = Path(args.pairs_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    pairs_dir.mkdir(parents=True, exist_ok=True)

    mesh_files = sorted(
        p for p in input_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if args.limit > 0:
        mesh_files = mesh_files[: args.limit]

    if not mesh_files:
        print(f"No meshes found in {input_dir} (supported: {sorted(SUPPORTED_EXTENSIONS)}).")
        return

    ok, skipped, failed = 0, 0, 0
    for path in tqdm(mesh_files, desc="Generating pairs"):
        name = path.stem
        try:
            mesh_peek = load_mesh(str(path))
            if len(mesh_peek.faces) < args.min_faces:
                skipped += 1
                continue

            result = process_mesh(path, args.reduction_ratio)

            torch.save(result["graph"], processed_dir / f"{name}.pt")
            export_mesh(result["high_mesh"], str(pairs_dir / f"{name}_high.obj"))
            export_mesh(result["low_mesh"], str(pairs_dir / f"{name}_low.obj"))
            ok += 1
        except Exception as exc:  # keep the batch running even if one mesh is malformed
            print(f"[WARN] Failed on {path.name}: {exc}")
            failed += 1

    print(f"\nDone. processed={ok} skipped={skipped} failed={failed}")
    print(f"Graphs   -> {processed_dir}")
    print(f"Pairs    -> {pairs_dir}")


if __name__ == "__main__":
    main()
