"""
AI-MeshOptimizer inference CLI.

    python inference/remesh.py dragon.obj dragon_low.obj --target_faces 20000

Pipeline:
    High Poly Mesh
      -> Mesh Feature Extraction   (preprocessing/feature_extractor.py)
      -> Graph Neural Network      (models/mesh_gnn.py)
      -> Edge Importance Prediction
      -> Feature Weighted QEM Simplification   (utils/qem.py)
      -> Export OBJ/PLY/STL

If no trained checkpoint is found, the tool still runs end-to-end using
plain (unweighted) QEM simplification, with a warning -- there is no hard
dependency on a pretrained model to be useful.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.mesh_gnn import MeshGNN
from preprocessing.feature_extractor import extract_edge_features, extract_node_features
from preprocessing.mesh_loader import export_mesh, load_mesh, mesh_from_arrays
from utils.mesh_quality import quality_report
from utils.qem import WeightedQEMSimplifier


def get_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_edge_importance(mesh, checkpoint_path: str, device: torch.device):
    """Returns a dict {(i, j): importance in [0, 1]} for every unique edge, or None if no checkpoint."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        print(f"[WARN] Checkpoint '{checkpoint_path}' not found -- falling back to unweighted QEM.")
        return None

    node_features = extract_node_features(mesh)
    edge_index_directed, edge_attr_directed, unique_edges, _ = extract_edge_features(mesh)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args = checkpoint.get("args", {})
    model = MeshGNN(
        hidden_dim=model_args.get("hidden_dim", 64),
        gat_heads=model_args.get("gat_heads", 4),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x = torch.from_numpy(node_features).to(device)
    edge_index = torch.from_numpy(edge_index_directed).to(device)
    edge_attr = torch.from_numpy(edge_attr_directed).to(device)

    with torch.no_grad():
        scores_directed = model(x, edge_index, edge_attr)
        unique_scores = MeshGNN.to_undirected_scores(scores_directed).cpu().numpy()

    return {(int(i), int(j)): float(s) for (i, j), s in zip(unique_edges, unique_scores)}


def remesh(
    input_path: str,
    output_path: str,
    target_faces: int,
    checkpoint_path: str = "checkpoints/best_model.pt",
    importance_strength: float = 8.0,
    device_str: str = "auto",
    compare: bool = False,
):
    device = get_device(device_str)

    print(f"Loading '{input_path}' ...")
    mesh = load_mesh(input_path)
    original_faces = len(mesh.faces)
    print(f"Original mesh: {len(mesh.vertices)} vertices, {original_faces} faces")

    if target_faces >= original_faces:
        print(
            f"[WARN] target_faces ({target_faces}) >= original face count ({original_faces}); "
            "exporting the (cleaned) input mesh unchanged."
        )
        export_mesh(mesh, output_path)
        return mesh, mesh

    edge_importance = predict_edge_importance(mesh, checkpoint_path, device)
    mode = "GNN feature-weighted QEM" if edge_importance is not None else "plain QEM"
    print(f"Simplifying {original_faces} -> {target_faces} faces using {mode} ...")

    t0 = time.time()
    simplifier = WeightedQEMSimplifier(mesh.vertices, mesh.faces)
    new_vertices, new_faces = simplifier.simplify(
        target_faces=target_faces,
        edge_importance=edge_importance,
        importance_strength=importance_strength,
    )
    elapsed = time.time() - t0

    simplified = mesh_from_arrays(new_vertices, new_faces)
    export_mesh(simplified, output_path)

    print(f"Done in {elapsed:.1f}s. Output: {len(simplified.vertices)} vertices, {len(simplified.faces)} faces")
    print(f"Saved -> {output_path}")

    if compare:
        print("\nComputing quality report (this samples the surfaces, may take a few seconds)...")
        report = quality_report(mesh, simplified)
        print("\n=== Quality Report ===")
        for key, value in report.items():
            if isinstance(value, float):
                print(f"{key:28s}: {value:.6f}")
            else:
                print(f"{key:28s}: {value}")

    return mesh, simplified


def main():
    parser = argparse.ArgumentParser(description="AI-MeshOptimizer: GNN-guided feature-weighted QEM remeshing.")
    parser.add_argument("input", help="Input mesh path (.obj / .ply / .stl)")
    parser.add_argument("output", help="Output mesh path (.obj / .ply / .stl)")
    parser.add_argument("--target_faces", type=int, required=True, help="Target triangle count.")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt", help="Path to trained model weights.")
    parser.add_argument(
        "--importance_strength",
        type=float,
        default=8.0,
        help="How strongly GNN-predicted importance inflates edge-collapse cost.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--compare", action="store_true", help="Print a Hausdorff/Chamfer/normal quality report.")
    args = parser.parse_args()

    remesh(
        input_path=args.input,
        output_path=args.output,
        target_faces=args.target_faces,
        checkpoint_path=args.checkpoint,
        importance_strength=args.importance_strength,
        device_str=args.device,
        compare=args.compare,
    )


if __name__ == "__main__":
    main()
