"""
Procedural raw-mesh generator for AI-MeshOptimizer, spanning four categories:

  complex_knot   -- twisting torus-knot tubes (complex topology, high curvature variation)
  organic_blob   -- noise-displaced icospheres (organic, sculpt-like surfaces)
  box_simple     -- lightly subdivided boxes (simple, flat, few features)
  sphere_simple  -- low-subdivision icospheres (simple, rounded, few features)

Every mesh in a category is generated from the same procedure with randomized
parameters (knot winding numbers, tube radius, resolution, noise seed/
frequency, box extents, subdivision level, scale, rotation, ...), so a large
`--count_per_category` produces genuine geometric variety rather than
identical copies -- meaningful for training, not just a placeholder dataset.

No external downloads, no licensing questions: everything is built with
trimesh + numpy from closed-form / procedural formulas.

This script ONLY generates raw meshes (dataset/raw/<category>/*.obj). It does
NOT run QEM/feature extraction -- that's preprocessing/generate_pairs.py's
job (run once per category, since complex and simple meshes want different
--reduction_ratio). See README.md > Dataset for the exact commands, or just
run AI_Remesher_Training.ipynb, which wires both stages together.

Usage:
    # small demo (fast, a handful of meshes per category)
    python dataset/make_sample_dataset.py --count_per_category 3 --out_dir dataset/raw/demo

    # full training-scale dataset (500-1000 per category; several minutes)
    python dataset/make_sample_dataset.py --count_per_category 750 --out_dir dataset/raw
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from preprocessing.mesh_loader import export_mesh, mesh_from_arrays, validate_mesh

CATEGORIES = ["complex_knot", "organic_blob", "box_simple", "sphere_simple"]
CATEGORY_SEED_ID = {name: i for i, name in enumerate(CATEGORIES)}


def _make_torus_knot(p: int, q: int, tube_radius: float, u_res: int, v_res: int) -> trimesh.Trimesh:
    """A (p,q) torus-knot tube -- twisting topology, high curvature variation, many faces."""
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


def _make_organic_blob(subdivisions: int, rng: np.random.Generator) -> trimesh.Trimesh:
    """A noise-displaced icosphere -- organic sculpt-like surface, many faces."""
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions)
    directions = mesh.vertices / np.linalg.norm(mesh.vertices, axis=1, keepdims=True)

    n_bands = int(rng.integers(3, 6))
    displacement = np.zeros(len(mesh.vertices))
    for _ in range(n_bands):
        freq = float(rng.uniform(2.5, 25.0))
        amp = float(rng.uniform(0.01, 0.07))
        phase = rng.uniform(0, 2 * np.pi, size=3)
        displacement += amp * np.sin(
            freq * directions[:, 0] + phase[0] + freq * directions[:, 1] + phase[1] + freq * directions[:, 2] + phase[2]
        )

    new_vertices = mesh.vertices + directions * displacement[:, None]
    return trimesh.Trimesh(vertices=new_vertices, faces=mesh.faces, process=True)


def _random_transform(mesh: trimesh.Trimesh, rng: np.random.Generator, scale_range=(0.7, 1.4)) -> trimesh.Trimesh:
    """Random anisotropic scale + rotation, so instances differ in pose, not just topology."""
    mesh.apply_scale(rng.uniform(*scale_range, size=3))
    mesh.apply_transform(trimesh.transformations.random_rotation_matrix(rng.random(3)))
    return mesh


def sample_complex_knot(rng: np.random.Generator) -> trimesh.Trimesh:
    p = int(rng.integers(2, 6))
    q = int(rng.integers(1, 4))
    if p == q:
        q = q + 1 if q + 1 < p else max(1, q - 1)
    tube_radius = float(rng.uniform(0.22, 0.38))
    u_res = int(rng.integers(120, 220))
    v_res = int(rng.integers(20, 32))
    mesh = _make_torus_knot(p, q, tube_radius, u_res, v_res)
    return _random_transform(mesh, rng)


def sample_organic_blob(rng: np.random.Generator) -> trimesh.Trimesh:
    subdivisions = int(rng.choice([3, 3, 4, 4, 5]))  # weighted toward smaller -> keeps average size down
    mesh = _make_organic_blob(subdivisions, rng)
    return _random_transform(mesh, rng)


def sample_box_simple(rng: np.random.Generator) -> trimesh.Trimesh:
    extents = rng.uniform(0.6, 2.5, size=3)
    subdivide_iters = int(rng.integers(1, 4))
    mesh = trimesh.creation.box(extents=extents)
    for _ in range(subdivide_iters):
        mesh = mesh.subdivide()
    return _random_transform(mesh, rng, scale_range=(0.9, 1.1))


def sample_sphere_simple(rng: np.random.Generator) -> trimesh.Trimesh:
    subdivisions = int(rng.integers(1, 4))
    radius = float(rng.uniform(0.5, 1.5))
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    return _random_transform(mesh, rng, scale_range=(0.9, 1.1))


GENERATORS = {
    "complex_knot": sample_complex_knot,
    "organic_blob": sample_organic_blob,
    "box_simple": sample_box_simple,
    "sphere_simple": sample_sphere_simple,
}


def generate_category(category: str, count: int, out_dir: Path, seed: int) -> list:
    generator = GENERATORS[category]
    category_dir = out_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)

    face_counts = []
    for idx in tqdm(range(count), desc=category, leave=False):
        # SeedSequence-style per-item seeding: regenerating a single index later
        # (e.g. after bumping --count_per_category) reproduces the same mesh.
        # (uses CATEGORY_SEED_ID, not Python's hash(), which is randomized per process)
        rng = np.random.default_rng((seed, CATEGORY_SEED_ID[category], idx))
        raw_mesh = generator(rng)
        mesh = validate_mesh(mesh_from_arrays(raw_mesh.vertices, raw_mesh.faces))
        export_mesh(mesh, str(category_dir / f"{category}_{idx:04d}.obj"))
        face_counts.append(len(mesh.faces))

    return face_counts


def main():
    parser = argparse.ArgumentParser(description="Generate procedural raw meshes for AI-MeshOptimizer.")
    parser.add_argument("--count_per_category", type=int, default=4, help="Meshes to generate per category.")
    parser.add_argument("--out_dir", default="dataset/raw", help="Output directory (one subfolder per category).")
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=CATEGORIES)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    print(f"Generating {args.count_per_category} mesh(es) per category into {out_dir}/ ...")

    summary = []
    for category in args.categories:
        face_counts = generate_category(category, args.count_per_category, out_dir, args.seed)
        summary.append((category, face_counts))

    print("\n=== Generated raw meshes ===")
    print(f"{'category':16s} {'count':>7s} {'min_faces':>10s} {'mean_faces':>11s} {'max_faces':>10s}")
    for category, face_counts in summary:
        fc = np.array(face_counts)
        print(f"{category:16s} {len(fc):7d} {fc.min():10d} {fc.mean():11.0f} {fc.max():10d}")

    print(f"\nRaw meshes -> {out_dir}/<category>/")
    print(
        "Next: build training graphs per category with preprocessing/generate_pairs.py "
        "(complex categories want a low --reduction_ratio, e.g. 0.08; simple categories "
        "want a higher one, e.g. 0.4 -- see README.md > Dataset)."
    )


if __name__ == "__main__":
    main()
