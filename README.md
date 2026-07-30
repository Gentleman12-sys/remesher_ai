# AI-MeshOptimizer

An open-source, learned mesh decimator: a Graph Neural Network predicts per-edge
"importance" on a high-poly mesh, and that prediction re-weights a classic
Quadric Error Metrics (QEM) edge-collapse simplifier so sharp features, seams and
silhouettes survive aggressive polycount reduction. Think "a learned, open
ZRemesher component" rather than a full retopology suite.

```
High Poly Mesh (.obj/.ply/.stl)
        |
        v
Mesh Feature Extraction        (preprocessing/feature_extractor.py)
        |
        v
Graph Neural Network            (models/mesh_gnn.py)
        |
        v
Edge Importance Prediction      (0 = collapse freely, 1 = preserve)
        |
        v
Feature-Weighted QEM            (utils/qem.py)
        |
        v
Low Poly Mesh (.obj/.ply/.stl)
```

## Why feature-weighted QEM instead of plain QEM?

Vanilla QEM only looks at local geometric error -- it happily removes short,
sharp edges (a nostril, a mechanical detail) if the plane-fit error is small.
The GNN instead sees graph-wide context (curvature, dihedral angle, boundary/UV
seams, valence, propagated through message passing) and learns which edges
*look like* the ones that survive an independent, high-quality simplification.
Its prediction inflates the collapse cost of important edges so they are
skipped even when their raw QEM cost is low.

## Project layout

```
AI-MeshOptimizer/
├── dataset/
│   ├── make_sample_dataset.py  # generates a license-clean complex+simple demo set
│   ├── raw/          # high-poly source meshes (ships with the sample dataset)
│   ├── processed/     # generated PyG graphs (.pt) -- one per mesh
│   └── pairs/          # generated {name}_high.obj / {name}_low.obj pairs
├── models/
│   └── mesh_gnn.py     # GraphConv -> GraphConv -> GATConv -> MLP
├── preprocessing/
│   ├── mesh_loader.py       # load/validate/export .obj .ply .stl
│   ├── feature_extractor.py # per-vertex / per-edge geometric features
│   └── generate_pairs.py    # builds the training set
├── training/
│   ├── dataset.py   # PyG Dataset over dataset/processed/*.pt
│   ├── losses.py    # combined BCE + Chamfer + Normal + Curvature + EdgeReg loss
│   ├── metrics.py   # accuracy / precision / recall / F1
│   └── train.py     # full training loop, checkpointing, plots
├── inference/
│   └── remesh.py    # CLI: predict + feature-weighted QEM simplify + export
├── utils/
│   ├── qem.py            # from-scratch weighted QEM edge-collapse simplifier
│   └── mesh_quality.py   # Hausdorff / Chamfer / normal deviation / triangle quality
├── requirements.txt
├── Dockerfile
└── AI_Remesher_Training.ipynb   # Google Colab (free T4) training notebook
```

## Installation

PyTorch and PyTorch Geometric are hardware-specific, so install them first.

**CPU only:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
pip install -r requirements.txt
```

**CUDA (example: CUDA 12.1):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric
pip install -r requirements.txt
```

**Google Colab:** open `AI_Remesher_Training.ipynb` -- it installs everything
in the correct order for the Colab Free T4 runtime.

**Docker:**
```bash
docker build -t ai-meshoptimizer .
docker run --rm -v "$PWD:/data" ai-meshoptimizer /data/input.obj /data/output.obj --target_faces 20000
```

## Dataset

Any collection of watertight-ish triangle meshes in `.obj` / `.ply` / `.stl`
works. Recommended open sources:

- [ABC Dataset](https://deep-geometry.github.io/abc-dataset/) (primary -- CAD-like sharp features)
- [Objaverse](https://objaverse.allenai.org/) (secondary -- organic/game assets)
- [Thingi10K](https://ten-thousand-models.appspot.com/) (secondary -- 3D-printable models)

Well-known GitHub mesh repos (`alecjacobson/common-3d-test-models`,
`libigl/libigl-tutorial-data`, Stanford Bunny/Armadillo/Dragon-style files)
are commonly used for research but **do not carry a clear CC0/MIT-style
license on the meshes themselves** -- they're provenance-attributed research
fixtures, not confirmed public-domain redistributables. This repo doesn't
vendor them for that reason; if you want to use them, pull them into your own
`dataset/raw/` and check the license situation for your use case first.

### Procedural sample dataset (included, license-clean, no download required)

`dataset/make_sample_dataset.py` procedurally generates raw meshes across four
categories -- no external download, no licensing ambiguity, everything built
from closed-form formulas with trimesh + numpy:

| category | meaning | typical faces |
|---|---|---:|
| `complex_knot` | **complex**: twisting torus-knot tubes, high curvature variation | ~5,000-14,000 |
| `organic_blob` | **complex**: noise-displaced icospheres, sculpt-like surfaces | ~1,300-20,500 |
| `box_simple` | **simple**: lightly subdivided boxes, flat, few features | ~50-770 |
| `sphere_simple` | **simple**: low-subdivision icospheres, rounded, few features | ~80-1,280 |

Each instance in a category randomizes its generating parameters (knot
winding numbers, tube radius, resolution, noise frequency/seed, box extents,
subdivision level, scale, rotation, ...), so `--count_per_category 750` gives
750 *genuinely different* shapes per category, not copies.

A small demo set (3 per category, already paired and processed) ships in the
repo under `dataset/{raw,pairs,processed}/demo/`. For real training, generate
at scale -- e.g. 500-1000 per category as recommended above:

```bash
# 1. generate raw meshes (fast: a few minutes for 750/category)
python dataset/make_sample_dataset.py --count_per_category 750 --out_dir dataset/raw

# 2. build training graphs -- one call per category, tuned reduction ratios
#    (complex meshes need aggressive reduction, simple ones need a gentle touch)
python preprocessing/generate_pairs.py --input_dir dataset/raw/complex_knot   --reduction_ratio 0.08 --min_faces 200
python preprocessing/generate_pairs.py --input_dir dataset/raw/organic_blob  --reduction_ratio 0.08 --min_faces 200
python preprocessing/generate_pairs.py --input_dir dataset/raw/box_simple    --reduction_ratio 0.4  --min_faces 20
python preprocessing/generate_pairs.py --input_dir dataset/raw/sphere_simple --reduction_ratio 0.4  --min_faces 20
```

(all four calls write into the same shared `dataset/processed/` /
`dataset/pairs/` -- filenames are already namespaced by category, e.g.
`complex_knot_0347.obj`, so there's no collision.) `AI_Remesher_Training.ipynb`
runs exactly this, with the count controlled by one variable you set at the
top of the notebook.

Inspect any pair directly:

```bash
python inference/remesh.py dataset/raw/demo/complex_knot/complex_knot_0000.obj out.obj --target_faces 500 --compare
```

This dataset (procedural primitives/knots/blobs) is meant to give the model
real complex-vs-simple structural variety to learn from without any
licensing/download friction. For maximum realism, replace or augment it with
ABC/Objaverse/Thingi10K (below).

Drop your own raw meshes into `dataset/raw/` (flat or in subfolders --
`generate_pairs.py --input_dir` searches recursively), then build the
training set:

```bash
python preprocessing/generate_pairs.py \
    --input_dir dataset/raw \
    --reduction_ratio 0.1 \
    --min_faces 100
```

This writes one `torch_geometric.data.Data` graph per mesh to
`dataset/processed/` (node features, edge features, QEM-derived importance
labels, and a normalized copy of an independently-generated low-poly target
mesh used for the Chamfer/normal loss terms), plus `{name}_high.obj` /
`{name}_low.obj` pairs to `dataset/pairs/`.

The low-poly target is generated with [`pyfqmr`](https://github.com/Kramer84/pyfqmr-Fast-Quadric-Mesh-Reduction)
(a fast C++ QEM decimator, used here as an accessible, pip-installable stand-in
for Instant Meshes / QuadriFlow). If `pyfqmr` is unavailable in your
environment, `generate_pairs.py` automatically falls back to this project's
own unweighted `utils.qem.WeightedQEMSimplifier`, so the pipeline never hard-fails.

### Feature vectors

Per-vertex node features (8-dim): `[x, y, z, nx, ny, nz, curvature, valence]`

Per-edge features (8-dim): `[length, dihedral_angle, curvature, valence,
normal_difference, boundary, uv_seam, face_area]`

## Training

```bash
python training/train.py \
    --processed_dir dataset/processed \
    --checkpoint_dir checkpoints \
    --epochs 100 \
    --batch_size 8 \
    --lr 0.0001
```

- Optimizer: `AdamW`, scheduler: `CosineAnnealingLR`
- Loss: `0.3*BCE + 0.3*Chamfer + 0.2*Normal + 0.1*Curvature + 0.1*EdgeReg` (`training/losses.py`)
- Saves `checkpoints/best_model.pt` (lowest validation loss) and
  `checkpoints/training_curves.png` (loss / accuracy / precision-recall-F1)

Or run the whole pipeline (GPU check -> install -> preprocess -> train -> plot)
in `AI_Remesher_Training.ipynb` on Google Colab's free T4 tier.

## Inference

```bash
python inference/remesh.py dragon.obj dragon_low.obj --target_faces 20000
```

```bash
python inference/remesh.py dragon.obj dragon_low.obj \
    --target_faces 20000 \
    --checkpoint checkpoints/best_model.pt \
    --importance_strength 8.0 \
    --compare
```

`--compare` prints a Hausdorff / Chamfer / normal-deviation / triangle-quality
report comparing the output against the input. If no checkpoint is found, the
tool still runs end-to-end using plain (unweighted) QEM, with a warning --
there's no hard dependency on a pretrained model.

## Quality metrics (`utils/mesh_quality.py`)

- **Hausdorff distance** -- worst-case surface deviation (sampled approximation)
- **Chamfer distance** -- mean squared nearest-surface distance
- **Normal deviation** -- mean angular difference between nearest-point normals
- **Triangle quality** -- `4*sqrt(3)*Area / (l0^2+l1^2+l2^2)`, mean and min
- **Face reduction %** -- how much the polycount dropped

## Sanity-check benchmark

A small, fast, CPU-only end-to-end run (icosphere, 1280 -> 320 faces, 75%
reduction) to sanity-check the pipeline -- not a claim of state-of-the-art
quality, just evidence every stage works and produces sane numbers:

| Metric | Unweighted QEM | GNN Feature-Weighted QEM |
|---|---|---|
| Faces | 320 | 320 |
| Hausdorff distance | 0.129 | 0.130 |
| Chamfer distance | 0.00272 | 0.00273 |
| Mean normal deviation | 5.8° | 6.3° |
| Mean triangle quality | 0.78 | 0.72 |

On a symmetric primitive like a sphere the two are expected to be close (there
are no sharp features for the GNN to protect); the gap should widen in favor
of the feature-weighted path on meshes with real sharp/CAD-like features and a
model trained on a real dataset (ABC, Objaverse, Thingi10K) rather than a
handful of epochs on synthetic primitives. Re-run
`python inference/remesh.py <mesh> <out> --target_faces N --compare` after
training on your own data to get numbers for your use case.

## How it works, in one paragraph

`preprocessing/generate_pairs.py` computes, for every edge of every training
mesh, a closed-form QEM collapse cost (`utils/qem.py`), rank-normalizes it to
`[0, 1]`, and uses it as the supervised target for `models/mesh_gnn.py`
(`GraphConv -> GraphConv -> GATConv -> MLP`, `training/losses.py`'s BCE term).
Three auxiliary, differentiable regularizers nudge the network past pure
QEM-mimicry: a weight-constrained Chamfer/normal loss rewards assigning high
importance to vertices that stay close (in position and normal) to an
independently-generated low-poly target mesh; a curvature regression term
rewards tracking local surface curvature; and a graph-Laplacian-style edge
regularizer keeps predictions spatially coherent. At inference
(`inference/remesh.py`), the trained model's prediction on an unseen mesh
feeds into `utils/qem.WeightedQEMSimplifier`, which inflates the collapse cost
of important edges by `1 + importance_strength * importance` before running
standard greedy edge-collapse decimation to the requested face count.

## License

MIT -- see [LICENSE](LICENSE). Fully open source; no paid services required to
train (Google Colab Free / T4) or run.
