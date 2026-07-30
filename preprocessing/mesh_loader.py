"""
Mesh I/O: loading, validation/cleanup and export for .obj / .ply / .stl.

All other modules in this project should go through load_mesh()/export_mesh()
rather than calling trimesh directly, so cleanup behavior stays consistent
across preprocessing, training and inference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

SUPPORTED_EXTENSIONS = {".obj", ".ply", ".stl"}


def _drop_degenerate_faces(mesh: trimesh.Trimesh) -> None:
    """In-place removal of zero-area / duplicate-vertex faces (trimesh>=4 dropped remove_degenerate_faces)."""
    mesh.update_faces(mesh.nondegenerate_faces())


def load_mesh(path: str, merge_vertices: bool = True) -> trimesh.Trimesh:
    """Load a mesh, merge duplicate vertices/degenerate faces, and ensure consistent winding."""
    path = str(path)
    ext = Path(path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported mesh format '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")

    loaded = trimesh.load(path, process=False, force="mesh")
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"'{path}' did not load as a single triangle mesh (got {type(loaded)}).")

    mesh = loaded
    if merge_vertices:
        mesh.merge_vertices()
    _drop_degenerate_faces(mesh)
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()

    return validate_mesh(mesh)


def validate_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Basic sanity checks; raises on empty/non-manifold-beyond-repair meshes."""
    if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
        raise ValueError("Mesh has no vertices or no faces after cleanup.")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("Mesh contains non-finite vertex coordinates (NaN/Inf).")
    if mesh.faces.max() >= mesh.vertices.shape[0]:
        raise ValueError("Mesh face indices reference out-of-range vertices.")
    return mesh


def export_mesh(mesh: trimesh.Trimesh, path: str) -> None:
    """Export a mesh, inferring format from the file extension."""
    ext = Path(path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported export format '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)


def mesh_from_arrays(vertices: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    """Build and lightly clean a Trimesh from raw vertex/face arrays (e.g. simplifier output)."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    _drop_degenerate_faces(mesh)
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    return mesh
