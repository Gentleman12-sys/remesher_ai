"""Thin PyG-compatible dataset over the .pt graph files produced by preprocessing/generate_pairs.py."""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch
from torch_geometric.data import Data, Dataset


class MeshPairDataset(Dataset):
    def __init__(self, processed_dir: str):
        super().__init__()
        self.paths: List[Path] = sorted(Path(processed_dir).glob("*.pt"))
        if not self.paths:
            raise FileNotFoundError(
                f"No processed graphs found in '{processed_dir}'. "
                "Run preprocessing/generate_pairs.py first."
            )

    def len(self) -> int:
        return len(self.paths)

    def get(self, idx: int) -> Data:
        return torch.load(self.paths[idx], weights_only=False)
