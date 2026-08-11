"""Fibonacci-sphere lookup for 64-direction SFER sampling."""

from pathlib import Path
from typing import Tuple

import numpy as np


def _fibonacci_sphere(n: int) -> np.ndarray:
    """Return ``n`` nearly-uniform points on the unit sphere."""
    dirs = np.empty((n, 3), dtype=np.float64)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        y = 1.0 - (i / (n - 1)) * 2.0
        r = np.sqrt(max(1.0 - y * y, 0.0))
        theta = golden * i
        dirs[i, 0] = r * np.cos(theta)
        dirs[i, 1] = y
        dirs[i, 2] = r * np.sin(theta)
    return dirs


def _build_adjacency(dirs: np.ndarray, k: int = 6) -> np.ndarray:
    """For each direction, indices of the ``k`` closest directions."""
    n = dirs.shape[0]
    adj = np.empty((n, k), dtype=np.int64)
    for i in range(n):
        dots = dirs @ dirs[i]
        dots[i] = -2.0  # exclude self
        adj[i] = np.argpartition(dots, -k)[-k:]
    return adj


def get_sphere_64_6(data_dir: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load or generate the 64-point Fibonacci sphere and 6-neighbour adjacency.
    Cached as ``.npy`` files under ``core/data`` (or ``data_dir``).
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent / "data"
    else:
        data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    dirs_path = data_dir / "sphere_dirs_64.npy"
    adj_path = data_dir / "sphere_adj_64x6.npy"
    if dirs_path.exists() and adj_path.exists():
        return np.load(dirs_path), np.load(adj_path)

    dirs = _fibonacci_sphere(64)
    adj = _build_adjacency(dirs, k=6)
    np.save(dirs_path, dirs)
    np.save(adj_path, adj)
    return dirs, adj
