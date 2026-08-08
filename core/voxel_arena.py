"""Slab-style memory arena for 3D voxel analysis.

Backed by a single anonymous ``mmap``.  The arena reserves a contiguous block
of virtual address space (no physical pages are committed until they are
actually written).  Allocations return NumPy views into the mmap; freeing a
slot returns it to the free list so that large 3D arrays can be reused without
asking the OS for new contiguous blocks.

Use as a context manager:

    with VoxelArena(reserve_bytes=4 * 1024**3) as arena:
        arr = arena.alloc(shape, np.float32, name='tmp')
        ...

The returned arrays are only valid until the arena is closed.
"""

import bisect
import mmap
from typing import Dict, List, Optional, Tuple

import numpy as np


class VoxelArena:
    """Monotonic arena with a first-fit free list for large NumPy arrays.

    Parameters
    ----------
    reserve_bytes:
        Total virtual address space to reserve.  This is *not* physical RAM;
        pages are committed by the OS only when the array is written.
    """

    def __init__(self, reserve_bytes: int):
        if reserve_bytes <= 0:
            raise ValueError("reserve_bytes must be positive")

        self._granularity = getattr(mmap, "ALLOCATIONGRANULARITY", mmap.PAGESIZE)
        self._reserve_bytes = self._round_up(reserve_bytes, self._granularity)

        try:
            self._mmap = mmap.mmap(-1, self._reserve_bytes, access=mmap.ACCESS_WRITE)
        except (OSError, OverflowError, MemoryError) as exc:
            raise MemoryError(
                f"VoxelArena: could not reserve {self._reserve_bytes / (1024 ** 3):.2f} GiB "
                f"of virtual address space ({exc}). Try a smaller reserve multiplier."
            ) from exc

        self._offset: int = 0
        self._free: List[Tuple[int, int]] = []
        self._used: Dict[str, Tuple[int, int]] = {}

    @staticmethod
    def _round_up(value: int, unit: int) -> int:
        return ((value + unit - 1) // unit) * unit

    @staticmethod
    def _align(value: int, itemsize: int) -> int:
        return ((value + itemsize - 1) // itemsize) * itemsize

    def _coalesce(self) -> None:
        """Merge adjacent free regions."""
        if not self._free:
            return
        self._free.sort()
        merged = [self._free[0]]
        for off, size in self._free[1:]:
            prev_off, prev_size = merged[-1]
            if prev_off + prev_size == off:
                merged[-1] = (prev_off, prev_size + size)
            else:
                merged.append((off, size))
        self._free = merged

    def alloc(self, shape, dtype, name: Optional[str] = None) -> np.ndarray:
        """Return a NumPy array view backed by the arena."""
        dtype = np.dtype(dtype)
        count = int(np.prod(shape, dtype=np.int64))
        bytes_needed = count * dtype.itemsize
        if bytes_needed < 0:
            raise ValueError("invalid allocation size")

        if bytes_needed == 0:
            return np.empty(shape, dtype=dtype)

        aligned, region_size = self._find_or_bump(bytes_needed, dtype.itemsize)
        end = aligned + region_size
        if end > self._reserve_bytes:
            used = self.used_bytes()
            raise MemoryError(
                f"VoxelArena exhausted: need {bytes_needed} bytes "
                f"({region_size} aligned) but only "
                f"{self._reserve_bytes - self._offset} bytes left at the bump and "
                f"{self._free_bytes()} bytes in free slots; "
                f"used={used}, reserved={self._reserve_bytes}."
            )

        if name is not None:
            self._used[name] = (aligned, region_size)

        arr = np.frombuffer(
            self._mmap,
            dtype=dtype,
            count=count,
            offset=aligned,
        )
        arr.shape = shape
        return arr

    def _find_or_bump(self, bytes_needed: int, itemsize: int) -> Tuple[int, int]:
        """Find a free slot or bump from the top.  Returns (offset, region_size)."""
        best_idx = -1
        best_waste = None
        for i, (off, size) in enumerate(self._free):
            aligned_off = self._align(off, itemsize)
            pad = aligned_off - off
            if pad + bytes_needed <= size:
                waste = size - (pad + bytes_needed)
                if best_waste is None or waste < best_waste:
                    best_waste = waste
                    best_idx = i

        if best_idx >= 0:
            off, size = self._free.pop(best_idx)
            aligned_off = self._align(off, itemsize)
            pad = aligned_off - off
            remaining = size - pad - bytes_needed
            if pad:
                self._free.append((off, pad))
            if remaining:
                self._free.append((aligned_off + bytes_needed, remaining))
            self._coalesce()
            return aligned_off, bytes_needed

        aligned = self._align(self._offset, itemsize)
        end = aligned + bytes_needed
        if end <= self._reserve_bytes:
            self._offset = end
        return aligned, bytes_needed

    def free(self, name: Optional[str] = None) -> None:
        """Return a named slot to the free list."""
        if name is None or name not in self._used:
            return
        off, size = self._used.pop(name)
        bisect.insort(self._free, (off, size))
        self._coalesce()

    def used_bytes(self) -> int:
        return sum(size for _, size in self._used.values())

    def free_bytes(self) -> int:
        return sum(size for _, size in self._free) + (self._reserve_bytes - self._offset)

    def total_bytes(self) -> int:
        return self._reserve_bytes

    def reset(self) -> None:
        """Reset the arena, invalidating all outstanding views."""
        self._offset = 0
        self._free.clear()
        self._used.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self) -> None:
        if hasattr(self, "_mmap") and self._mmap is not None:
            try:
                self._mmap.close()
            except BufferError:
                # Arrays may still be alive in the caller's scope.  Drop the
                # arena's reference; the mmap object will be unmapped by its
                # finalizer once the array views are released.
                pass
            self._mmap = None
