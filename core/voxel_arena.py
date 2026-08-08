"""Chunked, on-demand memory arena for 3D voxel analysis.

The arena is backed by many anonymous ``mmap`` blocks instead of a single huge
contiguous reservation.  On Windows this avoids the ``[WinError 1450]`` "insufficient
system resources" error that occurs when a 17 GiB single-block ``mmap`` is
requested while the address space is fragmented.

Each allocation lives entirely inside one chunk, so a chunk is sized to fit the
largest single array it must hold.  Pages are committed by the OS only when an
array is actually written, just like a single anonymous ``mmap``.

Use as a context manager::

    with VoxelArena(reserve_bytes=4 * 1024**3) as arena:
        arr = arena.alloc(shape, np.float32, name='tmp')
        ...

The returned arrays are only valid until the arena is closed.
"""

import bisect
import mmap
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


class _Chunk:
    __slots__ = ("mmap", "size", "free")

    def __init__(self, mem: mmap.mmap, size: int):
        self.mmap = mem
        self.size = size
        self.free: List[Tuple[int, int]] = [(0, size)]

    def coalesce(self) -> None:
        if not self.free:
            return
        self.free.sort()
        merged = [self.free[0]]
        for off, size in self.free[1:]:
            prev_off, prev_size = merged[-1]
            if prev_off + prev_size == off:
                merged[-1] = (prev_off, prev_size + size)
            else:
                merged.append((off, size))
        self.free = merged


class VoxelArena:
    """Chunked arena with per-chunk first-fit free list for large NumPy arrays.

    The ``reserve_bytes`` argument is treated as a backward-compatible hint and
    a soft budget; the arena only creates ``mmap`` blocks large enough to satisfy
    each individual ``alloc`` request.  No single huge ``mmap`` of ``reserve_bytes``
    bytes is ever created.
    """

    def __init__(self, reserve_bytes: int):
        if reserve_bytes <= 0:
            raise ValueError("reserve_bytes must be positive")

        self._granularity = getattr(mmap, "ALLOCATIONGRANULARITY", mmap.PAGESIZE)
        self._reserve_hint = reserve_bytes
        # Start small; grow on demand.  A 1 GiB default chunk gives enough
        # head-room for a (3, nx, ny, nz) float32 array on most grids while
        # keeping each ``mmap`` request modest compared with a 6-18 GiB block.
        self._default_chunk_size = min(1 << 30, max(1 << 28, reserve_bytes // 8))
        self._default_chunk_size = self._round_up(self._default_chunk_size, self._granularity)

        self._chunks: List[_Chunk] = []
        self._used: Dict[str, Tuple[int, int, int]] = {}
        self._closed = False

    @staticmethod
    def _round_up(value: int, unit: int) -> int:
        return ((value + unit - 1) // unit) * unit

    @staticmethod
    def _align_offset(value: int, itemsize: int) -> int:
        # ndarray view alignment is only required for itemsize.
        if itemsize == 1:
            return value
        return ((value + itemsize - 1) // itemsize) * itemsize

    def _add_chunk(self, min_size: int) -> _Chunk:
        """Create a new mmap chunk that is at least min_size bytes."""
        sizes_to_try = [
            self._default_chunk_size,
            1 << 30,
            1 << 29,
            1 << 28,
        ]
        # Remove duplicates while preserving order.
        seen = set()
        candidates = []
        for s in sizes_to_try:
            if s not in seen:
                seen.add(s)
                candidates.append(s)

        # The chunk must be at least min_size; try standard sizes first, then
        # a tight chunk exactly sized to the allocation.
        tight_size = self._round_up(min_size, self._granularity)
        sizes = [max(s, tight_size) for s in candidates]
        if tight_size not in sizes:
            sizes.append(tight_size)

        for size in sizes:
            try:
                mem = mmap.mmap(-1, size, access=mmap.ACCESS_WRITE)
            except (OSError, OverflowError, MemoryError):
                continue
            chunk = _Chunk(mem, size)
            self._chunks.append(chunk)
            return chunk
        raise MemoryError(
            f"VoxelArena: could not allocate a {min_size / (1024 ** 3):.2f} GiB mmap chunk "
            f"(tried up to {sizes[0] / (1024 ** 3):.2f} GiB)."
        )

    def alloc(self, shape, dtype, name: Optional[str] = None) -> np.ndarray:
        """Return a NumPy array view carved from one of the arena chunks."""
        if self._closed:
            raise ValueError("VoxelArena is closed")

        dtype = np.dtype(dtype)
        count = int(np.prod(shape, dtype=np.int64))
        bytes_needed = count * dtype.itemsize
        if bytes_needed < 0:
            raise ValueError("invalid allocation size")
        if bytes_needed == 0:
            return np.empty(shape, dtype=dtype)

        aligned, chunk_idx, offset, region_size = self._find_or_grow(bytes_needed, dtype.itemsize)

        if name is not None:
            self._used[name] = (chunk_idx, offset, region_size)

        chunk = self._chunks[chunk_idx]
        arr = np.frombuffer(
            chunk.mmap,
            dtype=dtype,
            count=count,
            offset=offset,
        )
        arr.shape = shape
        return arr

    def _find_or_grow(
        self, bytes_needed: int, itemsize: int
    ) -> Tuple[int, int, int, int]:
        """Search existing chunks, or grow a new one, for bytes_needed bytes.

        Returns (aligned_bytes, chunk_index, chunk_offset, region_size).
        """
        best = None
        best_waste = None

        for chunk_idx, chunk in enumerate(self._chunks):
            for free_idx, (off, size) in enumerate(chunk.free):
                aligned_off = self._align_offset(off, itemsize)
                pad = aligned_off - off
                if pad + bytes_needed <= size:
                    waste = size - (pad + bytes_needed)
                    if best_waste is None or waste < best_waste:
                        best_waste = waste
                        best = (chunk_idx, free_idx, aligned_off, off, size, pad)

        if best is not None:
            chunk_idx, free_idx, aligned_off, off, size, pad = best
            chunk = self._chunks[chunk_idx]
            chunk.free.pop(free_idx)
            remaining = size - pad - bytes_needed
            if pad:
                chunk.free.append((off, pad))
            if remaining:
                chunk.free.append((aligned_off + bytes_needed, remaining))
            chunk.coalesce()
            return bytes_needed, chunk_idx, aligned_off, bytes_needed

        # No existing chunk has room.  Try a new chunk sized for this allocation,
        # with a bit of head-room so small follow-up arrays can share the chunk.
        headroom = max(self._default_chunk_size // 4, bytes_needed)
        min_size = bytes_needed + headroom
        chunk = self._add_chunk(min_size)

        off, size = chunk.free[0]
        aligned_off = self._align_offset(off, itemsize)
        pad = aligned_off - off
        if pad + bytes_needed > size:
            # The new chunk was sized with headroom, but the allocation plus
            # alignment padding still does not fit; this should be impossible.
            raise MemoryError(
                f"VoxelArena: fresh chunk ({size} bytes) cannot hold allocation "
                f"{bytes_needed} with alignment {itemsize}."
            )

        chunk.free.pop(0)
        remaining = size - pad - bytes_needed
        if pad:
            chunk.free.append((off, pad))
        if remaining:
            chunk.free.append((aligned_off + bytes_needed, remaining))
        chunk.coalesce()
        return bytes_needed, len(self._chunks) - 1, aligned_off, bytes_needed

    def free(self, name: Optional[str] = None) -> None:
        """Return a named slot to the free list of its chunk."""
        if name is None or name not in self._used:
            return
        chunk_idx, offset, size = self._used.pop(name)
        chunk = self._chunks[chunk_idx]
        bisect.insort(chunk.free, (offset, size))
        chunk.coalesce()

    def used_bytes(self) -> int:
        return sum(region_size for _, _, region_size in self._used.values())

    def free_bytes(self) -> int:
        return sum(sum(size for _, size in chunk.free) for chunk in self._chunks)

    def total_bytes(self) -> int:
        return sum(chunk.size for chunk in self._chunks)

    def reset(self) -> None:
        """Reset the arena, invalidating all outstanding views."""
        for chunk in self._chunks:
            try:
                chunk.mmap.close()
            except BufferError:
                pass
        self._chunks.clear()
        self._used.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self) -> None:
        if not self._closed:
            for chunk in self._chunks:
                try:
                    chunk.mmap.close()
                except BufferError:
                    pass
            self._chunks.clear()
            self._used.clear()
            self._closed = True
