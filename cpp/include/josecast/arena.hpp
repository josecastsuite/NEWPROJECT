#pragma once

#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <stdexcept>
#include <vector>

namespace josecast {

// A virtual-address arena used as a std::pmr::memory_resource.
//
// On construction a single contiguous block of virtual address space is
// reserved from the operating system (VirtualAlloc MEM_RESERVE on Windows,
// mmap PROT_NONE on Linux/macOS).  Pages are committed only when
// do_allocate() is called, and physical RAM is allocated only when those
// pages are actually written (commit-on-touch).
//
// This is the C++ counterpart of core/voxel_arena.py: it guarantees that
// many large per-voxel arrays live in one contiguous virtual block instead of
// forcing the OS/heap to find a fresh 500+ MiB contiguous block for every
// std::vector.
class VirtualArena final : public std::pmr::memory_resource {
public:
    explicit VirtualArena(std::size_t reserve_bytes);
    VirtualArena() = delete;
    ~VirtualArena() override;
    VirtualArena(const VirtualArena&) = delete;
    VirtualArena& operator=(const VirtualArena&) = delete;

    // Recommended virtual reserve for n voxels assuming an expected per-voxel
    // working-set footprint.  Reserves 3x the expected committed size so that
    // temporary spikes and future growth have room without asking the OS for
    // a new arena.
    static std::size_t recommended(std::size_t n, std::size_t bytes_per_voxel);

    std::size_t used() const noexcept { return offset_; }
    std::size_t total() const noexcept { return reserve_bytes_; }

protected:
    void* do_allocate(std::size_t bytes, std::size_t alignment) override;
    void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override;
    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override;

private:
    void* base_;
    std::size_t reserve_bytes_;
    std::size_t offset_;
    std::size_t page_size_;
    bool ok_;

    void commit_range(std::size_t start, std::size_t end);
    void release();
    static std::size_t get_page_size();
    static std::size_t round_up(std::size_t value, std::size_t unit);
};

// Chunked virtual-address arena used as a std::pmr::memory_resource.
//
// Instead of one giant contiguous reservation, this arena keeps a list of
// 1-2 GiB virtual blocks.  Each do_allocate() request is satisfied inside a
// single chunk, so no single std::pmr::vector ever has to reserve a 9+ GiB
// contiguous virtual address block.  Chunks are grown on demand, pages are
// committed only when written, and individual deallocations are no-ops.
class ChunkedArena final : public std::pmr::memory_resource {
public:
    explicit ChunkedArena(std::size_t reserve_hint = 0);
    ChunkedArena() = delete;
    ~ChunkedArena() override;
    ChunkedArena(const ChunkedArena&) = delete;
    ChunkedArena& operator=(const ChunkedArena&) = delete;

    // Same semantics as VirtualArena::recommended: returns an expected
    // committed working-set size.  ChunkedArena uses it only as a hint for
    // the first pre-reserved chunks; it will grow if more is needed.
    static std::size_t recommended(std::size_t n, std::size_t bytes_per_voxel);

    std::size_t used() const noexcept;
    std::size_t total() const noexcept;

protected:
    void* do_allocate(std::size_t bytes, std::size_t alignment) override;
    void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override;
    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override;

private:
    struct Chunk {
        void* base;
        std::size_t size;
        std::size_t used;
    };

    std::vector<Chunk> chunks_;
    std::size_t page_size_;
    std::size_t min_chunk_size_;
    std::size_t used_;
    std::size_t total_;
    bool ok_;

    bool add_chunk(std::size_t min_size);
    void* reserve_raw(std::size_t size);
    void release_raw(void* base, std::size_t size);
    void commit_range(void* base, std::size_t start, std::size_t end);
    static std::size_t get_page_size();
    static std::size_t round_up(std::size_t value, std::size_t unit);
};

} // namespace josecast
