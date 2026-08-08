#pragma once

#include <cstddef>
#include <cstdint>
#include <memory_resource>
#include <stdexcept>

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

} // namespace josecast
