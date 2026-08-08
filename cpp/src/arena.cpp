#include "josecast/arena.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <new>
#include <stdexcept>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace josecast {

std::size_t VirtualArena::get_page_size() {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return static_cast<std::size_t>(si.dwPageSize);
#else
    long ps = sysconf(_SC_PAGESIZE);
    if (ps <= 0) return 4096;
    return static_cast<std::size_t>(ps);
#endif
}

std::size_t VirtualArena::round_up(std::size_t value, std::size_t unit) {
    return ((value + unit - 1) / unit) * unit;
}

VirtualArena::VirtualArena(std::size_t reserve_bytes)
    : base_(nullptr), reserve_bytes_(0), offset_(0), page_size_(get_page_size()), ok_(false)
{
    reserve_bytes_ = round_up(reserve_bytes, page_size_);
    if (reserve_bytes_ == 0) {
        throw std::bad_alloc();
    }

#ifdef _WIN32
    base_ = VirtualAlloc(nullptr, reserve_bytes_, MEM_RESERVE, PAGE_NOACCESS);
    if (base_ == nullptr) {
        throw std::bad_alloc();
    }
#else
    base_ = mmap(nullptr, reserve_bytes_, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base_ == MAP_FAILED) {
        base_ = nullptr;
        throw std::bad_alloc();
    }
#endif
    ok_ = true;
    offset_ = 0;
}

VirtualArena::~VirtualArena() {
    release();
}

void VirtualArena::release() {
    if (!ok_ || base_ == nullptr) return;
#ifdef _WIN32
    VirtualFree(base_, 0, MEM_RELEASE);
#else
    munmap(base_, reserve_bytes_);
#endif
    base_ = nullptr;
    ok_ = false;
}

void VirtualArena::commit_range(std::size_t start, std::size_t end) {
    if (start >= end) return;
    if (end > reserve_bytes_) {
        throw std::bad_alloc();
    }
    void* ptr = static_cast<char*>(base_) + start;
    std::size_t size = end - start;
#ifdef _WIN32
    if (VirtualAlloc(ptr, size, MEM_COMMIT, PAGE_READWRITE) == nullptr) {
        throw std::bad_alloc();
    }
#else
    if (mprotect(ptr, size, PROT_READ | PROT_WRITE) != 0) {
        throw std::bad_alloc();
    }
#endif
}

void* VirtualArena::do_allocate(std::size_t bytes, std::size_t alignment) {
    if (bytes == 0) {
        return base_;
    }
    // alignment is a power of two for standard types.
    std::size_t pad = (alignment - (offset_ % alignment)) % alignment;
    std::size_t aligned = offset_ + pad;
    if (aligned + bytes > reserve_bytes_ || aligned + bytes < aligned) {
        throw std::bad_alloc();
    }

    // Commit the whole page range touched by [aligned, aligned+bytes).
    std::size_t page_start = (aligned / page_size_) * page_size_;
    std::size_t page_end = round_up(aligned + bytes, page_size_);
    commit_range(page_start, page_end);

    offset_ = aligned + bytes;
    return static_cast<char*>(base_) + aligned;
}

void VirtualArena::do_deallocate(void*, std::size_t, std::size_t) {
    // Monotonic arena: individual deallocations are no-ops.  The whole arena
    // is released when the object is destroyed.
}

bool VirtualArena::do_is_equal(const std::pmr::memory_resource& other) const noexcept {
    return this == &other;
}

std::size_t VirtualArena::recommended(std::size_t n, std::size_t bytes_per_voxel) {
    // 3x virtual reserve of the expected committed working set, with a
    // minimum of 1 GiB so small grids do not fall back to tiny arenas.
    std::size_t expected = n * bytes_per_voxel;
    std::size_t triple = expected * 3;
    constexpr std::size_t one_gib = 1ULL << 30;
    std::size_t result = std::max(triple, one_gib);
    return result;
}

} // namespace josecast
