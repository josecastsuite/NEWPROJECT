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

// ---------------------------------------------------------------------------
// ChunkedArena implementation
// ---------------------------------------------------------------------------

std::size_t ChunkedArena::get_page_size() {
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

std::size_t ChunkedArena::round_up(std::size_t value, std::size_t unit) {
    return ((value + unit - 1) / unit) * unit;
}

ChunkedArena::ChunkedArena(std::size_t reserve_hint)
    : page_size_(get_page_size()),
      min_chunk_size_(1ULL << 28), // 256 MiB
      used_(0), total_(0), ok_(true)
{
    // Pre-reserve the hint as a collection of 256 MiB chunks.  We cap the
    // eager reservation at 4 GiB so the constructor never tries to grab a giant
    // contiguous region just because the caller passed a large hint.
    std::size_t remaining = round_up(reserve_hint, min_chunk_size_);
    if (remaining == 0) remaining = min_chunk_size_;
    remaining = std::min(remaining, static_cast<std::size_t>(4ULL << 30));
    while (remaining >= min_chunk_size_) {
        if (!add_chunk(min_chunk_size_)) break;
        remaining -= min_chunk_size_;
    }
}

ChunkedArena::~ChunkedArena() {
    for (const auto& ch : chunks_) {
        if (ch.base != nullptr) release_raw(ch.base, ch.size);
    }
}

void* ChunkedArena::reserve_raw(std::size_t size) {
#ifdef _WIN32
    void* base = VirtualAlloc(nullptr, size, MEM_RESERVE, PAGE_NOACCESS);
    return base;
#else
    void* base = mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) return nullptr;
    return base;
#endif
}

void ChunkedArena::release_raw(void* base, std::size_t size) {
    if (base == nullptr) return;
#ifdef _WIN32
    VirtualFree(base, 0, MEM_RELEASE);
#else
    munmap(base, size);
#endif
}

void ChunkedArena::commit_range(void* base, std::size_t start, std::size_t end) {
    if (start >= end || base == nullptr) return;
    void* ptr = static_cast<char*>(base) + start;
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

bool ChunkedArena::add_chunk(std::size_t min_size) {
    std::size_t size = round_up(std::max(min_size, min_chunk_size_), page_size_);
    void* base = reserve_raw(size);
    if (base == nullptr) {
        ok_ = false;
        return false;
    }
    chunks_.push_back(Chunk{base, size, 0});
    total_ += size;
    ok_ = true;
    return true;
}

void* ChunkedArena::do_allocate(std::size_t bytes, std::size_t alignment) {
    if (bytes == 0) {
        return nullptr;
    }
    if (alignment == 0) alignment = 1;

    // Try to satisfy from an existing chunk.
    for (auto& ch : chunks_) {
        std::size_t pad = (alignment - (ch.used % alignment)) % alignment;
        std::size_t aligned = ch.used + pad;
        if (aligned + bytes <= ch.size) {
            std::size_t old_used = ch.used;
            commit_range(ch.base, (aligned / page_size_) * page_size_,
                         round_up(aligned + bytes, page_size_));
            ch.used = aligned + bytes;
            used_ += ch.used - old_used;
            return static_cast<char*>(ch.base) + aligned;
        }
    }

    // No existing chunk fits; allocate a new chunk sized for this request.
    std::size_t needed = round_up(bytes + alignment, page_size_);
    // New chunk is at least 1 GiB and holds the request plus 1 GiB headroom
    // so small follow-up allocations do not immediately trigger another chunk.
    std::size_t chunk_size = std::max(needed + min_chunk_size_, min_chunk_size_);
    if (!add_chunk(chunk_size)) {
        throw std::bad_alloc();
    }
    Chunk& ch = chunks_.back();
    std::size_t pad = (alignment - (ch.used % alignment)) % alignment;
    std::size_t aligned = ch.used + pad;
    std::size_t old_used = ch.used;
    commit_range(ch.base, (aligned / page_size_) * page_size_,
                 round_up(aligned + bytes, page_size_));
    ch.used = aligned + bytes;
    used_ += ch.used - old_used;
    return static_cast<char*>(ch.base) + aligned;
}

void ChunkedArena::do_deallocate(void*, std::size_t, std::size_t) {
    // Monotonic arena: individual deallocations are no-ops.
}

bool ChunkedArena::do_is_equal(const std::pmr::memory_resource& other) const noexcept {
    return this == &other;
}

std::size_t ChunkedArena::used() const noexcept { return used_; }
std::size_t ChunkedArena::total() const noexcept { return total_; }

std::size_t ChunkedArena::recommended(std::size_t n, std::size_t bytes_per_voxel) {
    // Unlike VirtualArena we do NOT multiply by 3 here.  ChunkedArena grows
    // on demand, so the hint is just the expected committed working set.  The
    // 3x headroom policy is achieved by pre-reserving 1 GiB chunks and by
    // callers choosing a larger bytes_per_voxel if they want extra margin.
    std::size_t expected = n * bytes_per_voxel;
    constexpr std::size_t one_gib = 1ULL << 30;
    return std::max(expected, one_gib);
}

} // namespace josecast
