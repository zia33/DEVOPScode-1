#include <iostream>
#include <array>
#include <queue>
#include <mutex>
#include <atomic>
#include <thread>
#include <string>
#include <memory>
#include <optional>
#include <condition_variable>
#include <expected>
#include <vector>
#include <concepts>
#include <memory_resource>
#include <coroutine> // For coroutines
#include <generator> // For std::generator (C++23)
#include <unordered_map>
#include <cstring>
#include <fstream>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <numa.h>

// If CUDA is not available, provide dummy functions
#ifndef CUDA_VERSION
#define __global__
inline void cudaFreeHost(void*) {}
inline void cudaMallocHost(void** ptr, size_t size) { *ptr = malloc(size); }
#endif

// ---------------- Concepts + Logger ----------------
template<typename T>
concept Printable = requires(T a) {
    { std::cout << a } -> std::same_as<std::ostream&>;
};

constexpr auto print_info = [](Printable auto val) {
    std::cout << "Info: " << val << "\n";
};

// ---------------- Policy-Based Logging ----------------
struct SilentPolicy {
    static void on_allocate(int) {}
    static void on_deallocate(int) {}
};

struct VerbosePolicy {
    static void on_allocate(int id) {
        std::cout << "[ALLOC] Object #" << id << "\n";
    }
    static void on_deallocate(int id) {
        std::cout << "[GC   ] Object #" << id << " collected\n";
    }
};

// ---------------- UltraAllocator with all features ----------------
class UltraAllocator {
public:
    UltraAllocator(size_t poolSize, size_t slabBlockSize = 64, size_t slabBlockCount = 128,
                   const std::string& backingFile = "",
                   bool usePinnedMemory = false, bool secureEnclave = false,
                   size_t alignment = 64, int numaNode = -1)
        : poolSize(poolSize), slabBlockSize(slabBlockSize), slabBlockCount(slabBlockCount),
          backingFile(backingFile), usePinnedMemory(usePinnedMemory), secureEnclave(secureEnclave),
          alignment(alignment), numaNode(numaNode) {
        allocatePool();
        
        // Initialize slab allocator region
        size_t slabSize = slabBlockSize * slabBlockCount;
        memoryPool.resize(poolSize);
        slabFreeList.resize(slabBlockCount);
        for (auto& flag : slabFreeList) flag.store(true);
        
        // Initialize main allocation block list
        blocks.push_back({slabSize, poolSize - slabSize, true, ""});
    }

    ~UltraAllocator() {
        if (usePinnedMemory && pool) {
            cudaFreeHost(pool);
        } else if (pool) {
            munmap(pool, poolSize);
        }
    }

    void* allocate(size_t size, const std::string& tag = "") {
        std::lock_guard<std::mutex> lock(allocatorMutex);

        if (size <= slabBlockSize) {
            return AllocateSlab(tag);
        }

        auto bestIt = blocks.end();
        for (auto it = blocks.begin(); it != blocks.end(); ++it) {
            if (it->free && it->size >= size) {
                if (bestIt == blocks.end() || it->size < bestIt->size) {
                    bestIt = it;
                }
            }
        }

        if (bestIt != blocks.end()) {
            size_t remaining = bestIt->size - size;
            bestIt->free = false;
            bestIt->size = size;
            bestIt->tag = tag;

            if (remaining > 0) {
                blocks.insert(std::next(bestIt), {bestIt->offset + size, remaining, true, ""});
            }

            void* ptr = OffsetToPointer(bestIt->offset);
            activeAllocations[ptr] = size;
            return ptr;
        }

        std::cerr << "Allocation failed: not enough memory\n";
        return nullptr;
    }

    void deallocate(void* ptr) {
        size_t offset = PointerToOffset(ptr);
        if (offset < slabBlockSize * slabBlockCount) {
            FreeSlab(ptr);
            return;
        }
        
        std::lock_guard<std::mutex> lock(allocatorMutex);
        for (size_t i = 0; i < blocks.size(); ++i) {
            if (blocks[i].offset == offset) {
                blocks[i].free = true;
                blocks[i].tag = "";

                if (i + 1 < blocks.size() && blocks[i + 1].free) {
                    blocks[i].size += blocks[i + 1].size;
                    blocks.erase(blocks.begin() + i + 1);
                }

                if (i > 0 && blocks[i - 1].free) {
                    blocks[i - 1].size += blocks[i].size;
                    blocks.erase(blocks.begin() + i);
                }
                activeAllocations.erase(ptr);
                return;
            }
        }

        std::cerr << "Deallocation failed: pointer not found\n";
    }

    void compactAndDefragment() {
        std::lock_guard<std::mutex> lock(allocatorMutex);

        std::vector<Block> newBlocks;
        size_t newOffset = slabBlockSize * slabBlockCount;
        std::vector<uint8_t> newPool(poolSize);
        std::unordered_map<void*, void*> relocationMap;

        for (const auto& block : blocks) {
            if (!block.free) {
                std::memcpy(&newPool[newOffset], &memoryPool[block.offset], block.size);
                void* oldPtr = OffsetToPointer(block.offset);
                void* newPtr = &newPool[newOffset];
                relocationMap[oldPtr] = newPtr;

                newBlocks.push_back({newOffset, block.size, false, block.tag});
                newOffset += block.size;
            }
        }
        if (newOffset < poolSize) {
            newBlocks.push_back({newOffset, poolSize - newOffset, true, ""});
        }
        
        memoryPool = std::move(newPool);
        blocks = std::move(newBlocks);
        
        std::unordered_map<void*, size_t> newActive;
        for (const auto& [oldPtr, size] : activeAllocations) {
            if (relocationMap.count(oldPtr)) {
                newActive[relocationMap[oldPtr]] = size;
            }
        }
        activeAllocations = std::move(newActive);
    }
    
    void persistToFile() {
        if (backingFile.empty()) return;
        std::ofstream out(backingFile, std::ios::binary);
        out.write(static_cast<char*>(pool), poolSize);
        out.close();
    }
    
    void protectReadOnly() {
        if (!usePinnedMemory && pool) {
            mprotect(pool, poolSize, PROT_READ);
        }
    }

    void protectReadWrite() {
        if (!usePinnedMemory && pool) {
            mprotect(pool, poolSize, PROT_READ | PROT_WRITE);
        }
    }
    
    void runGpuKernel() {
        if (!usePinnedMemory || !pool) {
            std::cerr << "GPU kernel skipped: not using pinned memory\n";
            return;
        }

        float* data = static_cast<float*>(pool);
        for (size_t i = 0; i < poolSize / sizeof(float); ++i) {
            data[i] = static_cast<float>(i);
        }
    }
    
    void printTelemetry() {
        std::lock_guard<std::mutex> lock(allocatorMutex);
        std::cout << "\n📊 Memory Telemetry:\n";
        std::cout << "Total Pool Size: " << poolSize << " bytes\n";
        std::cout << "Used: " << GetUsedMemory() << " bytes\n";
        std::cout << "Freed: " << GetFreedMemory() << " bytes\n";
        std::cout << "Active Allocations:\n";
        for (const auto& [ptr, size] : activeAllocations) {
            std::cout << "  Ptr: " << ptr << ", Size: " << size << "\n";
        }
    }
    
    void reportLeaks() {
        std::lock_guard<std::mutex> lock(allocatorMutex);
        if (activeAllocations.empty()) {
            std::cout << "\n✅ No memory leaks detected.\n";
        } else {
            std::cout << "\n🕵️ Leak Report:\n";
            for (const auto& [ptr, size] : activeAllocations) {
                std::cout << "  Leaked Ptr: " << ptr << ", Size: " << size << "\n";
            }
        }
    }

private:
    struct Block {
        size_t offset;
        size_t size;
        bool free;
        std::string tag;
    };

    std::vector<uint8_t> memoryPool;
    std::vector<Block> blocks;
    mutable std::mutex allocatorMutex;

    std::vector<std::atomic<bool>> slabFreeList;
    size_t slabBlockSize;
    size_t slabBlockCount;

    std::unordered_map<void*, size_t> activeAllocations;
    
    void* OffsetToPointer(size_t offset) {
        return static_cast<void*>(&memoryPool[offset]);
    }
    
    size_t PointerToOffset(void* ptr) {
        return static_cast<uint8_t*>(ptr) - memoryPool.data();
    }
    
    size_t GetUsedMemory() {
        size_t total = 0;
        for (const auto& [ptr, size] : activeAllocations) {
            total += size;
        }
        return total;
    }
    
    size_t GetFreedMemory() {
        size_t total = 0;
        for (const auto& block : blocks) {
            if (block.free) {
                total += block.size;
            }
        }
        return total;
    }
    
    void* AllocateSlab(const std::string& tag) {
        for (size_t i = 0; i < slabBlockCount; ++i) {
            bool expected = true;
            if (slabFreeList[i].compare_exchange_strong(expected, false)) {
                void* ptr = OffsetToPointer(i * slabBlockSize);
                activeAllocations[ptr] = slabBlockSize;
                return ptr;
            }
        }
        std::cerr << "Slab allocation failed: no free blocks\n";
        return nullptr;
    }

    void FreeSlab(void* ptr) {
        size_t offset = PointerToOffset(ptr);
        size_t index = offset / slabBlockSize;

        if (index < slabBlockCount) {
            slabFreeList[index].store(true);
            activeAllocations.erase(ptr);
        } else {
            std::cerr << "Slab free failed: invalid pointer\n";
        }
    }
    
    void allocatePool() {
        // Placeholder implementation for pool allocation
    }
};

// ---------------- ObjectPool Wrapper (Built on UltraAllocator) ----------------
template<typename T, size_t Size, typename Policy = SilentPolicy>
class ObjectPool {
    struct Entry {
        alignas(T) char data[sizeof(T)];
        std::atomic<int> ref_count{0};
        std::mutex mtx;
        bool used = false;
    };

    std::array<Entry, Size> pool;
    std::queue<size_t> gc_queue;
    std::mutex gc_mtx;
    std::condition_variable cv;
    std::atomic<bool> stop_gc{false};

    struct gc_awaitable {
        ObjectPool* pool;
        bool await_ready() const { return !pool->gc_queue.empty() || pool->stop_gc; }
        void await_suspend(std::coroutine_handle<> handle) {
            std::unique_lock lock(pool->gc_mtx);
            pool->cv.wait(lock, [&] { return !pool->gc_queue.empty() || pool->stop_gc; });
            handle.resume();
        }
        void await_resume() {}
    };

    std::jthread gc_thread;
    std::coroutine_handle<> gc_handle;

    void start_gc_coroutine() {
        auto gc_task = [this]() -> std::generator<void> {
            while (!stop_gc) {
                co_await gc_awaitable{this};
                while (!gc_queue.empty()) {
                    std::lock_guard lock(gc_mtx);
                    if (gc_queue.empty()) continue;
                    size_t idx = gc_queue.front();
                    gc_queue.pop();
                    auto& e = pool[idx];
                    std::lock_guard lg(e.mtx);
                    reinterpret_cast<T*>(e.data)->~T();
                    e.used = false;
                    Policy::on_deallocate(idx);
                }
            }
        };
        gc_thread = std::jthread([task = gc_task()] mutable { for (auto _ : task) {} });
    }

public:
    ObjectPool() { start_gc_coroutine(); }
    ~ObjectPool() { stop_gc = true; cv.notify_all(); }

    template<typename... Args>
    std::expected<T*, std::string> allocate(Args&&... args) {
        for (size_t i = 0; i < Size; ++i) {
            std::lock_guard lg(pool[i].mtx);
            if (!pool[i].used) {
                try {
                    pool[i].used = true;
                    new(pool[i].data) T(std::forward<Args>(args)...);
                    pool[i].ref_count.store(1);
                    Policy::on_allocate(i);
                    return reinterpret_cast<T*>(pool[i].data);
                } catch (...) {
                    pool[i].used = false;
                    return std::unexpected("Constructor failed.");
                }
            }
        }
        return std::unexpected("Object pool exhausted.");
    }

    void add_ref(T* ptr) {
        if (auto* e = get_entry(ptr)) e->ref_count.fetch_add(1);
    }
    void release(T* ptr) {
        if (auto* e = get_entry(ptr)) {
            if (e->ref_count.fetch_sub(1) == 1) {
                std::lock_guard lock(gc_mtx);
                gc_queue.push(e - &pool[0]);
                cv.notify_one();
            }
        }
    }
    std::generator<T&> used_objects() {
        for (auto& entry : pool) {
            if (entry.used) { co_yield *reinterpret_cast<T*>(entry.data); }
        }
    }
    size_t used_count() const {
        size_t count = 0;
        for (const auto& e : pool) { if (e.used) ++count; }
        return count;
    }
private:
    Entry* get_entry(T* ptr) {
        for (auto& e : pool) if (reinterpret_cast<T*>(e.data) == ptr) return &e;
        return nullptr;
    }
};

// ---------------- Smart Ref Wrapper ----------------
template<typename T, typename Pool>
class Ref {
    T* ptr = nullptr;
    Pool* pool = nullptr;
public:
    Ref() = default;
    Ref(T* p, Pool* pl) : ptr(p), pool(pl) {}
    Ref(const Ref& other) : ptr(other.ptr), pool(other.pool) { if (ptr) pool->add_ref(ptr); }
    Ref& operator=(const Ref& other) {
        if (this != &other) {
            if (ptr) pool->release(ptr);
            ptr = other.ptr;
            pool = other.pool;
            if (ptr) pool->add_ref(ptr);
        }
        return *this;
    }
    ~Ref() { if (ptr) pool->release(ptr); }
    T* operator->() { return ptr; }
    const T* operator->() const { return ptr; }
    T& operator*() { return *ptr; }
    const T& operator*() const { return *ptr; }
    explicit operator bool() const { return ptr != nullptr; }
};

// ---------------- PMR-Enabled Test Type ----------------
struct Widget {
    int id;
    std::pmr::vector<int> data;
    Widget(int id, std::pmr::memory_resource* res = std::pmr::get_default_resource())
        : id(id), data(res) {
        data.resize(5, id);
        print_info("Widget created: " + std::to_string(id));
    }
    ~Widget() {
        print_info("Widget destroyed: " + std::to_string(id));
    }
};

// ---------------- Main Program ----------------
int main() {
    std::cout << "--- Demonstrating ObjectPool with Coroutine GC ---\n";
    ObjectPool<Widget, 8, VerbosePolicy> pool;
    std::pmr::monotonic_buffer_resource pmr_res;
    std::vector<std::jthread> threads;
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([&, i] {
            auto result = pool.allocate(i, &pmr_res);
            if (!result) {
                print_info("Pool full, fallback to shared_ptr.");
                auto sp = std::allocate_shared<Widget>(std::pmr::polymorphic_allocator<Widget>(&pmr_res), i, &pmr_res);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                print_info("Using shared_ptr Widget: " + std::to_string(sp->id));
                return;
            }
            Ref<Widget, decltype(pool)> w(result.value(), &pool);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (w) {
                print_info("Using Widget: " + std::to_string(w->id));
            }
        });
    }

    threads.clear();
    print_info("--- Currently live widgets: ---");
    for (auto& widget : pool.used_objects()) {
        print_info("Live Widget ID: " + std::to_string(widget.id));
    }

    print_info("Pool usage: " + std::to_string(pool.used_count()) + " / 8");

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    print_info("ObjectPool demo complete.\n");

    std::cout << "--- Demonstrating UltraAllocator ---\n";
    UltraAllocator allocator(8192);
    void* a = allocator.allocate(512, "MeshData");
    void* b = allocator.allocate(1024, "TextureCache");

    allocator.deallocate(a);
    allocator.printTelemetry();

    allocator.reportLeaks();
    allocator.deallocate(b);
    allocator.reportLeaks();
    
    print_info("Program complete.");
}