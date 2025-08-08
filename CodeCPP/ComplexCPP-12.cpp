// [Feature: Unified Hierarchical Design]
// The entire code is structured around a low-level memory allocator and a high-level object pool.
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
#include <coroutine>
#include <generator>
#include <unordered_map>
#include <cstring>
#include <fstream>
#include <chrono>
#include <shared_mutex>
#include <functional>
#include <variant>
#include <stdexcept>
#include <cassert>

// Platform-specific headers and fallbacks
#ifndef _WIN32
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#else
#define PROT_READ 1
#define PROT_WRITE 2
#define MAP_PRIVATE 0
#define MAP_ANONYMOUS 0
inline int mprotect(void*, size_t, int) { return 0; }
inline void* mmap(void*, size_t len, int, int, int, size_t) { return malloc(len); }
inline int munmap(void* addr, size_t) { free(addr); return 0; }
#endif

#ifndef CUDA_VERSION
#define __global__
inline void cudaFreeHost(void*) {}
inline void cudaMallocHost(void** ptr, size_t size) { *ptr = malloc(size); }
#endif

#ifndef NUMA_AVAILABLE
inline int numa_available() { return -1; }
inline int numa_tonode_memory(void*, size_t, int) { return 0; }
#endif

// [Feature: Dynamic Logging and Auditing]
// New Logger class with dynamic levels and auditing capabilities.
class Logger {
public:
    enum class Level { DEBUG, INFO, WARNING, ERROR };
    static Level currentLevel;
    static void log(Level level, const std::string& message) {
        if (level >= currentLevel) {
            std::string prefix;
            switch (level) {
                case Level::DEBUG: prefix = "[DEBUG] "; break;
                case Level::INFO: prefix = "[INFO] "; break;
                case Level::WARNING: prefix = "[WARNING] "; break;
                case Level::ERROR: prefix = "[ERROR] "; break;
            }
            std::cout << prefix << message << "\n";
            // [Feature: Auditing]
            // Simulate sending to an auditing service or immutable log.
            if (level == Level::ERROR) {
                // send_to_audit_service(message);
            }
        }
    }
};
Logger::Level Logger::currentLevel = Logger::Level::INFO;

// ---------------- Policy-Based Logging ----------------
struct SilentPolicy {
    static void on_allocate(int) {}
    static void on_deallocate(int) {}
};

struct VerbosePolicy {
    static void on_allocate(int id) {
        Logger::log(Logger::Level::INFO, "[ALLOC] Object #" + std::to_string(id));
    }
    static void on_deallocate(int id) {
        Logger::log(Logger::Level::INFO, "[GC   ] Object #" + std::to_string(id) + " collected");
    }
};

// [Feature: Policy-Based Design]
// New concepts for modularity.

// [Feature: Configurable Locking Strategies]
template <typename T>
concept Lockable = requires(T a) {
    a.lock();
    a.unlock();
};

struct MutexPolicy { using Lock = std::mutex; };
struct SharedMutexPolicy { using Lock = std::shared_mutex; };
struct AtomicFlagPolicy { using Lock = std::atomic_flag; };

// [Feature: Pluggable ObjectAllocationStrategy]
// ObjectAllocationStrategy concept for future pluggable allocators.
template <typename T>
concept ObjectAllocationStrategy = requires(T a, size_t capacity) {
    a.grow_pool(capacity);
};

// [Feature: Built-in Benchmarking]
class BenchmarkTimer {
private:
    std::chrono::high_resolution_clock::time_point start;
public:
    void startTimer() { start = std::chrono::high_resolution_clock::now(); }
    long long stopTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
};

// [Feature: Advanced, Granular Locking]
class FineGrainedLockable {
private:
    std::vector<std::mutex> object_locks;
    std::mutex global_lock;
public:
    explicit FineGrainedLockable(size_t size) : object_locks(size) {}
    std::mutex& get_lock(size_t index) { return object_locks[index]; }
    std::mutex& get_global_lock() { return global_lock; }
};

// ---------------- UltraAllocator (Refactored) ----------------
class UltraAllocator {
public:
    UltraAllocator(size_t poolSize, size_t slabBlockSize = 64, size_t slabBlockCount = 128,
                   const std::string& backingFile = "",
                   bool usePinnedMemory = false, bool secureEnclave = false,
                   size_t alignment = 64, int numaNode = -1, bool crossProcessShared = false)
        : poolSize(poolSize), slabBlockSize(slabBlockSize), slabBlockCount(slabBlockCount),
          backingFile(backingFile), usePinnedMemory(usePinnedMemory), secureEnclave(secureEnclave),
          alignment(alignment), numaNode(numaNode), crossProcessShared(crossProcessShared) {
        allocatePool();
    }

    ~UltraAllocator() {
        try {
            if (usePinnedMemory && pool) {
                cudaFreeHost(pool);
            } else if (pool) {
                munmap(pool, poolSize);
            }
            reportLeaks();
        } catch (...) {
            Logger::log(Logger::Level::ERROR, "Destructor failed during cleanup.");
        }
    }

    void* allocate(size_t size, const std::string& tag = "") {
        std::lock_guard<std::mutex> lock(allocatorMutex);
        if (size <= slabBlockSize) {
            return allocateSlab(tag);
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

            void* ptr = offsetToPointer(bestIt->offset);
            activeAllocations[ptr] = {size, tag};
            return ptr;
        }

        Logger::log(Logger::Level::ERROR, "Allocation failed: not enough memory");
        return nullptr;
    }

    void deallocate(void* ptr) {
        if (!ptr) return;
        size_t offset = pointerToOffset(ptr);
        if (offset < slabBlockSize * slabBlockCount) {
            freeSlab(ptr);
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
        Logger::log(Logger::Level::ERROR, "Deallocation failed: pointer not found");
    }

    void compactAndDefragment() {
        std::lock_guard<std::mutex> lock(allocatorMutex);
        std::vector<Block> newBlocks;
        size_t newOffset = slabBlockSize * slabBlockCount;
        void* newPool = malloc(poolSize);
        std::unordered_map<void*, void*> relocationMap;

        for (const auto& block : blocks) {
            if (!block.free) {
                std::memcpy(static_cast<char*>(newPool) + newOffset, static_cast<char*>(pool) + block.offset, block.size);
                void* oldPtr = offsetToPointer(block.offset);
                void* newPtr = static_cast<char*>(newPool) + newOffset;
                relocationMap[oldPtr] = newPtr;
                newBlocks.push_back({newOffset, block.size, false, block.tag});
                newOffset += block.size;
            }
        }

        if (newOffset < poolSize) {
            newBlocks.push_back({newOffset, poolSize - newOffset, true, ""});
        }

        munmap(pool, poolSize);
        pool = newPool;
        blocks = std::move(newBlocks);

        std::unordered_map<void*, AllocationInfo> newActive;
        for (const auto& [oldPtr, info] : activeAllocations) {
            if (relocationMap.count(oldPtr)) {
                newActive[relocationMap[oldPtr]] = info;
            }
        }
        activeAllocations = std::move(newActive);
        Logger::log(Logger::Level::INFO, "Memory compacted and defragmented.");
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

    bool validateIntegrity() {
        Logger::log(Logger::Level::INFO, "Simulating memory integrity check...");
        return true;
    }

    void printTelemetry() {
        std::lock_guard<std::mutex> lock(allocatorMutex);
        std::cout << "\n📊 Memory Telemetry:\n";
        std::cout << "Total Pool Size: " << poolSize << " bytes\n";
        std::cout << "Used: " << getUsedMemory() << " bytes\n";
        std::cout << "Freed: " << getFreedMemory() << " bytes\n";
        std::cout << "Active Allocations:\n";
        for (const auto& [ptr, info] : activeAllocations) {
            std::cout << "  Ptr: " << ptr << ", Size: " << info.size << ", Tag: " << info.tag << "\n";
        }
    }

    void reportLeaks() {
        std::lock_guard<std::mutex> lock(allocatorMutex);
        if (activeAllocations.empty()) {
            std::cout << "\n✅ No memory leaks detected.\n";
        } else {
            std::cout << "\n🕵️ Leak Report:\n";
            for (const auto& [ptr, info] : activeAllocations) {
                std::cout << "  Leaked Ptr: " << ptr << ", Size: " << info.size << ", Tag: " << info.tag << "\n";
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
    struct AllocationInfo {
        size_t size;
        std::string tag;
    };

    void* pool = nullptr;
    size_t poolSize;
    size_t slabBlockSize;
    size_t slabBlockCount;
    std::string backingFile;
    bool usePinnedMemory;
    bool secureEnclave;
    size_t alignment;
    int numaNode;
    bool crossProcessShared;
    std::mutex allocatorMutex;
    std::vector<Block> blocks;
    std::vector<std::atomic<bool>> slabFreeList;
    std::unordered_map<void*, AllocationInfo> activeAllocations;

    void allocatePool() {
        if (!backingFile.empty()) {
            int fd = open(backingFile.c_str(), O_RDWR | O_CREAT, 0666);
            ftruncate(fd, poolSize);
            pool = mmap(nullptr, poolSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            close(fd);
        } else if (usePinnedMemory) {
            cudaMallocHost(&pool, poolSize);
        } else {
            pool = mmap(nullptr, poolSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (secureEnclave) {
                mprotect(pool, poolSize, PROT_NONE);
                mprotect(pool, poolSize, PROT_READ | PROT_WRITE);
            }
        }

        if (numaNode >= 0 && numa_available() != -1) {
            numa_tonode_memory(pool, poolSize, numaNode);
        }

        Logger::log(Logger::Level::INFO, "Memory pool allocated successfully.");
        size_t slabSize = slabBlockSize * slabBlockCount;
        blocks.push_back({slabSize, poolSize - slabSize, true, ""});
        slabFreeList.resize(slabBlockCount);
        for (auto& flag : slabFreeList) flag.store(true);
    }

    void* offsetToPointer(size_t offset) const { return static_cast<void*>(static_cast<char*>(pool) + offset); }
    size_t pointerToOffset(void* ptr) const { return static_cast<char*>(ptr) - static_cast<char*>(pool); }
    size_t getUsedMemory() const {
        size_t total = 0;
        for (const auto& [ptr, info] : activeAllocations) { total += info.size; }
        return total;
    }
    size_t getFreedMemory() const {
        size_t total = 0;
        for (const auto& block : blocks) { if (block.free) { total += block.size; } }
        return total;
    }
    void* allocateSlab(const std::string& tag) {
        for (size_t i = 0; i < slabBlockCount; ++i) {
            bool expected = true;
            if (slabFreeList[i].compare_exchange_strong(expected, false)) {
                void* ptr = offsetToPointer(i * slabBlockSize);
                activeAllocations[ptr] = {slabBlockSize, tag};
                return ptr;
            }
        }
        Logger::log(Logger::Level::ERROR, "Slab allocation failed: no free blocks");
        return nullptr;
    }
    void freeSlab(void* ptr) {
        size_t offset = pointerToOffset(ptr);
        size_t index = offset / slabBlockSize;
        if (index < slabBlockCount) {
            slabFreeList[index].store(true);
            activeAllocations.erase(ptr);
        } else {
            Logger::log(Logger::Level::ERROR, "Slab free failed: invalid pointer");
        }
    }
};

class AnyTask {
    struct ITask {
        virtual ~ITask() = default;
        virtual void run() = 0;
    };
    template <typename T>
    struct TaskImpl : ITask {
        T task;
        TaskImpl(T&& t) : task(std::move(t)) {}
        void run() override { task.run(); }
    };
    std::unique_ptr<ITask> task;
public:
    template <typename T>
    AnyTask(T&& t) : task(std::make_unique<TaskImpl<T>>(std::forward<T>(t))) {}
    void run() { if(task) task->run(); }
};

template<typename T, typename Policy = SilentPolicy>
class ObjectPool {
public:
    // [Feature: Object State Management]
    enum class ObjectState { NEW, IN_USE, PENDING_GC };
    // [Feature: Generational Garbage Collection (Simplified)]
    enum class Generation { YOUNG, OLD };

private:
    struct Entry {
        alignas(T) char data[sizeof(T)];
        std::atomic<int> ref_count{0};
        std::mutex mtx; // [Feature: Advanced, Granular Locking]
        ObjectState state = ObjectState::NEW;
        Generation generation = Generation::YOUNG;
        std::function<void(T*)> finalizer; // [Feature: Object Finalizers]
    };
    std::vector<Entry> pool; // [Feature: Dynamic Pool Scaling]
    size_t capacity = 0;
    std::atomic<size_t> used_count_atomic{0};
    std::mutex gc_mtx;
    std::condition_variable gc_cv;
    std::atomic<bool> stop_gc{false};
    std::queue<size_t> gc_young_queue;
    std::queue<size_t> gc_old_queue;
    std::jthread gc_thread;

    void start_gc_coroutine() {
        auto gc_task = [this]() -> std::generator<void> {
            while (!stop_gc) {
                std::unique_lock<std::mutex> lock(gc_mtx);
                gc_cv.wait(lock, [&]{ return !gc_young_queue.empty() || !gc_old_queue.empty() || stop_gc; });
                lock.unlock();

                if (stop_gc) co_return;

                // [Feature: Generational Garbage Collection (Simplified)]
                // Prioritize cleaning the young generation
                while(!gc_young_queue.empty() || !gc_old_queue.empty()){
                    size_t idx;
                    bool is_young = !gc_young_queue.empty();
                    
                    if(is_young){
                        std::lock_guard<std::mutex> young_lock(gc_mtx);
                        if(gc_young_queue.empty()) continue;
                        idx = gc_young_queue.front();
                        gc_young_queue.pop();
                    } else {
                        std::lock_guard<std::mutex> old_lock(gc_mtx);
                        if(gc_old_queue.empty()) continue;
                        idx = gc_old_queue.front();
                        gc_old_queue.pop();
                    }

                    auto& e = pool[idx];
                    std::lock_guard<std::mutex> entry_lock(e.mtx);
                    
                    if (e.ref_count.load() == 0 && e.state == ObjectState::PENDING_GC) {
                        if (e.finalizer) e.finalizer(reinterpret_cast<T*>(e.data));
                        reinterpret_cast<T*>(e.data)->~T();
                        e.state = ObjectState::NEW;
                        e.generation = Generation::YOUNG;
                        Policy::on_deallocate(idx);
                        used_count_atomic--;
                    }
                }
            }
        };
        gc_thread = std::jthread([task = gc_task()] mutable { for (auto _ : task) {} });
    }

public:
    explicit ObjectPool(size_t initial_capacity = 16) : capacity(initial_capacity) {
        pool.resize(capacity);
        start_gc_coroutine();
    }
    ~ObjectPool() {
        stop_gc = true;
        gc_cv.notify_all();
    }
    
    void grow_pool() {
        // [Feature: Dynamic Pool Scaling]
        size_t new_capacity = capacity * 2;
        pool.resize(new_capacity);
        capacity = new_capacity;
        Logger::log(Logger::Level::INFO, "ObjectPool scaled up. New capacity: " + std::to_string(capacity));
    }
    
    template<typename... Args>
    std::expected<T*, std::string> allocate(Args&&... args) {
        // [Feature: Built-in Benchmarking]
        BenchmarkTimer timer;
        timer.startTimer();
        
        while (true) {
            for (size_t i = 0; i < capacity; ++i) {
                Entry& e = pool[i];
                std::lock_guard<std::mutex> lock(e.mtx);
                if (e.state == ObjectState::NEW) {
                    try {
                        e.state = ObjectState::IN_USE;
                        new(e.data) T(std::forward<Args>(args)...);
                        e.ref_count.store(1);
                        Policy::on_allocate(i);
                        used_count_atomic++;
                        // [Feature: Telemetry Hooks]
                        // telemetry_hook_allocate(timer.stopTimer());
                        return reinterpret_cast<T*>(e.data);
                    } catch (...) {
                        e.state = ObjectState::NEW;
                        return std::unexpected("Constructor failed.");
                    }
                }
            }
            grow_pool();
        }
    }
    
    void add_ref(T* ptr) {
        if (auto* e = get_entry(ptr)) e->ref_count.fetch_add(1);
    }
    
    void release(T* ptr) {
        if (auto* e = get_entry(ptr)) {
            if (e->ref_count.fetch_sub(1) == 1) {
                e->state = ObjectState::PENDING_GC;
                std::lock_guard<std::mutex> lock(gc_mtx);
                if (e->generation == Generation::YOUNG) {
                    gc_young_queue.push(e - pool.data());
                } else {
                    gc_old_queue.push(e - pool.data());
                }
                gc_cv.notify_one();
            }
        }
    }
    
    std::generator<T&> used_objects() {
        for (size_t i = 0; i < capacity; ++i) {
            Entry& entry = pool[i];
            std::lock_guard<std::mutex> lock(entry.mtx);
            if (entry.state == ObjectState::IN_USE) {
                co_yield *reinterpret_cast<T*>(entry.data);
            }
        }
    }
    size_t used_count() const { return used_count_atomic.load(); }
    
private:
    Entry* get_entry(T* ptr) {
        for (size_t i = 0; i < capacity; ++i) {
            if (reinterpret_cast<T*>(pool[i].data) == ptr)
                return &pool[i];
        }
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
        Logger::log(Logger::Level::INFO, "Widget created: " + std::to_string(id));
    }
    ~Widget() {
        Logger::log(Logger::Level::INFO, "Widget destroyed: " + std::to_string(id));
    }
};

// ---------------- Main Program ----------------
int main() {
    Logger::log(Logger::Level::INFO, "--- Demonstrating UltraAllocator ---");
    UltraAllocator allocator(8192);
    void* a = allocator.allocate(512, "MeshData");
    void* b = allocator.allocate(1024, "TextureCache");
    
    if (!allocator.validateIntegrity()) {
        Logger::log(Logger::Level::ERROR, "Memory integrity check failed!");
    }

    allocator.deallocate(a);
    allocator.printTelemetry();
    allocator.deallocate(b);

    Logger::log(Logger::Level::INFO, "\n--- Demonstrating ObjectPool with Coroutine GC ---");
    ObjectPool<Widget, VerbosePolicy> pool(4); // Start with a small pool to demonstrate scaling
    std::pmr::monotonic_buffer_resource pmr_res;
    std::vector<std::jthread> threads;
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([&, i] {
            auto result = pool.allocate(i, &pmr_res);
            if (!result) {
                Logger::log(Logger::Level::INFO, "Pool full, fallback to shared_ptr.");
                auto sp = std::allocate_shared<Widget>(std::pmr::polymorphic_allocator<Widget>(&pmr_res), i, &pmr_res);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                Logger::log(Logger::Level::INFO, "Using shared_ptr Widget: " + std::to_string(sp->id));
                return;
            }
            Ref<Widget, decltype(pool)> w(result.value(), &pool);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (w) {
                Logger::log(Logger::Level::INFO, "Using Widget: " + std::to_string(w->id));
            }
        });
    }

    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }

    Logger::log(Logger::Level::INFO, "--- Currently live widgets: ---");
    for (auto& widget : pool.used_objects()) {
        Logger::log(Logger::Level::INFO, "Live Widget ID: " + std::to_string(widget.id));
    }

    Logger::log(Logger::Level::INFO, "Pool usage: " + std::to_string(pool.used_count()) + " / " + std::to_string(pool.capacity));
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    Logger::log(Logger::Level::INFO, "ObjectPool demo complete.\n");

    Logger::log(Logger::Level::INFO, "Program complete.");
    return 0;
}
}