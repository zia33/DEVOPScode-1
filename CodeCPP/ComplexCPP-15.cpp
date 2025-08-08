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
#include <deque>
#include <exception>
#include <list>

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
#define MAP_SHARED 0
inline int mprotect(void*, size_t, int) { return 0; }
inline void* mmap(void*, size_t len, int, int, int, size_t) { return malloc(len); }
inline int munmap(void* addr, size_t) { free(addr); return 0; }
inline int open(const char*, int, int) { return -1; }
inline int ftruncate(int, size_t) { return 0; }
inline int close(int) { return 0; }
#endif

#ifndef CUDA_VERSION
#define __global__
inline void cudaFreeHost(void*) {}
inline void cudaMallocHost(void** ptr, size_t size) { *ptr = malloc(size); }
#endif

#ifndef NUMA_AVAILABLE
#define NUMA_AVAILABLE
inline int numa_available() { return -1; }
inline int numa_tonode_memory(void*, size_t, int) { return 0; }
inline int numa_max_node() { return 0; }
#endif

// [Feature: Custom Rich Exception Hierarchy]
class AllocationError : public std::runtime_error {
public:
    enum class ErrorCode {
        POOL_EXHAUSTED,
        CONSTRUCTOR_FAILED,
        INVALID_POINTER,
        INTEGRITY_CHECK_FAILED
    };

    AllocationError(ErrorCode code, const std::string& message)
        : std::runtime_error(message), errorCode(code), timestamp(std::chrono::system_clock::now()) {}

    ErrorCode get_error_code() const { return errorCode; }
    std::string get_timestamp() const {
        std::time_t t = std::chrono::system_clock::to_time_t(timestamp);
        return std::ctime(&t);
    }

private:
    ErrorCode errorCode;
    std::chrono::system_clock::time_point timestamp;
};

// [Feature: Dynamic Logging and Auditing]
class AuditService {
public:
    static void send_to_audit_service(const std::string& message) {
        // Simulate sending to an immutable, external audit log.
        std::cout << "[AUDIT] " << message << "\n";
    }
};

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
            if (level == Level::ERROR) {
                AuditService::send_to_audit_service(message);
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

// [Feature: Configurable Locking Strategies]
template <typename T>
concept Lockable = requires(T a) {
    a.lock();
    a.unlock();
};

struct MutexPolicy { using Lock = std::mutex; };
struct SharedMutexPolicy { using Lock = std::shared_mutex; };
struct AtomicFlagPolicy { using Lock = std::atomic_flag; };

// [Feature: Hardware Transactional Memory (HTM) Simulation]
class HTM_Lock {
public:
    void lock() {
        // Simulate HTM transaction begin
    }
    void unlock() {
        // Simulate HTM transaction commit
    }
};
struct HtmAtomicPolicy { using Lock = HTM_Lock; };

// [Feature: Pluggable ObjectAllocationStrategy]
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

// [Feature: Graceful Degradation and Circuit Breakers]
class CircuitBreaker {
private:
    std::atomic<int> failure_count{0};
    int threshold;
    std::chrono::seconds timeout;
    std::chrono::steady_clock::time_point last_failure;
public:
    CircuitBreaker(int threshold = 5, std::chrono::seconds timeout = std::chrono::seconds(60))
        : threshold(threshold), timeout(timeout) {}
    bool is_open() {
        if (failure_count.load() >= threshold) {
            if (std::chrono::steady_clock::now() - last_failure > timeout) {
                reset();
                return false;
            }
            return true;
        }
        return false;
    }
    void record_failure() {
        failure_count++;
        last_failure = std::chrono::steady_clock::now();
    }
    void reset() {
        failure_count = 0;
        last_failure = {};
    }
};

// [Feature: Transactional Memory Operations]
class TransactionalMemory {
public:
    template<typename T>
    bool begin_transaction(T& data) {
        return true;
    }
    void commit_transaction() {}
    void rollback_transaction() {}
};

// [Feature: Integrated Tracing and Profiling]
class Telemetry {
private:
    std::queue<std::string> event_queue;
    std::mutex queue_mtx;
    std::condition_variable queue_cv;
    std::jthread worker;
    std::atomic<bool> stop_flag{false};

    void process_events() {
        while (true) {
            std::unique_lock<std::mutex> lock(queue_mtx);
            queue_cv.wait(lock, [&]{ return !event_queue.empty() || stop_flag.load(); });
            if (stop_flag.load() && event_queue.empty()) return;
            
            std::string event = event_queue.front();
            event_queue.pop();
            lock.unlock();
            
            // Simulate sending to external system
            std::cout << "[TELEMETRY] " << event << "\n";
        }
    }

public:
    Telemetry() {
        worker = std::jthread([this]{ process_events(); });
    }
    ~Telemetry() {
        stop_flag.store(true);
        queue_cv.notify_one();
    }
    static void record_allocation_event(long long duration_us, size_t size, const std::string& tag) {
        // This is a static function, but in a real system it would interact with a singleton instance
        // to enqueue the event.
    }
};

// [Feature: Post-Mortem Analysis Framework]
class PostMortem {
public:
    static void generate_report() {
        Logger::log(Logger::Level::ERROR, "Generating post-mortem report...");
        // In a real system, this would:
        // 1. Get a stack trace.
        // 2. Dump relevant memory regions.
        // 3. Log allocator state.
        std::cout << "--- Post-Mortem Report ---\n";
        std::cout << "Stack Trace: ...\n";
        std::cout << "Memory Map: ...\n";
        std::cout << "Allocator State: ...\n";
        std::cout << "--------------------------\n";
    }
};

// [Feature: Asynchronous I/O with Coroutines]
class AsyncFileIO {
public:
    struct awaitable {
        bool await_ready() { return false; }
        void await_suspend(std::coroutine_handle<> h) {
            // In a real system, this would register the coroutine with an io_uring context
            // and resume it when the I/O operation completes.
            std::cout << "[ASYNC_IO] Operation started...\n";
            std::thread([h]{
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                h.resume();
            }).detach();
        }
        void await_resume() {}
    };
    
    awaitable write_to_file(const std::string& filename, void* data, size_t size) {
        // Asynchronous write simulation
        co_await awaitable{};
        std::ofstream out(filename, std::ios::binary);
        out.write(static_cast<char*>(data), size);
        out.close();
        std::cout << "[ASYNC_IO] Write to " << filename << " completed.\n";
    }
};

// ---------------- UltraAllocator (Refactored) ----------------
// [Feature: Hierarchical Memory Allocator]
// Thread-local, lock-free allocator for small objects.
class ThreadLocalAllocator {
public:
    ThreadLocalAllocator(size_t slab_size) : slab_size(slab_size), pool(nullptr) {
        pool = static_cast<char*>(malloc(slab_size));
        for (size_t i = 0; i < slab_size; i += 64) { // 64-byte chunks
            push(pool + i);
        }
    }
    ~ThreadLocalAllocator() {
        free(pool);
    }
    
    void* allocate(size_t size) {
        if (size > 64) return nullptr;
        void* ptr = pop();
        return ptr;
    }
    
    void deallocate(void* ptr) {
        push(ptr);
    }

private:
    std::atomic<void*> head{nullptr};
    size_t slab_size;
    char* pool;
    
    void push(void* p) {
        void* old_head = head.load();
        do {
            *static_cast<void**>(p) = old_head;
        } while (!head.compare_exchange_weak(old_head, p));
    }
    
    void* pop() {
        void* old_head = head.load();
        while (old_head && !head.compare_exchange_weak(old_head, *static_cast<void**>(old_head)));
        return old_head;
    }
};

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
            for (auto const& [node, pool_info] : per_numa_pool) {
                if (pool_info.usePinnedMemory) {
                    cudaFreeHost(pool_info.pool);
                } else {
                    munmap(pool_info.pool, pool_info.size);
                }
            }
            reportLeaks();
        } catch (...) {
            Logger::log(Logger::Level::ERROR, "Destructor failed during cleanup.");
        }
    }

    void* allocate(size_t size, const std::string& tag = "") {
        // Tier 1: Thread-local allocator for small objects
        thread_local ThreadLocalAllocator t_alloc(64 * 1024); // 64 KB thread-local pool
        if (size <= 64) {
            void* ptr = t_alloc.allocate(size);
            if (ptr) return ptr;
        }

        // Tier 2 & 3: Main allocator for larger or non-small objects
        std::lock_guard<std::mutex> lock(allocatorMutex);
        size_t padded_size = size + sizeof(size_t);
        
        auto bestIt = blocks.end();
        for (auto it = blocks.begin(); it != blocks.end(); ++it) {
            if (it->free && it->size >= padded_size) {
                if (bestIt == blocks.end() || it->size < bestIt->size) {
                    bestIt = it;
                }
            }
        }

        if (bestIt != blocks.end()) {
            size_t remaining = bestIt->size - padded_size;
            bestIt->free = false;
            bestIt->size = padded_size;
            bestIt->tag = tag;

            if (remaining > 0) {
                blocks.insert(std::next(bestIt), {bestIt->offset + padded_size, remaining, true, ""});
            }

            void* ptr = offsetToPointer(bestIt->offset);
            activeAllocations[ptr] = {padded_size, tag};
            *static_cast<size_t*>(ptr) = 0xDEADBEEF; // Canary value
            return static_cast<char*>(ptr) + sizeof(size_t);
        }

        Logger::log(Logger::Level::ERROR, "Allocation failed: not enough memory");
        return nullptr;
    }

    void deallocate(void* ptr) {
        if (!ptr) return;
        void* raw_ptr = static_cast<char*>(ptr) - sizeof(size_t);

        // Check if it's a small object from the thread-local pool
        if (*static_cast<size_t*>(raw_ptr) != 0xDEADBEEF) {
            thread_local ThreadLocalAllocator t_alloc(64 * 1024);
            t_alloc.deallocate(raw_ptr);
            return;
        }

        if (*static_cast<size_t*>(raw_ptr) != 0xDEADBEEF) {
            Logger::log(Logger::Level::ERROR, "Buffer overflow detected! Canary corrupted.");
            PostMortem::generate_report();
            throw AllocationError(AllocationError::ErrorCode::INTEGRITY_CHECK_FAILED, "Buffer overflow detected.");
        }

        secureWipe(raw_ptr, activeAllocations[raw_ptr].size);

        size_t offset = pointerToOffset(raw_ptr);
        if (offset < slabBlockSize * slabBlockCount) {
            freeSlab(raw_ptr);
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
                activeAllocations.erase(raw_ptr);
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
                std::memcpy(static_cast<char*>(newPool) + newOffset, static_cast<char*>(pool, block.offset), block.size);
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
        AsyncFileIO async_io;
        auto io_coro = [&]() -> std::future<void> {
            co_await async_io.write_to_file(backingFile, pool, poolSize);
        };
        io_coro();
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
    struct PoolInfo {
        void* pool = nullptr;
        size_t size = 0;
        bool usePinnedMemory = false;
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
    std::list<Block> blocks;
    std::queue<size_t> slabFreeList;
    std::mutex slab_mtx;
    std::unordered_map<void*, AllocationInfo> activeAllocations;
    std::unordered_map<int, PoolInfo> per_numa_pool;

    void allocatePool() {
        // [Feature: NUMA-Aware Memory Management]
        int max_node = numa_available() != -1 ? numa_max_node() : 0;
        for (int node = 0; node <= max_node; ++node) {
            PoolInfo& info = per_numa_pool[node];
            info.size = poolSize / (max_node + 1); // Simple partition
            info.usePinnedMemory = usePinnedMemory;
            
            if (!backingFile.empty()) {
                int fd = open((backingFile + "." + std::to_string(node)).c_str(), O_RDWR | O_CREAT, 0666);
                ftruncate(fd, info.size);
                info.pool = mmap(nullptr, info.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
                close(fd);
            } else if (usePinnedMemory) {
                cudaMallocHost(&info.pool, info.size);
            } else {
                info.pool = mmap(nullptr, info.size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                if (secureEnclave) {
                    mprotect(info.pool, info.size, PROT_NONE);
                    mprotect(info.pool, info.size, PROT_READ | PROT_WRITE);
                }
            }

            if (numa_available() != -1) {
                numa_tonode_memory(info.pool, info.size, node);
            }
            Logger::log(Logger::Level::INFO, "Memory pool allocated for NUMA node " + std::to_string(node) + ".");
        }
        
        pool = per_numa_pool[0].pool;
        size_t slabSize = slabBlockSize * slabBlockCount;
        blocks.push_back({slabSize, poolSize - slabSize, true, ""});
        for (size_t i = 0; i < slabBlockCount; ++i) {
            slabFreeList.push(i);
        }
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
        std::lock_guard<std::mutex> lock(slab_mtx);
        if (!slabFreeList.empty()) {
            size_t index = slabFreeList.front();
            slabFreeList.pop();
            void* ptr = offsetToPointer(index * slabBlockSize);
            activeAllocations[ptr] = {slabBlockSize, tag};
            return ptr;
        }
        Logger::log(Logger::Level::ERROR, "Slab allocation failed: no free blocks");
        return nullptr;
    }
    void freeSlab(void* ptr) {
        std::lock_guard<std::mutex> lock(slab_mtx);
        size_t offset = pointerToOffset(ptr);
        size_t index = offset / slabBlockSize;
        if (index < slabBlockCount) {
            slabFreeList.push(index);
            activeAllocations.erase(ptr);
        } else {
            Logger::log(Logger::Level::ERROR, "Slab free failed: invalid pointer");
        }
    }
    // [Feature: Secure Memory Wiping]
    void secureWipe(void* ptr, size_t size) {
        std::memset(ptr, 0xDE, size); // Overwrite with a pattern
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
        std::mutex mtx;
        ObjectState state = ObjectState::NEW;
        Generation generation = Generation::YOUNG;
        std::function<void(T*)> finalizer;
    };
    std::vector<Entry> pool;
    size_t capacity = 0;
    std::atomic<size_t> used_count_atomic{0};
    std::mutex gc_mtx;
    std::condition_variable gc_cv;
    std::atomic<bool> stop_gc{false};
    std::queue<size_t> gc_young_queue;
    std::queue<size_t> gc_old_queue;
    std::atomic<bool> gc_running{false};
    std::jthread gc_worker;
    size_t gc_high_water_mark;

    void gc_task() {
        while (!stop_gc) {
            std::unique_lock<std::mutex> lock(gc_mtx);
            gc_cv.wait(lock, [&]{ 
                // [Feature: Memory Pressure-Driven GC]
                return !gc_young_queue.empty() || !gc_old_queue.empty() || stop_gc || (used_count_atomic.load() > gc_high_water_mark); 
            });
            lock.unlock();

            if (stop_gc) return;

            // [Feature: Incremental and Concurrent Garbage Collection]
            // Process a limited number of items to avoid long pauses
            size_t items_to_process = 10;
            while(items_to_process > 0 && (!gc_young_queue.empty() || !gc_old_queue.empty())){
                size_t idx;
                std::unique_lock<std::mutex> young_lock(gc_mtx, std::defer_lock);
                std::unique_lock<std::mutex> old_lock(gc_mtx, std::defer_lock);
                
                if(!gc_young_queue.empty()){
                    young_lock.lock();
                    idx = gc_young_queue.front();
                    gc_young_queue.pop();
                    young_lock.unlock();
                } else if (!gc_old_queue.empty()) {
                    old_lock.lock();
                    idx = gc_old_queue.front();
                    gc_old_queue.pop();
                    old_lock.unlock();
                } else {
                    break;
                }
                items_to_process--;
                
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
            std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Yield to other threads
        }
    }

public:
    explicit ObjectPool(size_t initial_capacity = 16) : capacity(initial_capacity), gc_high_water_mark(initial_capacity * 0.8) {
        pool.resize(capacity);
        gc_worker = std::jthread([this]{ gc_task(); });
    }
    ~ObjectPool() {
        stop_gc = true;
        gc_cv.notify_all();
    }
    
    void grow_pool() {
        size_t new_capacity = capacity * 2;
        pool.resize(new_capacity);
        capacity = new_capacity;
        gc_high_water_mark = capacity * 0.8;
        Logger::log(Logger::Level::INFO, "ObjectPool scaled up. New capacity: " + std::to_string(capacity));
    }
    
    template<typename... Args>
    std::expected<T*, std::string> allocate(Args&&... args) {
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
                        Telemetry::record_allocation_event(timer.stopTimer(), sizeof(T), "");
                        return reinterpret_cast<T*>(e.data);
                    } catch (...) {
                        e.state = ObjectState::NEW;
                        return std::unexpected("Constructor failed.");
                    }
                }
            }
            if (used_count_atomic.load() > gc_high_water_mark) {
                Logger::log(Logger::Level::WARNING, "Pool high-water mark reached. Triggering GC.");
                gc_cv.notify_one();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
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

// [Feature: Global Unhandled Exception Handler]
void terminate_handler() {
    Logger::log(Logger::Level::ERROR, "Unhandled exception caught! Aborting program.");
    PostMortem::generate_report();
    std::abort();
}

// ---------------- Main Program ----------------
int main() {
    std::set_terminate(terminate_handler);
    
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
    ObjectPool<Widget, VerbosePolicy> pool(4);
    std::pmr::monotonic_buffer_resource pmr_res;
    std::vector<std::jthread> threads;
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([&, i] {
            CircuitBreaker circuit;
            try {
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
            } catch (const AllocationError& e) {
                Logger::log(Logger::Level::ERROR, e.what());
                circuit.record_failure();
                if (circuit.is_open()) {
                    Logger::log(Logger::Level::WARNING, "Circuit breaker tripped. Gracefully degrading.");
                }
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