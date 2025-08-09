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
#include <map>
#include <string_view>
#include <span>

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

// =================================================================================
// SECTION 1: CORE SERVICES (EXCEPTIONS, LOGGING, TELEMETRY)
// =================================================================================

// [Feature: Custom Rich Exception Hierarchy]
class AllocationError : public std::runtime_error {
public:
    enum class ErrorCode {
        POOL_EXHAUSTED,
        CONSTRUCTOR_FAILED,
        INVALID_POINTER,
        INTEGRITY_CHECK_FAILED,
        PERMISSION_DENIED
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

// [Feature: Policy-Based Logging]
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

// [Feature: Integrated Tracing and Profiling]
struct TelemetryData {
    std::chrono::system_clock::time_point timestamp;
    size_t active_allocations;
    double usage_percent;
};

class Telemetry {
private:
    std::deque<TelemetryData> history;
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
    
    void record_allocation_event(long long duration_us, size_t size, const std::string& tag) {
        std::lock_guard<std::mutex> lock(queue_mtx);
        event_queue.push("Alloc: " + tag + ", " + std::to_string(size) + " bytes, " + std::to_string(duration_us) + "us");
    }

    void record_pool_state(size_t active_count, double usage) {
        std::lock_guard<std::mutex> lock(queue_mtx);
        if (history.size() > 100) history.pop_front(); // Keep history bounded
        history.push_back({std::chrono::system_clock::now(), active_count, usage});
    }

    std::deque<TelemetryData> get_history() {
        std::lock_guard<std::mutex> lock(queue_mtx);
        return history;
    }
};

// =================================================================================
// SECTION 2: ADVANCED SYSTEM PATTERNS (CIRCUIT BREAKER, POLICIES, ETC.)
// =================================================================================

// [Feature: Configurable Locking Strategies]
template <typename T>
concept Lockable = requires(T a) {
    a.lock();
    a.unlock();
};

struct MutexPolicy { using Lock = std::mutex; };
struct SharedMutexPolicy { using Lock = std::shared_mutex; };

// [Feature: Hardware Transactional Memory (HTM) Simulation]
class HTM_Lock {
public:
    void lock() { /* Simulate HTM transaction begin */ }
    void unlock() { /* Simulate HTM transaction commit */ }
};
struct HtmAtomicPolicy { using Lock = HTM_Lock; };


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

// [Feature: Post-Mortem Analysis Framework]
class PostMortem {
public:
    static void generate_report() {
        Logger::log(Logger::Level::ERROR, "Generating post-mortem report...");
        // In a real system, this would dump stack trace, memory maps, and allocator state.
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
            std::cout << "[ASYNC_IO] Operation started...\n";
            std::thread([h]{
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                h.resume();
            }).detach();
        }
        void await_resume() {}
    };
    
    awaitable write_to_file(const std::string& filename, void* data, size_t size) {
        co_await awaitable{};
        std::ofstream out(filename, std::ios::binary);
        out.write(static_cast<char*>(data), size);
        out.close();
        std::cout << "[ASYNC_IO] Write to " << filename << " completed.\n";
    }
};

// =================================================================================
// SECTION 3: ENTERPRISE & DISTRIBUTED FEATURES (RBAC, QOS, PREDICTION, DISTRIBUTED)
// =================================================================================

// [Feature: Role-Based Access Control (RBAC) for Memory]
enum class Permission { Read, Write, Allocate, Deallocate };
using Role = std::string;

class RBACManager {
private:
    std::map<Role, std::map<std::string, std::vector<Permission>>> permissions;
public:
    void grant(const Role& role, Permission perm, const std::string& pool_name) {
        permissions[role][pool_name].push_back(perm);
    }

    bool check(const Role& role, Permission perm, const std::string& pool_name) const {
        if (auto role_it = permissions.find(role); role_it != permissions.end()) {
            if (auto pool_it = role_it->second.find(pool_name); pool_it != role_it->second.end()) {
                for (const auto& p : pool_it->second) {
                    if (p == perm) return true;
                }
            }
        }
        return false;
    }
};

// [Feature: Quality of Service (QoS) Guarantees]
enum class QoSTier {
    BestEffort,     // Standard memory from general pool
    Guaranteed,     // Memory from a reserved, non-paged pool
    RealTime        // Pinned memory, NUMA-local, highest priority
};


// [Feature: Predictive Resource Scaling with Machine Learning]
class PredictiveModel {
public:
    // Takes recent telemetry data and predicts future needs
    double predict_future_memory_pressure(const std::deque<TelemetryData>& history) {
        // In a real system, this would be a call to an ML inference engine.
        // We simulate a simple trend analysis.
        if (history.size() < 2) return 0.5;
        Logger::log(Logger::Level::DEBUG, "Predicting memory pressure based on telemetry.");
        return (history.back().usage_percent > history.front().usage_percent) ? 0.8 : 0.2;
    }
};


// [Feature: Distributed Object Framework with Location Transparency]
// In a new header, e.g., 'distributed/GlobalObjectID.h'
struct GlobalObjectID {
    uint16_t nodeId;
    void* localAddress;

    bool isLocal() const { return nodeId == 0; /* 0 is conventionally the local node */ }
    auto operator<=>(const GlobalObjectID&) const = default;
};

// In a new header, e.g., 'distributed/Communication.h'
template<typename T>
concept CommunicationPolicy = requires(T policy, GlobalObjectID id, std::span<std::byte> buffer) {
    // These would be async operations in a real implementation
    { policy.read(id, buffer) } -> std::same_as<void>;
    { policy.write(id, buffer) } -> std::same_as<void>;
};

// A dummy TCP policy for the concept
struct TCPPolicy {
    void read(GlobalObjectID id, std::span<std::byte> buffer) { /* Network read */ }
    void write(GlobalObjectID id, std::span<std::byte> buffer) { /* Network write */ }
};


// =================================================================================
// SECTION 4: CORE MEMORY ALLOCATOR (UltraAllocator)
// =================================================================================

// [Feature: Hierarchical Memory Allocator]
// Tier 1: Thread-local, lock-free allocator for small objects.
class ThreadLocalAllocator {
public:
    ThreadLocalAllocator(size_t slab_size) : slab_size(slab_size), pool(nullptr) {
        pool = static_cast<char*>(malloc(slab_size));
        for (size_t i = 0; i < slab_size; i += 64) {
            push(pool + i);
        }
    }
    ~ThreadLocalAllocator() {
        free(pool);
    }
    
    void* allocate(size_t size) {
        if (size > 64) return nullptr;
        return pop();
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
    UltraAllocator(std::string name, size_t poolSize, const std::string& backingFile = "",
                   bool usePinnedMemory = false, bool secureEnclave = false, int numaNode = -1)
        : pool_name(std::move(name)), poolSize(poolSize), backingFile(backingFile),
          usePinnedMemory(usePinnedMemory), secureEnclave(secureEnclave), numaNode(numaNode) {
        allocatePool();
    }

    ~UltraAllocator() {
        try {
            if (usePinnedMemory) {
                cudaFreeHost(pool);
            } else {
                munmap(pool, poolSize);
            }
            reportLeaks();
        } catch (...) {
            Logger::log(Logger::Level::ERROR, "Destructor failed during cleanup.");
        }
    }

    // [Feature Integration: RBAC and QoS]
    void* allocate(size_t size, const Role& requester_role, QoSTier qos = QoSTier::BestEffort, const std::string& tag = "") {
        if (!rbac_manager.check(requester_role, Permission::Allocate, this->pool_name)) {
            throw AllocationError(AllocationError::ErrorCode::PERMISSION_DENIED, "Role '" + requester_role + "' lacks allocation permission on pool '" + pool_name + "'.");
        }
        
        switch (qos) {
            case QoSTier::RealTime:
                 Logger::log(Logger::Level::INFO, "QoS: Allocating from RealTime tier.");
                 // In a real system, this would use a separate, pre-pinned pool.
                 break;
            case QoSTier::Guaranteed:
                 Logger::log(Logger::Level::INFO, "QoS: Allocating from Guaranteed tier.");
                 // This would use a reserved, non-overcommitted pool.
                 break;
            case QoSTier::BestEffort:
            default:
                 break;
        }

        return allocate_internal(size, tag);
    }
    
    // Original allocate for simple cases
    void* allocate(size_t size, const std::string& tag = "") {
        return allocate_internal(size, tag);
    }


    void deallocate(void* ptr) {
        if (!ptr) return;
        void* raw_ptr = static_cast<char*>(ptr) - sizeof(size_t);

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
        std::lock_guard<std::mutex> lock(allocatorMutex);
        for (auto it = blocks.begin(); it != blocks.end(); ++it) {
            if (it->offset == offset) {
                it->free = true;
                it->tag = "";
                // Coalesce with next block if free
                if (std::next(it) != blocks.end() && std::next(it)->free) {
                    it->size += std::next(it)->size;
                    blocks.erase(std::next(it));
                }
                // Coalesce with previous block if free
                if (it != blocks.begin() && std::prev(it)->free) {
                    std::prev(it)->size += it->size;
                    blocks.erase(it);
                }
                activeAllocations.erase(raw_ptr);
                return;
            }
        }
        Logger::log(Logger::Level::ERROR, "Deallocation failed: pointer not found");
    }

    void persistToFile() {
        if (backingFile.empty()) return;
        AsyncFileIO async_io;
        auto io_coro = [&]() -> std::future<void> {
            co_await async_io.write_to_file(backingFile, pool, poolSize);
        };
        io_coro();
    }

    bool validateIntegrity() {
        Logger::log(Logger::Level::INFO, "Simulating memory integrity check...");
        return true;
    }

    void printTelemetry() {
        std::lock_guard<std::mutex> lock(allocatorMutex);
        std::cout << "\n📊 Memory Telemetry for pool '" << pool_name << "':\n";
        std::cout << "Total Pool Size: " << poolSize << " bytes\n";
        std::cout << "Used: " << getUsedMemory() << " bytes\n";
        std::cout << "Freed: " << getFreedMemory() << " bytes\n";
        std::cout << "Active Allocations:\n";
        for (const auto& [ptr, info] : activeAllocations) {
            std::cout << "  Ptr: " << ptr << ", Size: " << info.size << ", Tag: " << info.tag << "\n";
        }
    }
    
    // Public accessor for the RBAC Manager
    RBACManager& get_rbac_manager() { return rbac_manager; }

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
    std::string pool_name;
    size_t poolSize;
    std::string backingFile;
    bool usePinnedMemory;
    bool secureEnclave;
    int numaNode;
    RBACManager rbac_manager;
    std::mutex allocatorMutex;
    std::list<Block> blocks;
    std::unordered_map<void*, AllocationInfo> activeAllocations;

    void* allocate_internal(size_t size, const std::string& tag) {
        // Tier 1: Thread-local allocator for small objects
        thread_local ThreadLocalAllocator t_alloc(64 * 1024);
        if (size <= 64) {
            void* ptr = t_alloc.allocate(size);
            if (ptr) return ptr;
        }

        // Tier 2: Main allocator
        std::lock_guard<std::mutex> lock(allocatorMutex);
        size_t padded_size = size + sizeof(size_t); // for canary
        
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

            if (remaining > sizeof(Block)) { // Don't create tiny slivers
                blocks.insert(std::next(bestIt), {bestIt->offset + padded_size, remaining, true, ""});
            }

            void* ptr = offsetToPointer(bestIt->offset);
            activeAllocations[ptr] = {padded_size, tag};
            *static_cast<size_t*>(ptr) = 0xDEADBEEF; // Canary value
            return static_cast<char*>(ptr) + sizeof(size_t);
        }

        Logger::log(Logger::Level::ERROR, "Allocation failed: not enough memory in pool '" + pool_name + "'");
        return nullptr;
    }
    
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

        // [Feature: NUMA-Aware Memory Management]
        if (numa_available() != -1 && numaNode != -1) {
            numa_tonode_memory(pool, poolSize, numaNode);
            Logger::log(Logger::Level::INFO, "Memory pool '" + pool_name + "' bound to NUMA node " + std::to_string(numaNode) + ".");
        } else {
            Logger::log(Logger::Level::INFO, "Memory pool '" + pool_name + "' allocated.");
        }
        
        blocks.push_back({0, poolSize, true, ""});
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
    
    // [Feature: Secure Memory Wiping]
    void secureWipe(void* ptr, size_t size) {
        std::memset(ptr, 0xDE, size); // Overwrite with a pattern
    }

    void reportLeaks() {
        std::lock_guard<std::mutex> lock(allocatorMutex);
        if (activeAllocations.empty()) {
            std::cout << "\n✅ No memory leaks detected in pool '" << pool_name << "'.\n";
        } else {
            std::cout << "\n🕵️ Leak Report for pool '" << pool_name << "':\n";
            for (const auto& [ptr, info] : activeAllocations) {
                std::cout << "  Leaked Ptr: " << ptr << ", Size: " << info.size << ", Tag: " << info.tag << "\n";
            }
        }
    }
};

// =================================================================================
// SECTION 5: OBJECT POOLING & GARBAGE COLLECTION
// =================================================================================

template<typename T, typename Policy = SilentPolicy>
class ObjectPool {
public:
    enum class ObjectState { NEW, IN_USE, PENDING_GC };
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
    std::jthread gc_worker;
    size_t gc_high_water_mark;
    Telemetry* telemetry_svc = nullptr;

    void gc_task() {
        while (!stop_gc) {
            std::unique_lock<std::mutex> lock(gc_mtx);
            // [Feature: Memory Pressure-Driven GC]
            gc_cv.wait(lock, [&]{ 
                return !gc_young_queue.empty() || !gc_old_queue.empty() || stop_gc || (used_count_atomic.load() > gc_high_water_mark); 
            });

            if (stop_gc) return;

            // [Feature: Incremental and Concurrent Garbage Collection]
            size_t items_to_process = 10;
            while(items_to_process-- > 0 && (!gc_young_queue.empty() || !gc_old_queue.empty())){
                size_t idx;
                if(!gc_young_queue.empty()){
                    idx = gc_young_queue.front();
                    gc_young_queue.pop();
                } else {
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
            if (telemetry_svc) telemetry_svc->record_pool_state(used_count(), (double)used_count() / capacity);
            lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Yield
        }
    }

public:
    explicit ObjectPool(size_t initial_capacity = 16, Telemetry* telemetry = nullptr) 
      : capacity(initial_capacity), gc_high_water_mark(initial_capacity * 0.8), telemetry_svc(telemetry) {
        pool.resize(capacity);
        gc_worker = std::jthread([this]{ gc_task(); });
    }
    ~ObjectPool() {
        stop_gc = true;
        gc_cv.notify_all();
    }
    
    void grow_pool() {
        std::lock_guard<std::mutex> lock(gc_mtx);
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
                if (e.state == ObjectState::NEW) {
                    std::lock_guard<std::mutex> lock(e.mtx);
                    if (e.state == ObjectState::NEW) { // Double-check lock
                        try {
                            e.state = ObjectState::IN_USE;
                            new(e.data) T(std::forward<Args>(args)...);
                            e.ref_count.store(1);
                            Policy::on_allocate(i);
                            used_count_atomic++;
                            if (telemetry_svc) telemetry_svc->record_allocation_event(timer.stopTimer(), sizeof(T), typeid(T).name());
                            return reinterpret_cast<T*>(e.data);
                        } catch (...) {
                            e.state = ObjectState::NEW;
                            return std::unexpected("Constructor failed.");
                        }
                    }
                }
            }
            
            Logger::log(Logger::Level::WARNING, "Pool full. Trying to grow...");
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

    void trigger_gc_if_needed(bool force = false) {
        if (force || used_count_atomic.load() > gc_high_water_mark) {
            Logger::log(Logger::Level::INFO, "GC explicitly triggered.");
            gc_cv.notify_one();
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

    std::string get_status_string() {
        return "Pool Usage: " + std::to_string(used_count()) + " / " + std::to_string(capacity);
    }

    size_t used_count() const { return used_count_atomic.load(); }
    
private:
    Entry* get_entry(T* ptr) {
        if (!ptr) return nullptr;
        // Pointer arithmetic to find the entry index
        auto offset = reinterpret_cast<char*>(ptr) - reinterpret_cast<char*>(pool.data());
        if (offset >= 0 && offset < capacity * sizeof(Entry)) {
            size_t index = offset / sizeof(Entry);
            return &pool[index];
        }
        return nullptr;
    }
};

// [Feature Conceptual Integration: Distributed Object Pool]
template<typename T, CommunicationPolicy Comms = TCPPolicy>
class DistributedObjectPool : public ObjectPool<T> {
private:
    Comms comms_policy;
    std::unordered_map<GlobalObjectID, T*> remote_proxies;
    uint16_t this_node_id = 0;

public:
    std::expected<GlobalObjectID, AllocationError> allocate_distributed(...) {
        auto local_ptr_exp = ObjectPool<T>::allocate(/* args */);
        if (!local_ptr_exp) { /* ... handle error */ }
        return GlobalObjectID{this_node_id, *local_ptr_exp};
    }
};

// Smart Ref Wrapper
template<typename T, typename Pool>
class Ref {
    T* ptr = nullptr;
    Pool* pool = nullptr;
public:
    Ref() = default;
    Ref(T* p, Pool* pl) : ptr(p), pool(pl) {}
    ~Ref() { if (ptr) pool->release(ptr); }
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
    T* operator->() { return ptr; }
    const T* operator->() const { return ptr; }
    T& operator*() { return *ptr; }
    const T& operator*() const { return *ptr; }
    explicit operator bool() const { return ptr != nullptr; }
};

// =================================================================================
// SECTION 6: APPLICATION & DEMONSTRATION
// =================================================================================

struct Widget {
    int id;
    std::pmr::vector<int> data;
    Widget(int id, std::pmr::memory_resource* res = std::pmr::get_default_resource())
        : id(id), data(res) {
        data.resize(5, id);
    }
    ~Widget() { /* Logger messages moved to main to reduce noise */ }
};

// [Feature Integration: Live Introspection and Management]
class ManagementService {
    ObjectPool<Widget>& pool;
    Telemetry& telemetry;
    PredictiveModel model;
    std::jthread service_thread;

public:
    ManagementService(ObjectPool<Widget>& p, Telemetry& t) : pool(p), telemetry(t) {}

    void run() {
        service_thread = std::jthread([this](std::stop_token st) {
            while (!st.stop_requested()) {
                // [Feature Integration: Predictive GC Trigger]
                auto history = telemetry.get_history();
                double pressure = model.predict_future_memory_pressure(history);
                if (pressure > 0.75) {
                    Logger::log(Logger::Level::WARNING, "Predicted high memory pressure! Proactively triggering GC.");
                    pool.trigger_gc_if_needed(true); 
                }

                // Simulate listening on a socket for commands
                // std::string cmd = listen_and_process_commands();
                // process_command(cmd);

                std::this_thread::sleep_for(std::chrono::seconds(5));
            }
        });
    }

    std::string process_command(const std::string& cmd) {
        if (cmd == "STATUS") return pool.get_status_string();
        if (cmd == "TRIGGER_GC") {
            pool.trigger_gc_if_needed(true);
            return "OK";
        }
        return "ERROR: Unknown command";
    }
};

// [Feature: Global Unhandled Exception Handler]
void terminate_handler() {
    Logger::log(Logger::Level::ERROR, "Unhandled exception caught! Aborting program.");
    PostMortem::generate_report();
    std::abort();
}

int main() {
    std::set_terminate(terminate_handler);
    
    Logger::log(Logger::Level::INFO, "--- Demonstrating UltraAllocator with RBAC and QoS ---");
    UltraAllocator allocator("MainPool", 8192);
    
    // Setup RBAC rules
    Role admin_role = "admin";
    Role user_role = "user";
    Role guest_role = "guest";
    allocator.get_rbac_manager().grant(admin_role, Permission::Allocate, "MainPool");
    allocator.get_rbac_manager().grant(user_role, Permission::Allocate, "MainPool");

    try {
        Logger::log(Logger::Level::INFO, "Admin allocating 512 bytes with RealTime QoS...");
        void* a = allocator.allocate(512, admin_role, QoSTier::RealTime, "MeshData");
        Logger::log(Logger::Level::INFO, "User allocating 1024 bytes...");
        void* b = allocator.allocate(1024, user_role, QoSTier::BestEffort, "TextureCache");
        
        allocator.deallocate(a);
        allocator.printTelemetry();
        allocator.deallocate(b);

        Logger::log(Logger::Level::WARNING, "Guest attempting to allocate (should fail)...");
        void* c = allocator.allocate(256, guest_role, QoSTier::BestEffort, "GuestData");
        if (!c) {
             Logger::log(Logger::Level::INFO, "Allocation correctly failed for guest role as expected.");
        }
    } catch(const AllocationError& e) {
        Logger::log(Logger::Level::ERROR, "Caught expected exception: " + std::string(e.what()));
    }
    
    Logger::log(Logger::Level::INFO, "\n--- Demonstrating ObjectPool with Management Service ---");
    Telemetry telemetry_service;
    ObjectPool<Widget, VerbosePolicy> pool(4, &telemetry_service);
    ManagementService manager(pool, telemetry_service);
    manager.run(); // Start the background management thread
    
    std::pmr::monotonic_buffer_resource pmr_res;
    std::vector<Ref<Widget, decltype(pool)>> refs;

    for (int i = 0; i < 10; ++i) {
        auto result = pool.allocate(i, &pmr_res);
        if (result) {
            Logger::log(Logger::Level::INFO, "Created Widget " + std::to_string(i));
            refs.emplace_back(result.value(), &pool);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    Logger::log(Logger::Level::INFO, "--- Releasing first 5 widgets to trigger GC ---");
    refs.erase(refs.begin(), refs.begin() + 5);

    Logger::log(Logger::Level::INFO, manager.process_command("STATUS"));
    std::this_thread::sleep_for(std::chrono::milliseconds(500)); // Wait for GC to run
    
    Logger::log(Logger::Level::INFO, "--- Final status after GC ---");
    Logger::log(Logger::Level::INFO, manager.process_command("STATUS"));

    Logger::log(Logger::Level::INFO, "Program complete.");
    return 0;
}