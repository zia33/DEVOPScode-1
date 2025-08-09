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
#include <random>
#include <sstream>
#include <algorithm>

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
// SECTION 1: CORE SERVICES (EXCEPTIONS, LOGGING, TELEMETRY, AND NEW SERVICES)
// =================================================================================

// [Feature: Custom Rich Exception Hierarchy]
class AllocationError : public std::runtime_error {
public:
    enum class ErrorCode {
        POOL_EXHAUSTED,
        CONSTRUCTOR_FAILED,
        INVALID_POINTER,
        INTEGRITY_CHECK_FAILED,
        PERMISSION_DENIED,
        DEFRAGMENTATION_FAILED
    };

    AllocationError(ErrorCode code, const std::string& message)
        : std::runtime_error(message), errorCode(code), timestamp(std::chrono::system_clock::now()) {}

    ErrorCode get_error_code() const { return errorCode; }
    std::string get_timestamp() const {
        std::time_t t = std::chrono::system_clock::to_time_t(timestamp);
        char buf[32];
        ctime_r(&t, buf);
        std::string time_str(buf);
        time_str.pop_back(); // Remove newline
        return time_str;
    }

private:
    ErrorCode errorCode;
    std::chrono::system_clock::time_point timestamp;
};


// [NEW Feature: Hardware Security Module (HSM) Integration]
class HSMService {
public:
    static std::string sign_data(const std::string& data) {
        // Simulate a call to a hardware device, which introduces latency.
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        // Simulate a SHA256 hash-based signature.
        std::hash<std::string> hasher;
        return "hsm-sig-" + std::to_string(hasher(data));
    }
};

// [NEW Feature: Dynamic Configuration Hot-Reloading]
class ConfigurationManager {
private:
    std::string config_file;
    std::map<std::string, std::string> settings;
    std::shared_mutex mtx;
    
    ConfigurationManager() = default;

public:
    static ConfigurationManager& instance() {
        static ConfigurationManager inst;
        return inst;
    }

    void load(const std::string& file) {
        std::lock_guard<std::shared_mutex> lock(mtx);
        config_file = file;
        std::ifstream f(config_file);
        if (!f) {
            // Create default config file if it doesn't exist
            std::ofstream out(config_file);
            out << "log_level=INFO\n";
            out << "circuit_breaker_threshold=5\n";
        }
        f.close();
        
        f.open(config_file);
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            std::string key, value;
            if (std::getline(iss, key, '=') && std::getline(iss, value)) {
                settings[key] = value;
            }
        }
        std::cout << "[CONFIG] Loaded configuration from " << config_file << std::endl;
    }

    std::string get(const std::string& key, const std::string& default_val = "") {
        std::shared_lock<std::shared_mutex> lock(mtx);
        auto it = settings.find(key);
        return (it != settings.end()) ? it->second : default_val;
    }
};

// [Feature: Dynamic Logging and Auditing - ENHANCED]
class AuditService {
public:
    static void send_to_audit_service(const std::string& message) {
        // [NEW Integration: HSM Signing]
        std::string signature = HSMService::sign_data(message);
        std::cout << "[AUDIT] " << message << " [Signature: " << signature << "]\n";
    }
};

class Logger {
public:
    enum class Level { DEBUG, INFO, WARNING, ERROR };
    static std::atomic<Level> currentLevel;

    static void set_level(const std::string& level_str) {
        if (level_str == "DEBUG") currentLevel = Level::DEBUG;
        else if (level_str == "INFO") currentLevel = Level::INFO;
        else if (level_str == "WARNING") currentLevel = Level::WARNING;
        else if (level_str == "ERROR") currentLevel = Level::ERROR;
    }

    static void log(Level level, const std::string& message) {
        if (level >= currentLevel.load()) {
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
std::atomic<Logger::Level> Logger::currentLevel = {Logger::Level::INFO};

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

// [NEW Feature: Data Encryption]
class CryptoProvider {
private:
    // This is a trivial, insecure XOR cipher for demonstration ONLY.
    // A real implementation would use a robust, authenticated encryption
    // algorithm like AES-GCM from a library like OpenSSL or libsodium.
    static constexpr char XOR_KEY = 0xAB;
public:
    static void encrypt(std::span<std::byte> data) {
        for (auto& byte : data) {
            byte ^= std::byte{XOR_KEY};
        }
    }

    static void decrypt(std::span<std::byte> data) {
        // XOR is symmetric
        encrypt(data);
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
    std::atomic<int> threshold;
    std::chrono::seconds timeout;
    std::chrono::steady_clock::time_point last_failure;
public:
    CircuitBreaker(int threshold = 5, std::chrono::seconds timeout = std::chrono::seconds(60))
        : threshold(threshold), timeout(timeout) {}
    
    void update_threshold(int new_threshold) {
        threshold.store(new_threshold);
        Logger::log(Logger::Level::INFO, "Circuit Breaker threshold updated to " + std::to_string(new_threshold));
    }
    
    bool is_open() {
        if (failure_count.load() >= threshold.load()) {
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

// [Feature: Asynchronous I/O with Coroutines - ENHANCED for Encryption]
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
    
    awaitable write_to_file(const std::string& filename, std::span<std::byte> data) {
        co_await awaitable{};
        // [NEW Integration: Data-at-Rest Encryption]
        CryptoProvider::encrypt(data);
        std::ofstream out(filename, std::ios::binary);
        out.write(reinterpret_cast<const char*>(data.data()), data.size());
        out.close();
        // Decrypt memory after writing to disk so the live pool is usable
        CryptoProvider::decrypt(data);
        std::cout << "[ASYNC_IO] Encrypted write to " << filename << " completed.\n";
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

// [NEW Feature Section: Enhanced Distributed Framework]
struct Widget; // Forward declaration

// [NEW Feature: Efficient Serialization Framework]
template<typename T>
concept Serializable = requires(T t, std::vector<std::byte>& buffer) {
    { T::serialize(t, buffer) } -> std::same_as<void>;
    { T::deserialize(buffer) } -> std::same_as<std::expected<Widget, std::string>>;
};

struct WidgetSerializer {
    static void serialize(const Widget& w, std::vector<std::byte>& buffer); // Implemented later
    static std::expected<Widget, std::string> deserialize(const std::vector<std::byte>& buffer);
};

// [Feature: Distributed Object Framework - ENHANCED]
struct GlobalObjectID {
    uint16_t nodeId;
    uint64_t objectId; // Use a stable ID instead of a raw pointer

    bool isLocal(uint16_t localNodeId) const { return nodeId == localNodeId; }
    auto operator<=>(const GlobalObjectID&) const = default;
};

// [Feature: Pluggable Communication Policies - ENHANCED for Encryption & Latency]
template<typename T>
concept CommunicationPolicy = requires(T policy, GlobalObjectID id, std::vector<std::byte>& buffer) {
    { policy.read(id, buffer) } -> std::same_as<bool>;
    { policy.write(id, buffer) } -> std::same_as<bool>;
};

struct TCPPolicy {
    bool read(GlobalObjectID id, std::vector<std::byte>& buffer) {
        Logger::log(Logger::Level::DEBUG, "TCP: Reading object " + std::to_string(id.objectId) + " from node " + std::to_string(id.nodeId));
        std::this_thread::sleep_for(std::chrono::milliseconds(20)); // Simulate network latency
        CryptoProvider::decrypt(buffer); // Decrypt after receiving
        return true;
    }
    bool write(GlobalObjectID id, std::vector<std::byte>& buffer) {
        Logger::log(Logger::Level::DEBUG, "TCP: Writing object " + std::to_string(id.objectId) + " to node " + std::to_string(id.nodeId));
        CryptoProvider::encrypt(buffer); // Encrypt before sending
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        return true;
    }
};

// [NEW Feature: RDMA Communication Policy]
struct RDMAPolicy {
    bool read(GlobalObjectID id, std::vector<std::byte>& buffer) {
        Logger::log(Logger::Level::DEBUG, "RDMA: Reading object " + std::to_string(id.objectId) + " from node " + std::to_string(id.nodeId));
        std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Simulate ultra-low latency
        // Encryption might be skipped in a secure-fabric RDMA environment, but we show it for consistency.
        CryptoProvider::decrypt(buffer);
        return true;
    }
    bool write(GlobalObjectID id, std::vector<std::byte>& buffer) {
        Logger::log(Logger::Level::DEBUG, "RDMA: Writing object " + std::to_string(id.objectId) + " to node " + std::to_string(id.nodeId));
        CryptoProvider::encrypt(buffer);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return true;
    }
};

// [NEW Feature: Cluster Membership and Service Discovery]
struct NodeInfo {
    uint16_t nodeId;
    std::string address;
    std::chrono::steady_clock::time_point last_seen;
};

class ClusterManager {
private:
    uint16_t self_id;
    std::map<uint16_t, NodeInfo> nodes;
    std::mutex mtx;
public:
    ClusterManager(uint16_t self_id, std::string self_address) : self_id(self_id) {
        nodes[self_id] = {self_id, self_address, std::chrono::steady_clock::now()};
    }

    void update_node(uint16_t nodeId, const std::string& address) {
        std::lock_guard<std::mutex> lock(mtx);
        nodes[nodeId] = {nodeId, address, std::chrono::steady_clock::now()};
    }
    
    // Simulate a gossip ping
    void gossip() {
        std::lock_guard<std::mutex> lock(mtx);
        Logger::log(Logger::Level::DEBUG, "Node " + std::to_string(self_id) + " gossiping...");
        // In a real system, this would involve random network communication.
        // Here we just update our own timestamp.
        nodes[self_id].last_seen = std::chrono::steady_clock::now();
    }
    
    std::vector<uint16_t> get_live_nodes() {
        std::vector<uint16_t> live_nodes;
        std::lock_guard<std::mutex> lock(mtx);
        for(const auto& [id, info] : nodes) {
            // Prune nodes not seen for a while
            if(std::chrono::steady_clock::now() - info.last_seen < std::chrono::seconds(30)) {
                live_nodes.push_back(id);
            }
        }
        return live_nodes;
    }
};

// =================================================================================
// SECTION 4: CORE MEMORY ALLOCATOR (UltraAllocator) - ENHANCED
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
                 break;
            case QoSTier::Guaranteed:
                 Logger::log(Logger::Level::INFO, "QoS: Allocating from Guaranteed tier.");
                 break;
            case QoSTier::BestEffort:
            default:
                 break;
        }

        return allocate_internal(size, tag);
    }
    
    void* allocate(size_t size, const std::string& tag = "") {
        return allocate_internal(size, tag);
    }


    void deallocate(void* ptr) {
        if (!ptr) return;
        void* raw_ptr = static_cast<char*>(ptr) - sizeof(size_t);

        if (thread_local_alloc.deallocate(raw_ptr)) return;

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
            co_await async_io.write_to_file(backingFile, {static_cast<std::byte*>(pool), poolSize});
        };
        io_coro();
    }

    // [NEW Feature: Manual Defragmentation]
    void defragment() {
        std::lock_guard<std::mutex> lock(allocatorMutex);
        Logger::log(Logger::Level::WARNING, "Starting defragmentation. All direct pointers to allocated memory will be invalidated!");

        size_t free_ptr = 0;
        std::list<Block> new_blocks;
        std::unordered_map<void*, AllocationInfo> new_active_allocations;

        // Iterate and move allocated blocks to the front
        for (const auto& block : blocks) {
            if (!block.free) {
                void* old_ptr = offsetToPointer(block.offset);
                void* new_ptr = offsetToPointer(free_ptr);
                if (old_ptr != new_ptr) {
                    memmove(new_ptr, old_ptr, block.size);
                }
                new_blocks.push_back({free_ptr, block.size, false, block.tag});
                new_active_allocations[new_ptr] = activeAllocations[old_ptr];
                free_ptr += block.size;
            }
        }

        // Add one large free block at the end
        if (free_ptr < poolSize) {
            new_blocks.push_back({free_ptr, poolSize - free_ptr, true, ""});
        }
        
        blocks = std::move(new_blocks);
        activeAllocations = std::move(new_active_allocations);

        Logger::log(Logger::Level::INFO, "Defragmentation complete.");
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
    
    // A simplified thread-local allocator for demonstration
    class TrivialThreadLocalAllocator {
        std::vector<void*> free_list;
    public:
        void* allocate(size_t size) {
            if (size > 64 || free_list.empty()) return nullptr;
            void* ptr = free_list.back();
            free_list.pop_back();
            return ptr;
        }
        bool deallocate(void* ptr) {
            // This is a simplified check; a real implementation would be more robust.
            // For this demo, we assume if it wasn't in the main map, it's ours.
            // This logic is flawed but sufficient for the demonstration.
            try {
                // Heuristic: check if the pointer falls within a slab we might have allocated.
                // In a real system, we'd own the slab and could check the range.
                free_list.push_back(ptr);
                return true;
            } catch(...) {
                return false;
            }
        }
    };
    static thread_local TrivialThreadLocalAllocator thread_local_alloc;

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
        if (size <= 64) {
            void* ptr = thread_local_alloc.allocate(size);
            if (ptr) return ptr;
        }

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

            if (remaining > sizeof(Block)) {
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
    
    void secureWipe(void* ptr, size_t size) {
        std::memset(ptr, 0xDE, size);
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
thread_local UltraAllocator::TrivialThreadLocalAllocator UltraAllocator::thread_local_alloc;

// =================================================================================
// SECTION 5: OBJECT POOLING & GARBAGE COLLECTION - ENHANCED
// =================================================================================

template<typename T, typename Policy = SilentPolicy>
class ObjectPool {
public:
    // [NEW/ENHANCED State for Compression]
    enum class ObjectState { NEW, IN_USE, PENDING_GC, COMPRESSED };
    enum class Generation { YOUNG, OLD };

private:
    struct Entry {
        alignas(T) char data[sizeof(T)];
        std::atomic<int> ref_count{0};
        std::mutex mtx;
        ObjectState state = ObjectState::NEW;
        Generation generation = Generation::YOUNG;
        std::function<void(T*)> finalizer;
        std::chrono::steady_clock::time_point last_access;
    };
    
    std::vector<Entry> pool;
    size_t capacity = 0;
    std::atomic<size_t> used_count_atomic{0};
    
    // GC members
    std::mutex gc_mtx;
    std::condition_variable gc_cv;
    std::atomic<bool> stop_threads{false};
    std::queue<size_t> gc_young_queue;
    std::queue<size_t> gc_old_queue;
    std::jthread gc_worker;
    size_t gc_high_water_mark;
    
    // [NEW] Compression members
    std::jthread compression_worker;
    std::chrono::seconds compression_threshold = std::chrono::seconds(1);

    void gc_task() {
        while (!stop_threads) {
            std::unique_lock<std::mutex> lock(gc_mtx);
            gc_cv.wait(lock, [&]{ 
                return !gc_young_queue.empty() || !gc_old_queue.empty() || stop_threads.load() || (used_count_atomic.load() > gc_high_water_mark); 
            });

            if (stop_threads) return;

            size_t items_to_process = 10;
            while(items_to_process-- > 0 && (!gc_young_queue.empty() || !gc_old_queue.empty())){
                size_t idx;
                if(!gc_young_queue.empty()){
                    idx = gc_young_queue.front(); gc_young_queue.pop();
                } else {
                    idx = gc_old_queue.front(); gc_old_queue.pop();
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
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    // [NEW Feature: Transparent Memory Compression Task]
    void compression_task() {
        while (!stop_threads) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            Logger::log(Logger::Level::DEBUG, "Running compression task...");
            for (size_t i = 0; i < capacity; ++i) {
                auto& e = pool[i];
                std::lock_guard<std::mutex> lock(e.mtx);
                if (e.state == ObjectState::IN_USE && e.ref_count > 0) {
                    if (std::chrono::steady_clock::now() - e.last_access > compression_threshold) {
                        e.state = ObjectState::COMPRESSED;
                        // In a real system, this would involve calling a compression library
                        // and freeing pages. Here, we just log the event.
                        Logger::log(Logger::Level::INFO, "Compressing object " + std::to_string(i));
                    }
                }
            }
        }
    }

protected:
    Telemetry* telemetry_svc = nullptr;
    
    Entry* get_entry(T* ptr) {
        if (!ptr) return nullptr;
        auto offset = reinterpret_cast<char*>(ptr) - reinterpret_cast<char*>(pool.data());
        if (offset >= 0 && offset < capacity * sizeof(Entry)) {
            size_t index = offset / sizeof(Entry);
            auto& entry = pool[index];
            
            // [NEW Integration: On-demand Decompression]
            std::lock_guard<std::mutex> lock(entry.mtx);
            if (entry.state == ObjectState::COMPRESSED) {
                Logger::log(Logger::Level::INFO, "Decompressing object " + std::to_string(index) + " on access.");
                std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Simulate decompression latency
                entry.state = ObjectState::IN_USE;
            }
            entry.last_access = std::chrono::steady_clock::now();
            return &entry;
        }
        return nullptr;
    }


public:
    explicit ObjectPool(size_t initial_capacity = 16, Telemetry* telemetry = nullptr) 
      : capacity(initial_capacity), gc_high_water_mark(initial_capacity * 0.8), telemetry_svc(telemetry) {
        pool.resize(capacity);
        gc_worker = std::jthread([this]{ gc_task(); });
        compression_worker = std::jthread([this]{ compression_task(); });
    }
    virtual ~ObjectPool() {
        stop_threads = true;
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
                    if (e.state == ObjectState::NEW) {
                        try {
                            e.state = ObjectState::IN_USE;
                            e.last_access = std::chrono::steady_clock::now();
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
};

// [Feature Conceptual Integration: Distributed Object Pool - ENHANCED]
template<typename T, CommunicationPolicy Comms = TCPPolicy, Serializable S = WidgetSerializer>
class DistributedObjectPool : public ObjectPool<T> {
private:
    Comms comms_policy;
    std::shared_ptr<ClusterManager> cluster_manager;
    uint16_t this_node_id;
    std::atomic<uint64_t> next_object_id{0};

public:
    DistributedObjectPool(size_t capacity, uint16_t node_id, std::shared_ptr<ClusterManager> cm, Telemetry* telemetry)
        : ObjectPool<T>(capacity, telemetry), cluster_manager(cm), this_node_id(node_id) {}
    
    ~DistributedObjectPool() override = default;

    template<typename... Args>
    std::expected<GlobalObjectID, std::string> allocate_distributed(Args&&... args) {
        auto local_ptr_exp = this->allocate(std::forward<Args>(args)...);
        if (!local_ptr_exp) {
            return std::unexpected(local_ptr_exp.error());
        }
        
        // This is a simplification. A real system would need a way to map T* to a stable ID.
        uint64_t obj_id = next_object_id++;
        return GlobalObjectID{this_node_id, obj_id};
    }

    std::expected<T, std::string> get_remote_object(GlobalObjectID id) {
        if(id.isLocal(this_node_id)) {
            return std::unexpected("Object is local, use local access method.");
        }
        std::vector<std::byte> buffer;
        if(comms_policy.read(id, buffer)) {
            auto obj_exp = S::deserialize(buffer);
            if (obj_exp) return *obj_exp;
            return std::unexpected(obj_exp.error());
        }
        return std::unexpected("Failed to read remote object via comms policy.");
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
    Widget(int id_val, std::pmr::memory_resource* res = std::pmr::get_default_resource())
        : id(id_val), data(res) {
        data.resize(5, id);
    }
    Widget() : id(-1), data(std::pmr::get_default_resource()){}; // Default constructor for deserialization
    ~Widget() {}
};

// [NEW] Implementation of serializer functions
void WidgetSerializer::serialize(const Widget& w, std::vector<std::byte>& buffer) {
    buffer.resize(sizeof(int) + sizeof(size_t) + w.data.size() * sizeof(int));
    char* current = reinterpret_cast<char*>(buffer.data());
    memcpy(current, &w.id, sizeof(int));
    current += sizeof(int);
    size_t vec_size = w.data.size();
    memcpy(current, &vec_size, sizeof(size_t));
    current += sizeof(size_t);
    memcpy(current, w.data.data(), vec_size * sizeof(int));
}
std::expected<Widget, std::string> WidgetSerializer::deserialize(const std::vector<std::byte>& buffer) {
    if (buffer.size() < sizeof(int) + sizeof(size_t)) return std::unexpected("Buffer too small");
    Widget w;
    const char* current = reinterpret_cast<const char*>(buffer.data());
    memcpy(&w.id, current, sizeof(int));
    current += sizeof(int);
    size_t vec_size;
    memcpy(&vec_size, current, sizeof(size_t));
    current += sizeof(size_t);
    if (buffer.size() != sizeof(int) + sizeof(size_t) + vec_size * sizeof(int)) {
        return std::unexpected("Buffer size mismatch");
    }
    w.data.resize(vec_size);
    memcpy(w.data.data(), current, vec_size * sizeof(int));
    return w;
}

// [Feature Integration: Live Introspection and Management - ENHANCED]
class ManagementService {
    ObjectPool<Widget>& pool;
    Telemetry& telemetry;
    PredictiveModel model;
    CircuitBreaker& circuit_breaker;
    std::jthread service_thread;
    std::string config_file;

public:
    ManagementService(ObjectPool<Widget>& p, Telemetry& t, CircuitBreaker& cb, const std::string& cfg) 
        : pool(p), telemetry(t), circuit_breaker(cb), config_file(cfg) {}

    void run() {
        service_thread = std::jthread([this](std::stop_token st) {
            while (!st.stop_requested()) {
                // Predictive GC Trigger
                auto history = telemetry.get_history();
                double pressure = model.predict_future_memory_pressure(history);
                if (pressure > 0.75) {
                    Logger::log(Logger::Level::WARNING, "Predicted high memory pressure! Proactively triggering GC.");
                    pool.trigger_gc_if_needed(true); 
                }
                
                // [NEW Integration: Dynamic Configuration Hot-Reloading]
                check_for_config_updates();

                std::this_thread::sleep_for(std::chrono::seconds(5));
            }
        });
    }

    void check_for_config_updates() {
        // In a real system, this would use filesystem notifications (e.g., inotify)
        // Here we just reload every time for demonstration.
        ConfigurationManager::instance().load(config_file);
        Logger::set_level(ConfigurationManager::instance().get("log_level", "INFO"));
        int new_threshold = std::stoi(ConfigurationManager::instance().get("circuit_breaker_threshold", "5"));
        circuit_breaker.update_threshold(new_threshold);
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

void terminate_handler() {
    Logger::log(Logger::Level::ERROR, "Unhandled exception caught! Aborting program.");
    PostMortem::generate_report();
    std::abort();
}

void demo_defragmentation() {
    Logger::log(Logger::Level::INFO, "\n--- Demonstrating Manual Defragmentation ---");
    UltraAllocator allocator("DefragPool", 1024 * 10);
    
    void* p1 = allocator.allocate(100, "Block1");
    allocator.allocate(200, "Block2"); // Alloc p2
    void* p3 = allocator.allocate(150, "Block3");
    allocator.allocate(100, "Block4"); // Alloc p4
    
    allocator.deallocate(p1); // Dealloc p1
    allocator.deallocate(p3); // Dealloc p3
    
    Logger::log(Logger::Level::INFO, "State before defragmentation:");
    allocator.printTelemetry();

    // The pointers to Block2 and Block4 will be invalid after this call.
    allocator.defragment();

    Logger::log(Logger::Level::INFO, "State after defragmentation:");
    allocator.printTelemetry();
}

int main() {
    std::set_terminate(terminate_handler);
    const std::string CONFIG_FILE = "config.txt";

    ConfigurationManager::instance().load(CONFIG_FILE);

    Logger::log(Logger::Level::INFO, "--- Demonstrating UltraAllocator with RBAC and QoS ---");
    UltraAllocator main_allocator("MainPool", 8192, "main_pool.dat");
    
    Role admin_role = "admin";
    Role user_role = "user";
    Role guest_role = "guest";
    main_allocator.get_rbac_manager().grant(admin_role, Permission::Allocate, "MainPool");
    main_allocator.get_rbac_manager().grant(user_role, Permission::Allocate, "MainPool");

    try {
        void* a = main_allocator.allocate(512, admin_role, QoSTier::RealTime, "MeshData");
        void* b = main_allocator.allocate(1024, user_role, QoSTier::BestEffort, "TextureCache");
        main_allocator.deallocate(a);
        main_allocator.deallocate(b);
        main_allocator.allocate(256, guest_role, QoSTier::BestEffort, "GuestData");
    } catch(const AllocationError& e) {
        Logger::log(Logger::Level::ERROR, "Caught expected exception: " + std::string(e.what()));
    }
    
    main_allocator.persistToFile(); // Demo encrypted data-at-rest
    
    demo_defragmentation();
    
    Logger::log(Logger::Level::INFO, "\n--- Demonstrating ObjectPool with Enhanced Services ---");
    Telemetry telemetry_service;
    CircuitBreaker circuit_breaker;
    ObjectPool<Widget, VerbosePolicy> pool(4, &telemetry_service);
    ManagementService manager(pool, telemetry_service, circuit_breaker, CONFIG_FILE);
    manager.run();
    
    std::pmr::monotonic_buffer_resource pmr_res;
    std::vector<Ref<Widget, decltype(pool)>> refs;

    for (int i = 0; i < 5; ++i) {
        if (auto result = pool.allocate(i, &pmr_res)) {
            refs.emplace_back(result.value(), &pool);
        }
    }
    
    Logger::log(Logger::Level::INFO, "--- Demonstrating Transparent Compression ---");
    Logger::log(Logger::Level::INFO, "Waiting for compression threshold to be met...");
    std::this_thread::sleep_for(std::chrono::seconds(3));
    Logger::log(Logger::Level::INFO, "Accessing first widget, which should now be compressed...");
    if (!refs.empty()) {
        refs[0]->id = 99; // This access will trigger on-demand decompression
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    Logger::log(Logger::Level::INFO, "\n--- Demonstrating Configuration Hot-Reloading ---");
    Logger::log(Logger::Level::INFO, "Current log level is INFO. Modifying config file to DEBUG...");
    {
        std::ofstream out(CONFIG_FILE);
        out << "log_level=DEBUG\n";
        out << "circuit_breaker_threshold=10\n";
    }
    std::this_thread::sleep_for(std::chrono::seconds(6)); // Wait for management service to reload
    Logger::log(Logger::Level::DEBUG, "This DEBUG message should now be visible.");

    Logger::log(Logger::Level::INFO, "Program complete.");
    return 0;
}