#include <iostream>
#include <array>
#include <queue>
#include <mutex>
#include <atomic>
#include <string>
#include <memory>
#include <optional>
#include <condition_variable>
#include <expected>
#include <vector>
#include <concepts>
#include <memory_resource>
#include <thread>
#include <coroutine> // For coroutines
#include <generator> // For std::generator (C++23)

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

// ---------------- ObjectPool with Coroutine GC and PMR ----------------
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

    // Awaitable type to suspend a coroutine until work is available.
    struct gc_awaitable {
        ObjectPool* pool;
        bool await_ready() const { return !pool->gc_queue.empty() || pool->stop_gc; }
        void await_suspend(std::coroutine_handle<> handle) {
            std::unique_lock lock(pool->gc_mtx);
            pool->cv.wait(lock, [&] {
                return !pool->gc_queue.empty() || pool->stop_gc;
            });
            handle.resume();
        }
        void await_resume() {}
    };

    // Coroutine for deferred garbage collection.
    std::jthread gc_thread;
    std::coroutine_handle<> gc_handle;

    void start_gc_coroutine() {
        auto gc_task = [this]() -> std::generator<void> {
            while (!stop_gc) {
                co_await gc_awaitable{this};
                while (!gc_queue.empty()) {
                    std::lock_guard lock(gc_mtx);
                    if (gc_queue.empty()) continue; // Check again after lock
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
        // This is a simplified way to start a coroutine on a jthread.
        gc_thread = std::jthread([task = gc_task()] mutable {
            for(auto _ : task) {}
        });
    }

public:
    ObjectPool() {
        start_gc_coroutine();
    }

    ~ObjectPool() {
        stop_gc = true;
        cv.notify_all();
    }

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

    // std::generator for elegant iteration over active objects.
    std::generator<T&> used_objects() {
        for (auto& entry : pool) {
            if (entry.used) {
                co_yield *reinterpret_cast<T*>(entry.data);
            }
        }
    }

    size_t used_count() const {
        size_t count = 0;
        for (const auto& e : pool) {
            if (e.used) ++count;
        }
        return count;
    }

private:
    Entry* get_entry(T* ptr) {
        for (auto& e : pool)
            if (reinterpret_cast<T*>(e.data) == ptr)
                return &e;
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

    Ref(const Ref& other) : ptr(other.ptr), pool(other.pool) {
        if (ptr) pool->add_ref(ptr);
    }

    Ref& operator=(const Ref& other) {
        if (this != &other) {
            if (ptr) pool->release(ptr);
            ptr = other.ptr;
            pool = other.pool;
            if (ptr) pool->add_ref(ptr);
        }
        return *this;
    }

    ~Ref() {
        if (ptr) pool->release(ptr);
    }

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

    threads.clear(); // Joins all threads
    
    print_info("--- Currently live widgets: ---");
    for (auto& widget : pool.used_objects()) {
        print_info("Live Widget ID: " + std::to_string(widget.id));
    }

    print_info("Pool usage: " + std::to_string(pool.used_count()) + " / 8");

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    print_info("Program complete.");
}