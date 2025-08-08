// UltraAllocator.cpp (C++23)
#include <iostream>
#include <vector>
#include <memory>
#include <cstring>
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <unistd.h>

class UltraAllocator {
public:
    UltraAllocator(size_t poolSize, bool usePinnedMemory = true, bool secureEnclave = false)
        : poolSize(poolSize), usePinnedMemory(usePinnedMemory), secureEnclave(secureEnclave) {
        allocatePool();
    }

    ~UltraAllocator() {
        if (usePinnedMemory && pool) {
            cudaFreeHost(pool);
        } else if (pool) {
            munmap(pool, poolSize);
        }
    }

    void* allocate(size_t size) {
        if (offset + size > poolSize) {
            std::cerr << "Allocation failed: pool exhausted\n";
            return nullptr;
        }
        void* ptr = static_cast<char*>(pool) + offset;
        offset += size;
        return ptr;
    }

    void reset() {
        offset = 0;
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

        float* devicePtr;
        cudaMalloc(&devicePtr, poolSize);
        cudaMemcpy(devicePtr, pool, poolSize, cudaMemcpyHostToDevice);

        dim3 threads(128);
        dim3 blocks((poolSize / sizeof(float) + threads.x - 1) / threads.x);

        squareKernel<<<blocks, threads>>>(devicePtr, poolSize / sizeof(float));
        cudaMemcpy(pool, devicePtr, poolSize, cudaMemcpyDeviceToHost);
        cudaFree(devicePtr);
    }

    void printBuffer(size_t count = 10) {
        float* data = static_cast<float*>(pool);
        for (size_t i = 0; i < count; ++i) {
            std::cout << data[i] << " ";
        }
        std::cout << "\n";
    }

private:
    void allocatePool() {
        if (usePinnedMemory) {
            cudaMallocHost(&pool, poolSize);
        } else {
            pool = mmap(nullptr, poolSize, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (secureEnclave) {
                mprotect(pool, poolSize, PROT_NONE); // Simulate enclave isolation
                mprotect(pool, poolSize, PROT_READ | PROT_WRITE); // Re-enable access
            }
        }
    }

    static __global__ void squareKernel(float* data, size_t count) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < count) {
            data[idx] *= data[idx];
        }
    }

    void* pool = nullptr;
    size_t poolSize;
    size_t offset = 0;
    bool usePinnedMemory;
    bool secureEnclave;
};

// 🧪 Example usage
int main() {
    UltraAllocator allocator(1024 * sizeof(float), true, true);
    allocator.runGpuKernel();
    allocator.printBuffer();

    allocator.protectReadOnly();
    allocator.protectReadWrite();

    void* block = allocator.allocate(128);
    if (block) {
        std::memset(block, 1, 128);
        std::cout << "Allocated 128 bytes from pool\n";
    }

    allocator.reset();
    std::cout << "Allocator reset\n";

    return 0;
}