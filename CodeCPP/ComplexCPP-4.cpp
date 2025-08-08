// UltraAllocator.h and .cpp (Single File)
#include <vector>
#include <iostream>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <string>

class UltraAllocator {
public:
    UltraAllocator(size_t poolSize, size_t slabBlockSize = 64, size_t slabBlockCount = 128)
        : slabBlockSize(slabBlockSize), slabBlockCount(slabBlockCount) {
        size_t slabSize = slabBlockSize * slabBlockCount;
        memoryPool.resize(poolSize);
        slabFreeList.resize(slabBlockCount);
        for (auto& flag : slabFreeList) flag.store(true);

        blocks.push_back({slabSize, poolSize - slabSize, true, ""});
    }

    void* Allocate(size_t size, const std::string& tag = "") {
        if (size <= slabBlockSize) {
            return AllocateSlab(tag);
        }

        std::lock_guard<std::mutex> lock(allocatorMutex);

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

    void Free(void* ptr) {
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

        std::cerr << "Free failed: pointer not found\n";
    }

    void CompactMemory() {
        std::lock_guard<std::mutex> lock(allocatorMutex);

        std::vector<Block> newBlocks;
        size_t newOffset = slabBlockSize * slabBlockCount;
        std::vector<uint8_t> newPool(memoryPool.size());
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

        if (newOffset < memoryPool.size()) {
            newBlocks.push_back({newOffset, memoryPool.size() - newOffset, true, ""});
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

    void PrintFragmentation() const {
        std::lock_guard<std::mutex> lock(allocatorMutex);
        std::cout << "Memory Fragmentation:\n";
        for (const auto& block : blocks) {
            std::cout << "Offset: " << block.offset
                      << ", Size: " << block.size
                      << ", Free: " << (block.free ? "Yes" : "No")
                      << ", Tag: " << block.tag << "\n";
        }
    }

    float GetFragmentationRatio() const {
        std::lock_guard<std::mutex> lock(allocatorMutex);
        size_t totalFree = 0;
        size_t largestFree = 0;

        for (const auto& block : blocks) {
            if (block.free) {
                totalFree += block.size;
                if (block.size > largestFree) largestFree = block.size;
            }
        }

        if (totalFree == 0) return 0.0f;
        return 1.0f - (float)largestFree / totalFree;
    }

    void ReportLeaks() const {
        std::lock_guard<std::mutex> lock(allocatorMutex);
        std::cout << "\n🔍 Leak Report:\n";
        for (const auto& [ptr, size] : activeAllocations) {
            std::cout << "Leaked Allocation at " << ptr << ", Size: " << size << "\n";
        }
    }

    void SyncWithGPU() {
        // Placeholder for GPU sync logic
        std::cout << "\n🎮 Syncing with GPU buffers... (stub)\n";
        // In real use: map/unmap buffers, flush caches, etc.
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
};

// 🧪 Example Usage
int main() {
    UltraAllocator allocator(2048, 64, 8); // 2KB pool, 8 slab blocks of 64B

    void* a = allocator.Allocate(32, "TempBuffer");   // Slab
    void* b = allocator.Allocate(128, "MeshData");    // Best-fit
    void* c = allocator.Allocate(256, "TextureCache"); // Best-fit

    allocator.Free(b);
    allocator.PrintFragmentation();
    std::cout << "Fragmentation Ratio: " << allocator.GetFragmentationRatio() << "\n";

    allocator.SyncWithGPU();

    std::cout << "\n🧹 Compacting memory...\n";
    allocator.CompactMemory();
    allocator.PrintFragmentation();

    allocator.ReportLeaks(); // a and c still allocated

    allocator.Free(a);
    allocator.Free(c);
    allocator.ReportLeaks(); // should be empty

    return 0;
}