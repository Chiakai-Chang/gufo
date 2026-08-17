// Negative fixture: Forbidden CUDA API symbol
#include <cstddef>

extern "C" int cudaMalloc(void** devPtr, size_t size);

int test_cuda() {
    void* ptr = nullptr;
    return cudaMalloc(&ptr, 1024);
}
