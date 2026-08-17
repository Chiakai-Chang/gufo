// Positive fixture: Valid modern C++20 + HIP code for AMD Strix Halo
#include <cstdio>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <rocblas/rocblas.h>

int main() {
    int count = 0;
    hipError_t err = hipGetDeviceCount(&count);
    if (err == hipSuccess) {
        std::printf("Found %d HIP devices\n", count);
    }
    void* ptr = nullptr;
    hipMalloc(&ptr, 1024);
    hipFree(ptr);
    return 0;
}
