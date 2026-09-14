#include "qfn_mmq_prelude.h"
namespace qfn_mmq {
// SPDX-License-Identifier: MIT
// Implementations of the ggml-API stubs declared in qfn_ggml_stubs.h plus
// the bodies of ggml_backend_cuda_context / ggml_cuda_info /
// new_pool_for_device that the vendored common.cuh declares without
// defining.
//
// Phase 0: pool is plain cudaMallocAsync / cudaFreeAsync. Phase 4 swaps
// in ds4's existing cuda_tmp_alloc slab allocator.

#include "common.cuh"   // pulls in qfn_ggml_stubs.h via redirect headers

#if defined(GGML_USE_HIP)
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#else
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <vector>

// ----------------------------------------------------------------------------
// Device info singleton.
//
// Common.cuh declares `const ggml_cuda_device_info & ggml_cuda_info();` -
// we provide the body. The struct layout (common.cuh:1091) is:
//   { int device_count; cuda_device_info devices[GGML_CUDA_MAX_DEVICES];
//     std::array<float, GGML_CUDA_MAX_DEVICES> default_tensor_split; }
// where cuda_device_info has { cc, nsm, smpb, smpbo, integrated, vmm,
// vmm_granularity, total_vram, warp_size, supports_cooperative_launch }.
// ----------------------------------------------------------------------------

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info;
    static std::once_flag once;
    std::call_once(once, []{
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        if (err != cudaSuccess) {
            fprintf(stderr, "ggml_cuda_info: cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
            count = 0;
        }
        if (count > GGML_CUDA_MAX_DEVICES) count = GGML_CUDA_MAX_DEVICES;
        info.device_count = count;
        for (int i = 0; i < count; i++) {
            cudaDeviceProp p;
            CUDA_CHECK(cudaGetDeviceProperties(&p, i));
#if defined(GGML_USE_HIP)
            unsigned arch = 0;
            if (sscanf(p.gcnArchName, "gfx%x", &arch) != 1) {
                fprintf(stderr, "ggml_cuda_info: unsupported HIP architecture '%s'\n", p.gcnArchName);
            }
            info.devices[i].cc                          = GGML_CUDA_CC_OFFSET_AMD + (int)arch;
#else
            info.devices[i].cc                          = p.major * 100 + p.minor * 10;
#endif
            info.devices[i].nsm                         = p.multiProcessorCount;
            info.devices[i].smpb                        = p.sharedMemPerBlock;
            info.devices[i].smpbo                       = p.sharedMemPerBlockOptin;
            info.devices[i].integrated                  = p.integrated != 0;
            info.devices[i].vmm                         = false;
            info.devices[i].vmm_granularity             = 0;
            info.devices[i].total_vram                  = p.totalGlobalMem;
            info.devices[i].warp_size                   = p.warpSize;
            info.devices[i].supports_cooperative_launch = p.cooperativeLaunch != 0;
        }
    });
    return info;
}

// ggml_cuda_get_device / ggml_cuda_set_device are declared (not defined) in
// common.cuh. We provide thin wrappers.

int ggml_cuda_get_device() {
    int dev = 0;
    cudaGetDevice(&dev);
    return dev;
}

void ggml_cuda_set_device(int device) {
    int cur = -1;
    cudaGetDevice(&cur);
    if (cur != device) {
        CUDA_CHECK(cudaSetDevice(device));
    }
}

int64_t ggml_time_us() {
    using clock = std::chrono::steady_clock;
    static const auto t0 = clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t0).count();
}

// ----------------------------------------------------------------------------
// ggml_cuda_error: invoked by the CUDA_CHECK / CUBLAS_CHECK macros defined
// in common.cuh on the error path. Marked [[noreturn]] in the declaration
// (common.cuh:155) - abort() satisfies that contract.
// ----------------------------------------------------------------------------

[[noreturn]] void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    fprintf(stderr, "CUDA error: %s\n  call: %s\n  in: %s at %s:%d\n", msg, stmt, func, file, line);
    fflush(stderr);
    abort();
}

// ----------------------------------------------------------------------------
// Concrete pool wrapping cudaMallocAsync / cudaFreeAsync.
// ----------------------------------------------------------------------------

namespace {

/* Thread-local stream the tier records for its pool operations (the stack
 * pool below needs none, the API is kept for the entry points that set it) /
 * cudaFreeAsync.  Defaults to cudaStreamPerThread (preserves prior
 * behaviour).  Step 8 / CUDA Graphs sets this to the capture stream
 * before allocating, so the alloc node lives on the captured stream
 * and capture is not invalidated.  Set via qfn_pool_set_stream() from
 * qfn_mmq.cu wrappers; an explicit stream=0 means the legacy default
 * stream so pool ops stay ordered with legacy-stream kernels. */
static thread_local cudaStream_t t_qfn_pool_stream = cudaStreamPerThread;

} // anonymous namespace

extern "C" void qfn_pool_set_stream(cudaStream_t stream) {
    t_qfn_pool_stream = stream;
}

extern "C" cudaStream_t qfn_pool_get_stream(void) {
    return t_qfn_pool_stream;
}

namespace {

/* qwen38: every allocation the tier makes is scoped to one entry point and
 * released in reverse order, and every kernel runs on the executor's single
 * stream, so a stack over one device arena is enough. It replaces
 * hipMallocAsync/hipFreeAsync, whose reuse of a just-freed block raced with
 * in-flight kernels on this ROCm and made prefill logits nondeterministic. */
struct qfn_stack_pool : public ggml_cuda_pool {
    int device;
    char *base = nullptr;
    size_t capacity = 0;
    size_t top = 0;
    std::vector<std::pair<char *, size_t>> live;

    explicit qfn_stack_pool(int device) : device(device) {}

    ~qfn_stack_pool() override {
        if (base) (void)cudaFree(base);
    }

    void reserve(size_t bytes) {
        if (bytes <= capacity) return;
        /* Growing invalidates outstanding pointers, so it only happens while
         * nothing is live; the arena starts generous to keep that true. */
        if (top != 0) {
            fprintf(stderr, "qfn_stack_pool: arena exhausted with live allocations\n");
            abort();
        }
        (void)cudaDeviceSynchronize();
        if (base) (void)cudaFree(base);
        size_t next = capacity ? capacity : (size_t) 256 << 20;
        while (next < bytes) next *= 2;
        CUDA_CHECK(cudaMalloc((void **) &base, next));
        capacity = next;
    }

    void * alloc(size_t size, size_t * actual_size) override {
        ggml_cuda_set_device(device);
        const size_t aligned = (size + 255) & ~(size_t) 255;
        if (top + aligned > capacity) reserve(top + aligned);
        char *ptr = base + top;
        top += aligned;
        live.emplace_back(ptr, aligned);
        if (actual_size) *actual_size = aligned;
        return ptr;
    }

    void free(void * ptr, size_t /*size*/) override {
        if (!ptr) return;
        if (live.empty() || live.back().first != ptr) {
            fprintf(stderr, "qfn_stack_pool: non-LIFO free\n");
            abort();
        }
        top -= live.back().second;
        live.pop_back();
    }
};

} // anonymous namespace

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int device, int /*stream_no*/) {
    return std::unique_ptr<ggml_cuda_pool>(new qfn_stack_pool(device));
}

ggml_backend_cuda_context::~ggml_backend_cuda_context() {
    if (copy_event) {
        cudaEventDestroy(copy_event);
        copy_event = nullptr;
    }
    // streams[][], cublas_handles[], and pools[][] are owned-by-value
    // (cudaStream_t and cublasHandle_t are opaque handles - destroying the
    // context "should" tear them down, but in our shim ds4 manages streams
    // externally and we leave them alone. The pools auto-destruct via
    // unique_ptr.).
}

}  // namespace qfn_mmq
