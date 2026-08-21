extern "C" int ds4_gpu_add_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *a, const ds4_gpu_tensor *b, uint32_t n) {
    if (!hip_tensor_has_f32(out, n) || !hip_tensor_has_f32(a, n) || !hip_tensor_has_f32(b, n)) return 0;
    if (n == 0u) return 1;
    add_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)a->ptr, (const float *)b->ptr, n);
    return hip_ok(hipGetLastError(), "add launch");
}
