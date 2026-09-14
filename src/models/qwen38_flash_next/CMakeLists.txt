# Qwen3.8-Flash-Next (qwen4exp) model package. CPU side: GGUF binding,
# tokenizer, n-gram disk reader and the float32 reference graph. The ROCm
# runtime is added below when HIP is enabled.
add_library(gufo_qwen38_flash_next STATIC
  config.cpp
  cpu_ops.cpp
  ngram.cpp
  reference.cpp
  tokenizer.cpp
  weights.cpp)

target_include_directories(gufo_qwen38_flash_next PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_qwen38_flash_next PUBLIC gufo_core Threads::Threads)

# The reference decodes every weight on the fly; unoptimized it is unusable
# even as an oracle, so it keeps -O2 in every configuration.
set_source_files_properties(cpu_ops.cpp reference.cpp
  PROPERTIES COMPILE_OPTIONS "-O2")

if (ENGINE_ENABLE_HIP)
  set(QFN_ROCM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels/rocm")
  set(QFN_MMQ_DIR "${QFN_ROCM_DIR}/mmq")

  # Vendored llama.cpp quantized-matmul tier, the same files the DeepSeek V4
  # Flash package carries, wrapped in namespace qfn_mmq so both copies link
  # into one binary. It keeps its own C++17 dialect and GGML feature macros,
  # so it stays a separate target.
  set(QFN_MMQ_SOURCES
    ${QFN_MMQ_DIR}/qfn_ggml_stubs.hip.cpp
    ${QFN_MMQ_DIR}/qfn_mmq.hip.cpp
    ${QFN_MMQ_DIR}/qfn_mmq_d2r.hip.cpp
    ${QFN_MMQ_DIR}/mmid.hip.cpp
    ${QFN_MMQ_DIR}/mmvq.hip.cpp
    ${QFN_MMQ_DIR}/quantize.hip.cpp)

  add_library(gufo_qwen38_flash_next_mmq STATIC ${QFN_MMQ_SOURCES})
  set_source_files_properties(${QFN_MMQ_SOURCES} PROPERTIES LANGUAGE HIP)
  set_target_properties(gufo_qwen38_flash_next_mmq PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(gufo_qwen38_flash_next_mmq PRIVATE
    GGML_USE_HIP
    GGML_HIP_NO_VMM
    DS4_ROCM_WMMA_MMQ_Y=64)
  target_compile_options(gufo_qwen38_flash_next_mmq PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-value;-Wno-unused-command-line-argument>)
  target_include_directories(gufo_qwen38_flash_next_mmq SYSTEM PUBLIC ${QFN_MMQ_DIR})
  target_link_libraries(gufo_qwen38_flash_next_mmq PUBLIC hip::host roc::hipblas)

  set(QFN_HIP_SOURCES
    ${QFN_ROCM_DIR}/kernels.hip.cpp
    ${QFN_ROCM_DIR}/blaslt.cpp
    ${QFN_ROCM_DIR}/device_model.cpp
    ${QFN_ROCM_DIR}/executor.cpp)
  target_sources(gufo_qwen38_flash_next PRIVATE ${QFN_HIP_SOURCES})
  set_source_files_properties(${QFN_ROCM_DIR}/kernels.hip.cpp PROPERTIES LANGUAGE HIP)
  target_compile_definitions(gufo_qwen38_flash_next PRIVATE QFN_ROCM_BUILD=1)
  target_compile_options(gufo_qwen38_flash_next PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)
  target_include_directories(gufo_qwen38_flash_next SYSTEM PUBLIC
    ${HIPCUB_INCLUDE_DIR}
    ${ROCPRIM_INCLUDE_DIR}
    ${ROCWMMA_INCLUDE_DIR})
  target_link_libraries(gufo_qwen38_flash_next PUBLIC
    gufo_qwen38_flash_next_mmq hip::host roc::hipblas roc::hipblaslt)
endif()

if (ENGINE_ENABLE_HIP)
  target_sources(gufo_qwen38_flash_next PRIVATE engine.cpp)
endif()
