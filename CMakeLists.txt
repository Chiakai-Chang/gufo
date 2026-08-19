cmake_minimum_required(VERSION 3.21)
project(strix LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
include_directories(${CMAKE_SOURCE_DIR})

# ROCm/HIP GPU backend — compiled natively for the Strix Halo gfx1151 target.
option(ENGINE_ENABLE_HIP "Build strix with ROCm/HIP GPU backend" OFF)

# ROCm/HIP GPU discovery — compiled natively for the Strix Halo gfx1151 target.
# Dependencies are supplied by the Nix environment via $ROCM_PATH.
# Host ambient /usr fallbacks are rejected.
if (NOT DEFINED ENV{ROCM_PATH})
  if (EXISTS /opt/rocm)
    set(ENV{ROCM_PATH} /opt/rocm)
  endif()
endif()

if (DEFINED ENV{ROCM_PATH})
  list(APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH})
  list(APPEND CMAKE_PREFIX_PATH "$ENV{ROCM_PATH}/lib64/cmake")
endif()

# Forward GPU_TARGETS (set by the nix package) to CMAKE_HIP_ARCHITECTURES.
# e.g. -DGPU_TARGETS=gfx1151
if (GPU_TARGETS AND NOT CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES "${GPU_TARGETS}" CACHE STRING "HIP architectures")
endif()

if (ENGINE_ENABLE_HIP)
  enable_language(HIP)
  find_package(hip REQUIRED)
  find_package(hipblas REQUIRED)
  find_package(hipblaslt REQUIRED)
  find_package(rocblas REQUIRED)
  find_path(COMPOSABLE_KERNEL_INCLUDE_DIR ck/ck.hpp REQUIRED)

  message(STATUS "strix: ROCm/HIP enabled, architectures=${CMAKE_HIP_ARCHITECTURES}")
  message(STATUS "strix: ROCM_PATH=$ENV{ROCM_PATH}")
endif()

# XRT NPU shim — AMD XDNA2 userspace runtime over the amdxdna kernel driver.
option(ENGINE_ENABLE_XRT "Build strix with the XRT NPU shim" OFF)

if (ENGINE_ENABLE_XRT)
  # XRT install prefix: $XRT_PATH or the default /opt/xilinx/xrt layout.
  find_path(XRT_INCLUDE_DIR xrt/experimental/xrt_system.h
    PATHS "$ENV{XRT_PATH}/include" "/opt/xilinx/xrt/include"
    NO_DEFAULT_PATH)
  find_library(XRT_COREUTIL_LIB xrt_coreutil
    PATHS "$ENV{XRT_PATH}/lib" "/opt/xilinx/xrt/lib"
    NO_DEFAULT_PATH)

  if (NOT XRT_INCLUDE_DIR OR NOT XRT_COREUTIL_LIB)
    message(FATAL_ERROR "XRT not found (ENGINE_ENABLE_XRT=ON). Please enter the Nix development environment (nix develop)")
  endif()
  message(STATUS "strix: XRT NPU shim enabled, include=${XRT_INCLUDE_DIR}")
endif()

# Compiler warnings and sanitizers
option(STRIX_ENABLE_WARNINGS "Enable compiler warnings" ON)
option(STRIX_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

if (STRIX_ENABLE_WARNINGS)
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
  endif()
endif()

if (STRIX_ENABLE_SANITIZERS)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

add_executable(strix src/main.cpp)

if (ENGINE_ENABLE_HIP)
  target_compile_definitions(strix PRIVATE ENGINE_ENABLE_HIP=1)
  target_link_libraries(strix PRIVATE hip::host roc::hipblas roc::rocblas)
endif()

if (ENGINE_ENABLE_XRT)
  target_compile_definitions(strix PRIVATE ENGINE_ENABLE_XRT=1)
  target_include_directories(strix PRIVATE ${XRT_INCLUDE_DIR})
  target_link_libraries(strix PRIVATE ${XRT_COREUTIL_LIB})
endif()

add_library(strix_diagnostics STATIC
  src/core/diagnostics/linux_sysfs.cpp
  src/core/diagnostics/system_inventory.cpp
  src/core/diagnostics/compatibility.cpp
  src/core/diagnostics/fingerprint.cpp
  src/core/diagnostics/artifact_validator.cpp
  src/core/diagnostics/bandwidth.cpp
  src/core/hip/bandwidth.cpp
  src/core/xdna2/bandwidth.cpp
  src/core/diagnostics/report.cpp
  src/cli/diagnose.cpp
)
target_include_directories(strix_diagnostics PUBLIC ${CMAKE_SOURCE_DIR})

if (ENGINE_ENABLE_HIP)
  target_compile_definitions(strix_diagnostics PRIVATE ENGINE_ENABLE_HIP=1)
  target_link_libraries(strix_diagnostics PRIVATE hip::host roc::hipblas roc::rocblas)
endif()

if (ENGINE_ENABLE_XRT)
  target_compile_definitions(strix_diagnostics PRIVATE ENGINE_ENABLE_XRT=1)
  target_include_directories(strix_diagnostics PRIVATE ${XRT_INCLUDE_DIR})
  target_link_libraries(strix_diagnostics PRIVATE ${XRT_COREUTIL_LIB})
endif()

add_library(strix_core STATIC
  src/core/model_registry.cpp
  src/core/gguf_reader.cpp
  src/tokenization/qwen_tokenizer.cpp
  src/tokenization/qwen_chat_template.cpp
  src/models/qwen_oracles.cpp
  src/models/qwen_state.cpp
  src/models/qwen_ssm.cpp
  src/models/qwen_forward.cpp
  src/models/qwen_generator.cpp
  src/testing/compare/logit_comparator.cpp
  src/eval/regrade.cpp
)
target_include_directories(strix_core PUBLIC ${CMAKE_SOURCE_DIR})

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
  target_link_libraries(strix_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if (ENGINE_ENABLE_HIP)
  target_sources(strix_core PRIVATE
    src/core/hip/qwen_gpu_attention.hip
    src/core/hip/qwen_gpu_batched_ssm.hip
    src/core/hip/qwen_gpu_blas.hip
    src/core/hip/qwen_gpu_decode_ops.hip
    src/core/hip/qwen_gpu_prefill_ops.hip
    src/core/hip/qwen_gpu_arena.cpp
    src/core/hip/qwen_gpu_decode.cpp
    src/core/hip/qwen_gpu_executor.cpp
    src/core/hip/qwen_gpu_model_loader.cpp
    src/core/hip/qwen_gpu_prefill.cpp
    src/core/speculative/draft_heads.cpp
    src/core/speculative/self_speculative.cpp
    src/core/speculative/prompt_lookup_backend.cpp
    src/core/speculative/speculative_verifier.cpp
    src/core/heterogeneous/npu_drafter.cpp
    src/core/quant/ggml_dequant.cpp
  )
  target_compile_definitions(strix_core PUBLIC ENGINE_ENABLE_HIP=1)
  target_include_directories(strix_core SYSTEM PRIVATE
    ${COMPOSABLE_KERNEL_INCLUDE_DIR})
  target_link_libraries(strix_core PUBLIC hip::host roc::hipblas
    roc::hipblaslt roc::rocblas)
endif()

if (ENGINE_ENABLE_XRT)
  target_compile_definitions(strix_core PUBLIC ENGINE_ENABLE_XRT=1)
  target_include_directories(strix_core PUBLIC ${XRT_INCLUDE_DIR})
  target_link_libraries(strix_core PUBLIC ${XRT_COREUTIL_LIB})
endif()

add_executable(strix-server
  src/server/main.cpp
  src/server/bench_cli.cpp
  src/server/prompt_cli.cpp
  src/server/inference_backend.cpp
  src/server/http_server.cpp
)
target_link_libraries(strix-server PRIVATE strix_diagnostics strix_core)

add_executable(strix-bench
  src/bench/main.cpp
  src/server/bench_cli.cpp
)
target_link_libraries(strix-bench PRIVATE strix_diagnostics strix_core)

install(TARGETS strix strix-server strix-bench DESTINATION bin)

# CTest integration
include(CTest)

if (BUILD_TESTING)
  add_test(NAME strix_server_version COMMAND strix-server --version)
  set_tests_properties(strix_server_version PROPERTIES LABELS "cpu")

  add_test(NAME strix_server_help COMMAND strix-server --help)
  set_tests_properties(strix_server_help PROPERTIES LABELS "cpu")

  add_test(NAME strix_server_diagnose_help COMMAND strix-server diagnose --help)
  set_tests_properties(strix_server_diagnose_help PROPERTIES LABELS "cpu")

  add_test(NAME strix_server_diagnose_json COMMAND strix-server diagnose --json --section platform)
  set_tests_properties(strix_server_diagnose_json PROPERTIES LABELS "cpu")

  add_test(NAME strix_server_prompt_help COMMAND strix-server prompt --help)
  set_tests_properties(strix_server_prompt_help PROPERTIES LABELS "cpu;server;cli")

  add_test(NAME strix_server_serve_help COMMAND strix-server serve --help)
  set_tests_properties(strix_server_serve_help PROPERTIES LABELS "cpu;server;cli")

  add_test(NAME strix_server_prompt_smoke COMMAND strix-server prompt "Hello world")
  set_tests_properties(strix_server_prompt_smoke PROPERTIES LABELS "cpu;server;cli")

  add_executable(prompt_cli_test
    tests/server/prompt_cli_test.cpp
    src/server/prompt_cli.cpp
  )
  target_link_libraries(prompt_cli_test PRIVATE strix_core)

  add_test(NAME prompt_cli_test COMMAND prompt_cli_test)
  set_tests_properties(prompt_cli_test PROPERTIES LABELS "cpu;server;cli")

  add_executable(bench_cli_test
    tests/server/bench_cli_test.cpp
    src/server/bench_cli.cpp
  )
  target_link_libraries(bench_cli_test PRIVATE strix_core)

  add_test(NAME bench_cli_test COMMAND bench_cli_test)
  set_tests_properties(bench_cli_test PROPERTIES LABELS "cpu;server;cli")

  add_executable(bandwidth_test tests/diagnostics/bandwidth_test.cpp)
  target_compile_definitions(bandwidth_test PRIVATE TEST_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/diagnostics")
  target_link_libraries(bandwidth_test PRIVATE strix_diagnostics)

  add_test(NAME bandwidth_test COMMAND bandwidth_test)
  set_tests_properties(bandwidth_test PROPERTIES LABELS "cpu;diagnostics;bandwidth;bandwidth_schema;bandwidth_correctness")

  add_executable(gguf_reader_test tests/core/gguf_reader_test.cpp)
  target_link_libraries(gguf_reader_test PRIVATE strix_core)

  add_test(NAME gguf_reader_test COMMAND gguf_reader_test)
  set_tests_properties(gguf_reader_test PROPERTIES LABELS "cpu;core;gguf;model_config")

  add_executable(qwen_tokenizer_test tests/tokenization/qwen_tokenizer_test.cpp)
  target_link_libraries(qwen_tokenizer_test PRIVATE strix_core)

  add_test(NAME qwen_tokenizer_test COMMAND qwen_tokenizer_test)
  set_tests_properties(qwen_tokenizer_test PROPERTIES LABELS "cpu;tokenization;qwen")

  add_executable(qwen_chat_template_test tests/tokenization/qwen_chat_template_test.cpp)
  target_link_libraries(qwen_chat_template_test PRIVATE strix_core)

  add_test(NAME qwen_chat_template_test COMMAND qwen_chat_template_test)
  set_tests_properties(qwen_chat_template_test PROPERTIES LABELS "cpu;tokenization;chat_template")

  add_executable(qwen_cpu_oracles_test tests/models/qwen_cpu_oracles_test.cpp)
  target_link_libraries(qwen_cpu_oracles_test PRIVATE strix_core)

  add_test(NAME qwen_cpu_oracles_test COMMAND qwen_cpu_oracles_test)
  set_tests_properties(qwen_cpu_oracles_test PROPERTIES LABELS "cpu;models;oracles")

  add_executable(qwen_forward_test tests/models/qwen_forward_test.cpp)
  target_link_libraries(qwen_forward_test PRIVATE strix_core)

  add_test(NAME qwen_forward_test COMMAND qwen_forward_test)
  set_tests_properties(qwen_forward_test PROPERTIES LABELS "cpu;models;forward")

  add_executable(qwen_generator_test tests/models/qwen_generator_test.cpp)
  target_link_libraries(qwen_generator_test PRIVATE strix_core)

  add_test(NAME qwen_generator_test COMMAND qwen_generator_test)
  set_tests_properties(qwen_generator_test PROPERTIES LABELS "cpu;models;generator")

  add_executable(qwen_ssm_test tests/models/qwen_ssm_test.cpp)
  target_link_libraries(qwen_ssm_test PRIVATE strix_core)

  add_test(NAME qwen_ssm_test COMMAND qwen_ssm_test)
  set_tests_properties(qwen_ssm_test PROPERTIES LABELS "cpu;models;ssm")

  add_executable(qwen_oracle_test tests/models/qwen_oracle_test.cpp)
  target_link_libraries(qwen_oracle_test PRIVATE strix_core)
  add_test(NAME qwen_oracle_test COMMAND qwen_oracle_test)
  set_tests_properties(qwen_oracle_test PROPERTIES LABELS "cpu;models;oracles")

  add_executable(exact_token_test tests/quality/exact_token_test.cpp)
  target_link_libraries(exact_token_test PRIVATE strix_core)
  add_test(NAME exact_token_test COMMAND exact_token_test)
  set_tests_properties(exact_token_test PROPERTIES LABELS "cpu;quality;tokens")

  add_executable(eval_regrade_test tests/quality/eval_regrade_test.cpp)
  target_link_libraries(eval_regrade_test PRIVATE strix_core)
  add_test(NAME eval_regrade_test COMMAND eval_regrade_test)
  set_tests_properties(eval_regrade_test PROPERTIES LABELS "cpu;quality;eval")

  add_executable(qwen_attention_policy_test
    tests/kernels/qwen_attention_policy_test.cpp)
  target_link_libraries(qwen_attention_policy_test PRIVATE strix_core)
  add_test(NAME qwen_attention_policy_test COMMAND qwen_attention_policy_test)
  set_tests_properties(qwen_attention_policy_test PROPERTIES
    LABELS "cpu;models;attention;dispatch")

  if (ENGINE_ENABLE_HIP)
    add_executable(qwen_gpu_ops_test tests/kernels/qwen_gpu_ops_test.cpp)
    target_link_libraries(qwen_gpu_ops_test PRIVATE strix_core hip::host roc::hipblas roc::rocblas)
    add_test(NAME qwen_gpu_ops_test COMMAND qwen_gpu_ops_test)
    set_tests_properties(qwen_gpu_ops_test PROPERTIES LABELS "hip;gpu;kernels")

    add_executable(gfx1151_acceptance_test tests/benchmarks/gfx1151_acceptance_test.cpp)
    target_link_libraries(gfx1151_acceptance_test PRIVATE strix_core hip::host roc::hipblas roc::rocblas)
    add_test(NAME gfx1151_acceptance_test COMMAND gfx1151_acceptance_test)
    set_tests_properties(gfx1151_acceptance_test PROPERTIES LABELS "hip;gpu;benchmarks")

    add_executable(speculative_verification_test tests/core/test_speculative_verification.cpp)
    target_link_libraries(speculative_verification_test PRIVATE strix_core hip::host roc::hipblas roc::rocblas)
    add_test(NAME speculative_verification_test COMMAND speculative_verification_test)
    set_tests_properties(speculative_verification_test PROPERTIES LABELS "hip;gpu;speculative")

    add_executable(test_draft_heads tests/core/test_draft_heads.cpp)
    target_link_libraries(test_draft_heads PRIVATE strix_core hip::host roc::hipblas roc::rocblas)
    add_test(NAME test_draft_heads COMMAND test_draft_heads)
    set_tests_properties(test_draft_heads PROPERTIES LABELS "hip;gpu;draft_heads;speculative")

    add_executable(test_npu_speculative tests/npu/test_npu_speculative.cpp)
    target_link_libraries(test_npu_speculative PRIVATE strix_core hip::host roc::hipblas roc::rocblas)
    if (ENGINE_ENABLE_XRT)
      target_link_libraries(test_npu_speculative PRIVATE ${XRT_COREUTIL_LIB})
    endif()
    add_test(NAME test_npu_speculative COMMAND test_npu_speculative)
    set_tests_properties(test_npu_speculative PROPERTIES LABELS "npu;xdna2;speculative")

    add_executable(test_prompt_lookup tests/core/test_prompt_lookup.cpp)
    target_link_libraries(test_prompt_lookup PRIVATE strix_core hip::host roc::hipblas roc::rocblas)
    add_test(NAME test_prompt_lookup COMMAND test_prompt_lookup)
    set_tests_properties(test_prompt_lookup PROPERTIES LABELS "speculative;lookup;pld")

  endif()

  if (ENGINE_ENABLE_HIP AND ENGINE_ENABLE_XRT)
    add_test(NAME strix_npu_smoke COMMAND strix)
    set_tests_properties(strix_npu_smoke PROPERTIES LABELS "npu;xdna2;xrt;hip;gpu")
  elseif (ENGINE_ENABLE_HIP)
    add_test(NAME strix_hip_smoke COMMAND strix)
    set_tests_properties(strix_hip_smoke PROPERTIES LABELS "hip;gpu")
  elseif (ENGINE_ENABLE_XRT)
    add_test(NAME strix_npu_smoke COMMAND strix)
    set_tests_properties(strix_npu_smoke PROPERTIES LABELS "npu;xdna2;xrt")
  else()
    add_test(NAME strix_cpu_smoke COMMAND strix)
    set_tests_properties(strix_cpu_smoke PROPERTIES LABELS "cpu")
  endif()
endif()

# Static analysis and formatting targets
include(cmake/StaticAnalysis.cmake)
