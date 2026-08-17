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
  find_package(rocblas REQUIRED)

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

add_executable(strix-server src/server/main.cpp)
target_link_libraries(strix-server PRIVATE strix_diagnostics)

install(TARGETS strix strix-server DESTINATION bin)

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

  add_executable(diagnose_cli_test tests/diagnostics/diagnose_cli_test.cpp)
  target_link_libraries(diagnose_cli_test PRIVATE strix_diagnostics)

  add_test(NAME diagnose_cli_test COMMAND diagnose_cli_test)
  set_tests_properties(diagnose_cli_test PROPERTIES LABELS "cpu;diagnostics;cli")

  add_executable(system_inventory_test tests/diagnostics/system_inventory_test.cpp)
  target_compile_definitions(system_inventory_test PRIVATE TEST_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/diagnostics/sysfs")
  target_link_libraries(system_inventory_test PRIVATE strix_diagnostics)

  add_test(NAME system_inventory_test COMMAND system_inventory_test)
  set_tests_properties(system_inventory_test PROPERTIES LABELS "cpu;diagnostics;inventory;compatibility")

  add_executable(fingerprint_test tests/diagnostics/fingerprint_test.cpp)
  target_compile_definitions(fingerprint_test PRIVATE TEST_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/diagnostics")
  target_link_libraries(fingerprint_test PRIVATE strix_diagnostics)

  add_test(NAME fingerprint_test COMMAND fingerprint_test)
  set_tests_properties(fingerprint_test PROPERTIES LABELS "cpu;diagnostics;fingerprint;artifact_validator")

  add_executable(bandwidth_test tests/diagnostics/bandwidth_test.cpp)
  target_compile_definitions(bandwidth_test PRIVATE TEST_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/diagnostics")
  target_link_libraries(bandwidth_test PRIVATE strix_diagnostics)

  add_test(NAME bandwidth_test COMMAND bandwidth_test)
  set_tests_properties(bandwidth_test PROPERTIES LABELS "cpu;diagnostics;bandwidth;bandwidth_schema;bandwidth_correctness")

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

# Anti-CUDA boundary checks and targets
include(cmake/CheckNoCuda.cmake)

# Static analysis and formatting targets
include(cmake/StaticAnalysis.cmake)


