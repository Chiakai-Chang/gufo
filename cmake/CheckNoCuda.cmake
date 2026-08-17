# cmake/CheckNoCuda.cmake
# Configures anti-CUDA boundary scanning targets and CTest tests.

find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(CHECK_NO_CUDA_SCRIPT "${CMAKE_SOURCE_DIR}/tools/check-no-cuda.py")
set(NO_CUDA_FIXTURES_DIR "${CMAKE_SOURCE_DIR}/tests/static/fixtures")
set(NO_CUDA_REPORT_FILE "${CMAKE_BINARY_DIR}/no-cuda-report.json")

# Custom target: check-no-cuda
add_custom_target(check-no-cuda
  COMMAND ${Python3_EXECUTABLE} ${CHECK_NO_CUDA_SCRIPT}
    --source "${CMAKE_SOURCE_DIR}/src"
    --source "${CMAKE_SOURCE_DIR}/models"
    --source "${CMAKE_SOURCE_DIR}/CMakeLists.txt"
    $<$<BOOL:${CMAKE_EXPORT_COMPILE_COMMANDS}>:--compile-commands>
    $<$<BOOL:${CMAKE_EXPORT_COMPILE_COMMANDS}>:${CMAKE_BINARY_DIR}/compile_commands.json>
    --binary "$<TARGET_FILE:strix>"
    --binary "$<TARGET_FILE:strix-server>"
    --json-report "${NO_CUDA_REPORT_FILE}"
  DEPENDS strix strix-server
  COMMENT "Scanning source files, compile commands, and binaries for forbidden CUDA dependencies"
  VERBATIM
)

# Register CTest tests if testing is enabled
if (BUILD_TESTING)
  # Test 1: Self-test against positive and negative fixtures
  add_test(
    NAME check-no-cuda-fixtures
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_SOURCE_DIR}/tests/static/test_check_no_cuda.py"
  )
  set_tests_properties(check-no-cuda-fixtures PROPERTIES LABELS "cpu;static;no-cuda")

  # Test 2: Source tree scan
  add_test(
    NAME check-no-cuda-source
    COMMAND ${Python3_EXECUTABLE} ${CHECK_NO_CUDA_SCRIPT}
      --source "${CMAKE_SOURCE_DIR}/src"
      --source "${CMAKE_SOURCE_DIR}/models"
      --source "${CMAKE_SOURCE_DIR}/CMakeLists.txt"
      --json-report "${NO_CUDA_REPORT_FILE}"
  )
  set_tests_properties(check-no-cuda-source PROPERTIES LABELS "cpu;static;no-cuda")

  # Test 3: Binary dynamic dependency and symbol scan
  add_test(
    NAME check-no-cuda-binaries
    COMMAND ${Python3_EXECUTABLE} ${CHECK_NO_CUDA_SCRIPT}
      --binary "$<TARGET_FILE:strix>"
      --binary "$<TARGET_FILE:strix-server>"
  )
  set_tests_properties(check-no-cuda-binaries PROPERTIES LABELS "cpu;static;no-cuda")
endif()
