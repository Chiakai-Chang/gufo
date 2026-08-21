set(DEEPSEEK_V4_FLASH_RUNTIME_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/runtime")
set(DEEPSEEK_V4_FLASH_ROCM_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/kernels/rocm")

add_library(strix_deepseek_v4_flash STATIC
  engine.cpp
  ${DEEPSEEK_V4_FLASH_RUNTIME_DIR}/model_data.cpp
  ${DEEPSEEK_V4_FLASH_RUNTIME_DIR}/rocm_graph.cpp
  ${DEEPSEEK_V4_FLASH_RUNTIME_DIR}/sampling.cpp
  ${DEEPSEEK_V4_FLASH_RUNTIME_DIR}/session.cpp
  ${DEEPSEEK_V4_FLASH_RUNTIME_DIR}/tokenizer.cpp
  ${DEEPSEEK_V4_FLASH_ROCM_DIR}/backend.hip.cpp)

set_source_files_properties(
  ${DEEPSEEK_V4_FLASH_ROCM_DIR}/backend.hip.cpp
  PROPERTIES LANGUAGE HIP)

target_compile_definitions(strix_deepseek_v4_flash PRIVATE
  _GNU_SOURCE
  DS4_ROCM_BUILD=1)

target_compile_options(strix_deepseek_v4_flash PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3;-ffast-math;-fno-finite-math-only>
  $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)

target_include_directories(strix_deepseek_v4_flash
  PUBLIC
    ${PROJECT_SOURCE_DIR}
  SYSTEM PUBLIC
    ${DEEPSEEK_V4_FLASH_RUNTIME_DIR}
    ${DEEPSEEK_V4_FLASH_ROCM_DIR}
    ${DEEPSEEK_V4_FLASH_ROCM_DIR}/detail
    ${HIPCUB_INCLUDE_DIR}
    ${ROCPRIM_INCLUDE_DIR}
    ${ROCWMMA_INCLUDE_DIR})

target_link_libraries(strix_deepseek_v4_flash PUBLIC
  hip::host
  roc::hipblas
  roc::hipblaslt
  Threads::Threads
  ${CMAKE_DL_LIBS})
