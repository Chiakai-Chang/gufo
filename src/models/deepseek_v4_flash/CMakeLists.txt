set(DEEPSEEK_V4_FLASH_VENDOR_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/vendor/antirez")

add_library(strix_deepseek_v4_flash STATIC
  engine.cpp
  ${DEEPSEEK_V4_FLASH_VENDOR_DIR}/ds4.c
  ${DEEPSEEK_V4_FLASH_VENDOR_DIR}/ds4_distributed.c
  ${DEEPSEEK_V4_FLASH_VENDOR_DIR}/ds4_tp.c
  ${DEEPSEEK_V4_FLASH_VENDOR_DIR}/ds4_ssd.c
  ${DEEPSEEK_V4_FLASH_VENDOR_DIR}/ds4_layer_pack.c
  ${DEEPSEEK_V4_FLASH_VENDOR_DIR}/ds4_rocm.hip.cpp
  ${DEEPSEEK_V4_FLASH_VENDOR_DIR}/ds4_rocm_compat.cpp
  ${DEEPSEEK_V4_FLASH_VENDOR_DIR}/ds4_rocm_unavailable.cpp)

set_source_files_properties(
  ${DEEPSEEK_V4_FLASH_VENDOR_DIR}/ds4_rocm.hip.cpp
  PROPERTIES LANGUAGE HIP)

target_compile_definitions(strix_deepseek_v4_flash PRIVATE
  _GNU_SOURCE
  DS4_ROCM_BUILD=1)

target_compile_options(strix_deepseek_v4_flash PRIVATE
  $<$<COMPILE_LANGUAGE:C>:-O3;-ffast-math;-fno-finite-math-only>
  $<$<COMPILE_LANGUAGE:CXX>:-O3;-ffast-math;-fno-finite-math-only>
  $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-command-line-argument>)

target_include_directories(strix_deepseek_v4_flash
  PUBLIC
    ${PROJECT_SOURCE_DIR}
  SYSTEM PUBLIC
    ${DEEPSEEK_V4_FLASH_VENDOR_DIR}
    ${HIPCUB_INCLUDE_DIR}
    ${ROCPRIM_INCLUDE_DIR}
    ${ROCWMMA_INCLUDE_DIR})

target_link_libraries(strix_deepseek_v4_flash PUBLIC
  hip::host
  roc::hipblas
  roc::hipblaslt
  Threads::Threads
  ${CMAKE_DL_LIBS})
