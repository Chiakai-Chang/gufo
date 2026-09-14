# Qwen3.8-Flash-Next checks. Model-backed programs skip (77) without the
# artifact; pure unit tests run everywhere.
add_executable(qwen38_flash_next_reference_probe reference_probe.cpp)
target_link_libraries(qwen38_flash_next_reference_probe PRIVATE
  gufo_qwen38_flash_next)
set_source_files_properties(reference_probe.cpp PROPERTIES COMPILE_OPTIONS "-O2")

add_executable(qwen38_flash_next_ngram_test ngram_test.cpp)
target_link_libraries(qwen38_flash_next_ngram_test PRIVATE
  gufo_qwen38_flash_next)
add_test(NAME qwen38_flash_next.ngram COMMAND qwen38_flash_next_ngram_test)
set_tests_properties(qwen38_flash_next.ngram PROPERTIES
  LABELS "cpu;qwen38_flash_next")

if(ENGINE_ENABLE_HIP)
  add_executable(qwen38_flash_next_gpu_probe gpu_probe.cpp)
  target_link_libraries(qwen38_flash_next_gpu_probe PRIVATE
    gufo_qwen38_flash_next)
  set_source_files_properties(gpu_probe.cpp PROPERTIES COMPILE_OPTIONS "-O2")
endif()
