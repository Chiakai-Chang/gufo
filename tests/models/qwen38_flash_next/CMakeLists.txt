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

add_executable(qwen38_flash_next_mtp_sampling_test mtp_sampling_test.cpp)
target_link_libraries(qwen38_flash_next_mtp_sampling_test PRIVATE gufo_core)
add_test(NAME qwen38_flash_next.mtp_sampling
  COMMAND qwen38_flash_next_mtp_sampling_test)
set_tests_properties(qwen38_flash_next.mtp_sampling PROPERTIES
  LABELS "cpu;qwen38_flash_next")

add_custom_target(qwen38_flash_next_tests
  DEPENDS qwen38_flash_next_ngram_test qwen38_flash_next_mtp_sampling_test)

if(ENGINE_ENABLE_HIP)
  add_executable(qwen38_flash_next_blaslt_ops_test blaslt_ops_test.cpp)
  target_link_libraries(qwen38_flash_next_blaslt_ops_test PRIVATE
    gufo_qwen38_flash_next)
  add_test(NAME qwen38_flash_next.blaslt_ops
    COMMAND qwen38_flash_next_blaslt_ops_test)
  set_tests_properties(qwen38_flash_next.blaslt_ops PROPERTIES
    LABELS "hip;gpu;qwen38_flash_next;prefill;gemm")
  add_dependencies(qwen38_flash_next_tests qwen38_flash_next_blaslt_ops_test)

  add_executable(qwen38_flash_next_weight_upload_test weight_upload_test.cpp)
  target_link_libraries(qwen38_flash_next_weight_upload_test PRIVATE
    gufo_qwen38_flash_next)
  add_test(NAME qwen38_flash_next.weight_upload
    COMMAND qwen38_flash_next_weight_upload_test)
  set_tests_properties(qwen38_flash_next.weight_upload PROPERTIES
    LABELS "hip;gpu;qwen38_flash_next")
  add_dependencies(qwen38_flash_next_tests qwen38_flash_next_weight_upload_test)

  add_executable(qwen38_flash_next_session_test session_test.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/continuation_cache.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/continuation_disk_store.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/logging.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/text_model_runner.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/text_generation_scheduler.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/serve/inference_backend.cpp)
  target_link_libraries(qwen38_flash_next_session_test PRIVATE
    gufo_qwen38_flash_next gufo_deepseek_v4_flash hip::host)
  add_executable(qwen38_flash_next_gpu_probe gpu_probe.cpp)
  target_link_libraries(qwen38_flash_next_gpu_probe PRIVATE
    gufo_qwen38_flash_next)
  set_source_files_properties(gpu_probe.cpp PROPERTIES COMPILE_OPTIONS "-O2")

  add_executable(qwen38_flash_next_hc_mix_ops_test hc_mix_ops_test.cpp)
  target_link_libraries(qwen38_flash_next_hc_mix_ops_test PRIVATE
    gufo_qwen38_flash_next)
  add_test(NAME qwen38_flash_next.hc_mix_ops
    COMMAND qwen38_flash_next_hc_mix_ops_test)
  set_tests_properties(qwen38_flash_next.hc_mix_ops PROPERTIES
    LABELS "hip;gpu;gfx1151;qwen38_flash_next;prefill;fusion;opt-qfn-hc-mix-vec4")

  add_executable(qwen38_flash_next_gdn_ops_test gdn_ops_test.cpp)
  target_link_libraries(qwen38_flash_next_gdn_ops_test PRIVATE
    gufo_qwen38_flash_next)
  add_test(NAME qwen38_flash_next.gdn_ops
    COMMAND qwen38_flash_next_gdn_ops_test)
  set_tests_properties(qwen38_flash_next.gdn_ops PROPERTIES
    LABELS "hip;gpu;gfx1151;qwen38_flash_next;prefill;ssm;opt-qfn-gdn-rowsplit")

  add_executable(qwen38_flash_next_attention_ops_test attention_ops_test.cpp)
  target_link_libraries(qwen38_flash_next_attention_ops_test PRIVATE
    gufo_qwen38_flash_next)
  add_test(NAME qwen38_flash_next.attention_ops
    COMMAND qwen38_flash_next_attention_ops_test)
  set_tests_properties(qwen38_flash_next.attention_ops PROPERTIES
    LABELS "hip;gpu;gfx1151;qwen38_flash_next;prefill;attention;opt-qfn-attn-wmma")

  add_executable(qwen38_flash_next_moe_epilogue_ops_test
    moe_epilogue_ops_test.cpp)
  target_link_libraries(qwen38_flash_next_moe_epilogue_ops_test PRIVATE
    gufo_qwen38_flash_next)
  add_test(NAME qwen38_flash_next.moe_epilogue_ops
    COMMAND qwen38_flash_next_moe_epilogue_ops_test)
  set_tests_properties(qwen38_flash_next.moe_epilogue_ops PROPERTIES
    LABELS "hip;gpu;gfx1151;qwen38_flash_next;prefill;moe;opt-qfn-moe-epilogue-vec4")

  add_executable(qwen38_flash_next_moe_ids_ops_test moe_ids_ops_test.cpp)
  target_link_libraries(qwen38_flash_next_moe_ids_ops_test PRIVATE
    gufo_qwen38_flash_next)
  add_test(NAME qwen38_flash_next.moe_ids_ops
    COMMAND qwen38_flash_next_moe_ids_ops_test)
  set_tests_properties(qwen38_flash_next.moe_ids_ops PROPERTIES
    LABELS "hip;gpu;gfx1151;qwen38_flash_next;prefill;moe;opt-qfn-ids-scan")

  add_executable(qwen38_flash_next_w8a8_ops_test w8a8_ops_test.cpp)
  target_link_libraries(qwen38_flash_next_w8a8_ops_test PRIVATE
    gufo_qwen38_flash_next)
  add_test(NAME qwen38_flash_next.w8a8_ops
    COMMAND qwen38_flash_next_w8a8_ops_test)
  set_tests_properties(qwen38_flash_next.w8a8_ops PROPERTIES
    LABELS "hip;gpu;gfx1151;qwen38_flash_next;prefill;gemm;opt-qfn-w8a8")

  add_executable(qwen38_flash_next_routed_wmma_ops_test
    routed_wmma_ops_test.cpp)
  target_link_libraries(qwen38_flash_next_routed_wmma_ops_test PRIVATE
    gufo_qwen38_flash_next)
  add_test(NAME qwen38_flash_next.routed_wmma_ops
    COMMAND qwen38_flash_next_routed_wmma_ops_test)
  set_tests_properties(qwen38_flash_next.routed_wmma_ops PROPERTIES
    LABELS "hip;gpu;gfx1151;qwen38_flash_next;prefill;moe;opt-qfn-moe-wmma")

  add_executable(qwen38_flash_next_select_ops_test select_ops_test.cpp)
  target_link_libraries(qwen38_flash_next_select_ops_test PRIVATE
    gufo_qwen38_flash_next)
  add_test(NAME qwen38_flash_next.select_ops
    COMMAND qwen38_flash_next_select_ops_test)
  set_tests_properties(qwen38_flash_next.select_ops PROPERTIES
    LABELS "hip;gpu;gfx1151;qwen38_flash_next;prefill;attention")

  add_dependencies(qwen38_flash_next_tests
    qwen38_flash_next_session_test
    qwen38_flash_next_hc_mix_ops_test
    qwen38_flash_next_gdn_ops_test
    qwen38_flash_next_attention_ops_test
    qwen38_flash_next_moe_epilogue_ops_test
    qwen38_flash_next_moe_ids_ops_test
    qwen38_flash_next_w8a8_ops_test
    qwen38_flash_next_routed_wmma_ops_test
    qwen38_flash_next_select_ops_test)
endif()
