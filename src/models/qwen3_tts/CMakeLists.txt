# Qwen3-TTS model port (CustomVoice 12Hz, gfx1151) — self-contained.
# No cross-folder references: all needed components live under this tree.

set(QWEN3_TTS_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_library(strix_qwen3_tts STATIC
  config.cpp
  loader.cpp
  reference_runner.cpp
  tensor.cpp
  tokenizer.cpp
  hip/speech_decoder_runtime.cpp
  hip/talker_runtime.cpp
)

target_include_directories(strix_qwen3_tts PUBLIC ${PROJECT_SOURCE_DIR})

target_link_libraries(strix_qwen3_tts PUBLIC
  ICU::uc
  Threads::Threads)

if (ENGINE_ENABLE_HIP)
  target_sources(strix_qwen3_tts PRIVATE
    hip/speech_decoder_ops.hip
    hip/talker_ops.hip)
  target_compile_definitions(strix_qwen3_tts PRIVATE ENGINE_ENABLE_HIP=1)
  target_link_libraries(strix_qwen3_tts PUBLIC
    hip::host
    roc::hipblas
    roc::rocblas)
endif()
