# Qwen3-TTS 12Hz 1.7B model port (CustomVoice, VoiceDesign, and Base) —
# self-contained.
# No cross-folder references: all needed components live under this tree.

set(QWEN3_TTS_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_library(strix_qwen3_tts STATIC
  audio.cpp
  config.cpp
  loader.cpp
  prompt.cpp
  reference_runner.cpp
  tensor.cpp
  tokenizer.cpp
  hip/encoder_support.cpp
  hip/speaker_encoder_runtime.cpp
  hip/speech_encoder_runtime.cpp
  hip/speech_decoder_runtime.cpp
  hip/synthesis_runtime.cpp
  hip/talker_runtime.cpp
)

target_include_directories(strix_qwen3_tts PUBLIC ${PROJECT_SOURCE_DIR})

target_link_libraries(strix_qwen3_tts PUBLIC
  ICU::uc
  Threads::Threads)

if (ENGINE_ENABLE_HIP)
  target_sources(strix_qwen3_tts PRIVATE
    hip/encoder_ops.hip
    hip/speech_decoder_ops.hip
    hip/talker_ops.hip)
  target_compile_definitions(strix_qwen3_tts PRIVATE ENGINE_ENABLE_HIP=1)
  target_link_libraries(strix_qwen3_tts PUBLIC
    hip::host
    roc::hipblas
    roc::rocblas)
endif()
