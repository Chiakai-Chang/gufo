# Gufo on native Windows

This fork builds and serves Gufo on Windows 11 with no WSL. The target is AMD Strix Halo (`gfx1151`).
It was validated on a Ryzen AI MAX+ 395 with the BIOS carve at 64 GB dedicated / 64 GB system, serving Qwen3.8-Flash-Next UD-Q4_K_XL and Qwen3.8-27B.

## Requirements

- Windows 11 with a current AMD Adrenalin driver.
- The TheRock ROCm SDK for Windows. It was tested with ROCm 10.1 (clang 24) at `C:\rocm-sdk\rocm`. It must include hipBLAS, hipBLASLt, rocBLAS, hipCUB, rocPRIM and rocWMMA.
- Visual Studio 2022 or newer with the C++ x64 tools. Its bundled CMake and Ninja are used, and `vcvars64.bat` provides the MSVC/SDK libraries.
- vcpkg (standalone clone), with:

```powershell
git clone https://github.com/microsoft/vcpkg C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat -disableMetrics
C:\vcpkg\vcpkg install icu curl openssl libpng libjpeg-turbo --triplet x64-windows
```

## Build

```powershell
.\build-win.ps1                    # Visual Studio found with vswhere
.\build-win.ps1 -Rocm D:/rocm -Vcpkg D:/vcpkg -VisualStudio "C:\Program Files\Microsoft Visual Studio\2022\Community"
```

Output: `build\win-release\gufo.exe`. At run time, put `<rocm>\bin` and `<vcpkg>\installed\x64-windows\bin` on `PATH`.

## Run

```bat
set "PATH=C:\rocm-sdk\rocm\bin;C:\vcpkg\installed\x64-windows\bin;%PATH%"
set "GUFO_ACCEPT_ANY_MODEL=1"
gufo.exe serve llm -m <first-shard>.gguf --mmproj <mmproj>.gguf ^
  --speculative mtp --mtp-model <mtp>.gguf ^
  --chat-template-file <template>.jinja ^
  -c 262144 -n 16384 -j 2 -i 0.0.0.0 -p 8080
```

## What the port changes

- **`compat/win/`**: wrapper headers for the POSIX headers Gufo includes, plus `gufo_posix.cpp`. It covers:
  - `mmap` (file mappings; offsets aligned to the 64 KiB allocation granularity);
  - `pread`/`pwrite` (overlapped I/O);
  - `O_DIRECT` (emulated with `FILE_FLAG_NO_BUFFERING`);
  - `sysconf`, `flock`, `mkstemp` and `getrusage`;
  - Winsock `poll`/`close`.
  Process spawning and `*at()` calls are stubs.
- **Memory accounting**: `hipMemGetInfo` on Windows reports only the dedicated segment. Serving capacity checks also count the DXGI non-local (shared) budget, because Flash-Next Q4 spills past a 64 GB carve.
- **Flash-Next PLE table**:
  - On Linux, unbuffered row reads hide behind compute. On Windows they do not, and prefill of real text was I/O bound.
  - The table is instead memory-mapped and prefetched in the background, with pages left on the standby list so they count as available memory.
  - `GUFO_PLE_MODE=buffered|direct` selects the other readers.
- **Qwen27B weights** are copied to device memory (`GUFO_WEIGHTS=host` keeps the Linux zero-copy registration).
- **`--chat-template-file`** renders Qwen prompts with the llama.cpp Jinja engine (vendored in `third_party/llama_jinja`, MIT).
  - Rendered through this path, the embedded official template is byte-identical to the built-in formatter.
  - `chat_template_kwargs` pass through to the template.
  - `GUFO_DUMP_PROMPT=FILE` appends each rendered prompt, for debugging.
- **Vision**: finetuned `mmproj` files without a `general.basename` are accepted.
- **`GUFO_ACCEPT_ANY_MODEL=1`**: requests may carry any `model` value, or none, as with llama-server.
- **llama-server/vLLM client compatibility**:
  - `/props` answers without `?model=` and reports `default_generation_settings.n_ctx`, `total_slots` and `modalities`;
  - `/v1/models` carries `context_length` / `max_model_len` / `meta.n_ctx`;
  - chat requests accept `stop` (streaming holds back a possible stop prefix and ends generation at the match), `logprobs`, `top_logprobs`, `response_format` and `modalities: ["text"]`. Fields Gufo cannot honour exactly are logged as `ignored=`.
  - Agent clients such as Hermes probe these and use `response_format` for title generation.

## Not supported on Windows yet

`--cache-disk`, FFmpeg-based media and video generation, and the DeepSeek V4 Flash snapshot path (`fmemopen`).

## Measured on this box (same-batch A/B)

These are agent-shaped sessions of 6 turns, about 3.4K new prompt tokens and up to 400 generated tokens per turn.

| Flash-Next setup | normalized session s | PP tok/s | TG tok/s |
| --- | ---: | ---: | ---: |
| Gufo, UD-Q4_K_XL + MTP | ~102-104 | ~830-990 | ~26-30 |
| llama.cpp (ROCm fork), IQ2 pack + MTP | ~120 | ~620 | ~26-28 |

- A single 58.7K-token prompt runs at PP 958 / TG 25.1.
- For the 27B dense model, Gufo prefills faster but llama.cpp's built-in MTP decodes faster, so no winner is claimed there.
