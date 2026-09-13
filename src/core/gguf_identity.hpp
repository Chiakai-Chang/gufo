#ifndef GUFO_CORE_GGUF_IDENTITY_HPP_
#define GUFO_CORE_GGUF_IDENTITY_HPP_

#include <string>
#include <string_view>

#include "src/core/gguf_reader.hpp"

namespace gufo::core {

/// Scheme tag for GgufSampledIdentityHex. Bump when the sampled stream changes.
inline constexpr std::string_view kGgufSampledIdentityScheme =
    "gguf-sampled-v1";

/// Content-derived identity of a GGUF artifact for cache compatibility.
///
/// Hashes the parsed header, every metadata entry in key order, the full tensor
/// table, and three 4 KiB windows (start, middle, end) of each tensor's payload
/// extent. Any change of layout, quantization, metadata, or whole-tensor data
/// changes the digest; only a sub-tensor same-size edit with identical metadata
/// escapes it. Cost is a few thousand random 4 KiB reads regardless of file
/// size, so startup pays sub-second on any artifact.
///
/// Depends only on file bytes: path, mtime, inode, and copies are irrelevant.
/// Returns 64 lowercase hex characters.
[[nodiscard]] std::string GgufSampledIdentityHex(const GgufReader& reader);

}  // namespace gufo::core

#endif  // GUFO_CORE_GGUF_IDENTITY_HPP_
