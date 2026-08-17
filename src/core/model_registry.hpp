#ifndef STRIX_CORE_MODEL_REGISTRY_HPP_
#define STRIX_CORE_MODEL_REGISTRY_HPP_

#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "src/core/model_kind.hpp"

namespace strix::core {

/// Thread-safe, immutable compiled model registry.
/// Dynamic or arbitrary architecture loading is rejected; all supported
/// kinds are compiled into the binary.
class ModelRegistry {
public:
  ModelRegistry() = delete;

  /// Returns all compiled model descriptors registered in the binary.
  [[nodiscard]] static std::span<const ModelDescriptor> GetAllModels() noexcept;

  /// Finds a descriptor by compiled ModelKind enum. Returns nullptr if not
  /// found.
  [[nodiscard]] static const ModelDescriptor* FindByKind(
      ModelKind kind) noexcept;

  /// Finds a descriptor by canonical ID. Returns nullptr if not found.
  [[nodiscard]] static const ModelDescriptor* FindById(
      std::string_view id) noexcept;

  /// Resolves a string ID to a ModelKind enum. Returns std::nullopt if unknown.
  [[nodiscard]] static std::optional<ModelKind> ResolveKind(
      std::string_view id) noexcept;

  /// Finds all model descriptors associated with the specified implementation
  /// family.
  [[nodiscard]] static std::vector<const ModelDescriptor*> FindByFamily(
      ModelFamily family);

  /// Checks if a string ID is registered.
  [[nodiscard]] static bool IsRegistered(std::string_view id) noexcept;

  /// Validates an artifact-claimed model kind against expected compiled kind.
  /// Rejects unknown claims, malformed strings, or mismatched kind IDs.
  [[nodiscard]] static bool ValidateArtifactClaim(
      std::string_view claimed_kind, ModelKind expected_kind) noexcept;
};

/// Free convenience lookup functions
[[nodiscard]] inline const ModelDescriptor* FindModelDescriptor(
    ModelKind kind) noexcept {
  return ModelRegistry::FindByKind(kind);
}

[[nodiscard]] inline const ModelDescriptor* FindModelDescriptor(
    std::string_view id) noexcept {
  return ModelRegistry::FindById(id);
}

[[nodiscard]] inline std::optional<ModelKind> ResolveModelKind(
    std::string_view id) noexcept {
  return ModelRegistry::ResolveKind(id);
}

[[nodiscard]] inline std::span<const ModelDescriptor>
GetRegisteredModels() noexcept {
  return ModelRegistry::GetAllModels();
}

}  // namespace strix::core

#endif  // STRIX_CORE_MODEL_REGISTRY_HPP_
