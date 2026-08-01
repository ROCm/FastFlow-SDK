// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "kernel_capabilities.hpp"

#include <fstream>
#include <set>
#include <stdexcept>

#include "nlohmann/json.hpp"

namespace {

using json = nlohmann::json;

float positive_float(const json& value, const char* label) {
  const float result = value.get<float>();
  if (!(result > 0.0f)) {
    throw std::runtime_error(
        std::string("kernel capability must be positive: ") + label);
  }
  return result;
}

std::string nonempty_string(const json& value, const char* label) {
  const std::string result = value.get<std::string>();
  if (result.empty()) {
    throw std::runtime_error(
        std::string("kernel capability must be nonempty: ") + label);
  }
  return result;
}

std::filesystem::path relative_path(
    const json& value, const char* label) {
  const std::filesystem::path result = nonempty_string(value, label);
  if (result.is_absolute()) {
    throw std::runtime_error(
        std::string("kernel capability path must be relative: ") + label);
  }
  for (const auto& component : result) {
    if (component == "..") {
      throw std::runtime_error(
          std::string("kernel capability path escapes its root: ") + label);
    }
  }
  return result;
}

NibbleSemantics parse_nibble(const std::string& value) {
  if (value == "unsigned") return NibbleSemantics::kUnsigned;
  if (value == "signed-twos-complement") {
    return NibbleSemantics::kSignedTwosComplement;
  }
  throw std::runtime_error("unsupported nibble semantics: " + value);
}

ZeroPointSemantics parse_zero_point(const std::string& value) {
  if (value == "add-after-scale") {
    return ZeroPointSemantics::kAddAfterScale;
  }
  if (value == "ignored") return ZeroPointSemantics::kIgnored;
  throw std::runtime_error("unsupported zero-point semantics: " + value);
}

}  // namespace

KernelCapabilities load_kernel_capabilities(
    const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error(
        "cannot open kernel capability manifest: " + path.string());
  }
  json document;
  input >> document;
  if (document.at("schema_version").get<int>() != 1 ||
      document.at("kind").get<std::string>() !=
          "rocm-npu-staging-fastflowlm-kernel-test-capabilities" ||
      document.at("provenance")
          .at("public_fastflowlm_abi")
          .get<bool>()) {
    throw std::runtime_error(
        "invalid kernel capability manifest identity or provenance");
  }
  const auto& contracts = document.at("contracts");
  if (contracts.at("dequant_output_layout").get<std::string>() !=
          "observed-q4nx-mm-bit-permutation-v1" ||
      contracts.at("gemm_activation_layout").get<std::string>() !=
          "observed-k1024-n128-bit-permutation-v1" ||
      contracts.at("gemm_output_layout").get<std::string>() !=
          "row-major-m1024-n128" ||
      contracts.at("bindings").get<std::string>() !=
          "generic-output-weight-activation-v1") {
    throw std::runtime_error(
        "kernel capability manifest requests an unsupported test contract");
  }

  const auto& shape = document.at("shape");
  const auto& policy = document.at("test_policy");
  KernelCapabilities result{
      nonempty_string(document.at("contract_id"), "contract_id"),
      shape.at("m").get<size_t>(),
      shape.at("k").get<size_t>(),
      shape.at("n").get<size_t>(),
      positive_float(policy.at("relative_tolerance"), "relative_tolerance"),
      positive_float(
          policy.at("structured_absolute_tolerance"),
          "structured_absolute_tolerance"),
      {}};
  if (result.m != 1024 || result.k != 1024 || result.n != 128) {
    throw std::runtime_error(
        "kernel capability manifest requests an unsupported shape");
  }

  std::set<std::string> family_names;
  for (const auto& family : document.at("families")) {
    const auto& xclbins = family.at("xclbins");
    const auto& groups = family.at("implementation_groups");
    const auto& gemm = family.at("gemm_policy");
    FamilyCapability parsed{
        nonempty_string(family.at("name"), "family.name"),
        relative_path(
            family.at("xclbin_directory"), "family.xclbin_directory"),
        relative_path(xclbins.at("dequant"), "xclbins.dequant"),
        relative_path(xclbins.at("mm"), "xclbins.mm"),
        nonempty_string(groups.at("dequant"), "groups.dequant"),
        nonempty_string(groups.at("mm"), "groups.mm"),
        nonempty_string(groups.at("shared_subset"), "groups.shared_subset"),
        parse_nibble(family.at("nibble_semantics").get<std::string>()),
        parse_zero_point(
            family.at("zero_point_semantics").get<std::string>()),
        {
            positive_float(
                gemm.at("full_nibble_absolute_tolerance"),
                "full_nibble_absolute_tolerance"),
            positive_float(
                gemm.at("full_affine_absolute_tolerance"),
                "full_affine_absolute_tolerance"),
        }};
    if (!family_names.insert(parsed.name).second) {
      throw std::runtime_error(
          "duplicate family in kernel capability manifest: " +
          parsed.name);
    }
    result.families.push_back(std::move(parsed));
  }
  if (result.families.empty()) {
    throw std::runtime_error("kernel capability manifest has no families");
  }
  return result;
}
