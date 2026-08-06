// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace rocm_npu {

std::vector<std::filesystem::path> ProductManifestCandidates();
std::optional<std::filesystem::path> FindProductManifest();
std::optional<nlohmann::json> ReadJsonFile(const std::filesystem::path& path,
                                           std::string& error);
bool VerifyProductManifestFiles(const std::filesystem::path& manifest_path,
                                const nlohmann::json& manifest,
                                nlohmann::json& details, std::string& error);
std::filesystem::path SelfTestDataRoot();

}  // namespace rocm_npu
