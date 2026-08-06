// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace rocm_npu {

std::optional<std::filesystem::path::string_type> PluginSearchPathOverride();
std::optional<std::filesystem::path::string_type> SystemExecutableSearchPath();
std::filesystem::path::value_type PluginPathListSeparator();
std::filesystem::path::value_type PluginPathListQuote();
std::string PluginDirectoryKey(std::string value);
std::optional<std::string> PluginExecutableName(
    const std::filesystem::path& path);
bool PluginEntryLess(const std::filesystem::path& left,
                     const std::filesystem::path& right);
int ExecutePluginProcess(
    const std::filesystem::path& executable,
    const std::vector<std::string>& arguments,
    const std::map<std::string, std::string>& environment_overrides,
    std::string& error);

}  // namespace rocm_npu
