#pragma once

#include <filesystem>
#include <optional>
#include <string>

namespace rocm_npu {

std::optional<std::filesystem::path::string_type> PluginSearchPathOverride();
std::optional<std::filesystem::path::string_type> SystemExecutableSearchPath();
std::filesystem::path::value_type PluginPathListSeparator();
std::filesystem::path::value_type PluginPathListQuote();
bool PluginEmptyPathEntryMeansCurrentDirectory();
std::string PluginDirectoryKey(std::string value);
std::optional<std::string> PluginExecutableName(
    const std::filesystem::path& path);
bool PluginEntryLess(const std::filesystem::path& left,
                     const std::filesystem::path& right);

}  // namespace rocm_npu
