// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Defines portable filesystem, environment, and process helpers for CLI tests.

#pragma once

#include <filesystem>
#include <string>

namespace fastflow_sdk::test {

std::filesystem::path TestRoot(const std::string& name);
std::filesystem::path PluginExecutable(const std::filesystem::path& directory,
                                       const std::string& command);
void SetEnvironmentPath(const std::string& name,
                        const std::filesystem::path& value);
void ClearEnvironment(const std::string& name);
void MakeExecutable(const std::filesystem::path& path);
bool SupportsIsolatedEmptyPluginPath();
[[noreturn]] void WaitForSignal();

}  // namespace fastflow_sdk::test
