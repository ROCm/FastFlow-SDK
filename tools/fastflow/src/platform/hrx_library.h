// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace fastflow_sdk {

std::optional<std::filesystem::path> HrxEnvironmentPath();
const char* HrxLibraryFilename();
std::vector<std::filesystem::path> PackagedHrxLibraryCandidates(
    const std::filesystem::path& executable_directory);
std::vector<std::filesystem::path> HrxLibraryFallbackCandidates();
void* OpenHrxLibrary(const std::filesystem::path& path, std::string& error);
void CloseHrxLibrary(void* handle);
void* LookupHrxSymbol(void* handle, const char* name);

}  // namespace fastflow_sdk
