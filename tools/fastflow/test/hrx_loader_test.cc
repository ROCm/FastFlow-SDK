// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include <algorithm>
#include <cassert>
#include <filesystem>
#include <string>

#include "platform.h"
#include "fastflow_sdk/hrx_loader.h"

int main(int argc, char** argv) {
  assert(argc == 4);
  const std::filesystem::path missing = argv[1];
  const std::filesystem::path incompatible = argv[2];
  const std::filesystem::path complete = argv[3];
  fastflow_sdk::test::ClearEnvironment("HRX_LIBHRX");
  fastflow_sdk::HrxLoadResult load = fastflow_sdk::HrxLibrary::Load(missing);
  assert(!load.library);
  assert(!load.errors.empty());
  assert(!load.missing_symbols.empty());
  assert(std::find(load.missing_symbols.begin(), load.missing_symbols.end(),
                   "hrx_gpu_initialize") != load.missing_symbols.end());
  assert(std::find(load.missing_symbols.begin(), load.missing_symbols.end(),
                   "hrx_status_code") == load.missing_symbols.end());

  const auto candidates =
      fastflow_sdk::HrxLibrary::Candidates(missing.parent_path());
  assert(!candidates.empty());
  assert(candidates.front().parent_path() == missing.parent_path());

  fastflow_sdk::HrxLoadResult abi =
      fastflow_sdk::HrxLibrary::Load(incompatible);
  assert(!abi.library);
  assert(abi.errors.size() == 1);
  assert(abi.errors.front().find("incompatible HRX ABI") != std::string::npos);

  const std::filesystem::path unicode_directory =
      fastflow_sdk::test::TestRoot("hrx");
  std::filesystem::remove_all(unicode_directory);
  std::filesystem::create_directories(unicode_directory);
  const std::filesystem::path environment_library =
      unicode_directory / complete.filename();
  std::filesystem::copy_file(complete, environment_library);
  fastflow_sdk::test::SetEnvironmentPath("HRX_LIBHRX", environment_library);
  fastflow_sdk::HrxLoadResult fallback =
      fastflow_sdk::HrxLibrary::Load(std::nullopt);
  assert(fallback.library);
  assert(fallback.library->path() == environment_library);
  assert(fallback.missing_symbols.empty());
  assert(fallback.errors.empty());
  assert(fallback.library->runtime_version_major() == HRX_VERSION_MAJOR);
  assert(fallback.library->runtime_version_minor() == HRX_VERSION_MINOR);
  fallback.library.reset();
  std::filesystem::remove_all(unicode_directory);
  return 0;
}
