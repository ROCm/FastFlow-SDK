#include <algorithm>
#include <cassert>
#include <filesystem>
#include <string>

#include "platform.h"
#include "rocm_npu/hrx_loader.h"

int main(int argc, char** argv) {
  assert(argc == 4);
  const std::filesystem::path missing = argv[1];
  const std::filesystem::path incompatible = argv[2];
  const std::filesystem::path complete = argv[3];
  rocm_npu::test::ClearEnvironment("HRX_LIBHRX");
  rocm_npu::HrxLoadResult load = rocm_npu::HrxLibrary::Load(missing);
  assert(!load.library);
  assert(!load.errors.empty());
  assert(!load.missing_symbols.empty());
  assert(std::find(load.missing_symbols.begin(), load.missing_symbols.end(),
                   "hrx_gpu_initialize") != load.missing_symbols.end());
  assert(std::find(load.missing_symbols.begin(), load.missing_symbols.end(),
                   "hrx_status_code") == load.missing_symbols.end());

  const auto candidates =
      rocm_npu::HrxLibrary::Candidates(missing.parent_path());
  assert(!candidates.empty());
  assert(candidates.front().parent_path() == missing.parent_path());

  rocm_npu::HrxLoadResult abi =
      rocm_npu::HrxLibrary::Load(incompatible);
  assert(!abi.library);
  assert(abi.errors.size() == 1);
  assert(abi.errors.front().find("incompatible HRX ABI") !=
         std::string::npos);

  const std::filesystem::path unicode_directory =
      rocm_npu::test::TestRoot("hrx");
  std::filesystem::remove_all(unicode_directory);
  std::filesystem::create_directories(unicode_directory);
  const std::filesystem::path environment_library =
      unicode_directory / complete.filename();
  std::filesystem::copy_file(complete, environment_library);
  rocm_npu::test::SetEnvironmentPath("HRX_LIBHRX", environment_library);
  rocm_npu::HrxLoadResult fallback = rocm_npu::HrxLibrary::Load(std::nullopt);
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
