#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <string>

#include "rocm_npu/hrx_loader.h"

int main(int argc, char** argv) {
  assert(argc == 4);
  const std::filesystem::path missing = argv[1];
  const std::filesystem::path incompatible = argv[2];
  const std::filesystem::path complete = argv[3];
#ifdef _WIN32
  _wputenv_s(L"HRX_LIBHRX", L"");
#else
  unsetenv("HRX_LIBHRX");
#endif
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

#ifdef _WIN32
  const std::filesystem::path unicode_directory =
      std::filesystem::temp_directory_path() / L"rocm-npu-hrx-\u00e9";
  std::filesystem::remove_all(unicode_directory);
  std::filesystem::create_directories(unicode_directory);
  const std::filesystem::path environment_library =
      unicode_directory / complete.filename();
  std::filesystem::copy_file(complete, environment_library);
  _wputenv_s(L"HRX_LIBHRX", environment_library.c_str());
#else
  setenv("HRX_LIBHRX", complete.string().c_str(), 1);
  const std::filesystem::path environment_library = complete;
#endif
  rocm_npu::HrxLoadResult fallback = rocm_npu::HrxLibrary::Load(std::nullopt);
  assert(fallback.library);
  assert(fallback.library->path() == environment_library);
  assert(fallback.missing_symbols.empty());
  assert(fallback.errors.empty());
  assert(fallback.library->runtime_version_major() == HRX_VERSION_MAJOR);
  assert(fallback.library->runtime_version_minor() == HRX_VERSION_MINOR);
#ifdef _WIN32
  fallback.library.reset();
  std::filesystem::remove_all(unicode_directory);
#endif
  return 0;
}
