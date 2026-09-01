// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Implements portable CLI test helpers with POSIX facilities.

#include "platform.h"

#include <cstdlib>
#include <unistd.h>

namespace rocm_npu::test {

std::filesystem::path TestRoot(const std::string& name) {
  return std::filesystem::temp_directory_path() /
         std::filesystem::u8path("rocm-npu-" + name + "-\xc3\xa9-test");
}

std::filesystem::path PluginExecutable(const std::filesystem::path& directory,
                                       const std::string& command) {
  return directory / ("rocm-npu-" + command);
}

void SetEnvironmentPath(const std::string& name,
                        const std::filesystem::path& value) {
  setenv(name.c_str(), value.c_str(), 1);
}

void ClearEnvironment(const std::string& name) { unsetenv(name.c_str()); }

void MakeExecutable(const std::filesystem::path& path) {
  std::filesystem::permissions(path, std::filesystem::perms::owner_exec,
                               std::filesystem::perm_options::add);
}

bool SupportsIsolatedEmptyPluginPath() { return true; }

[[noreturn]] void WaitForSignal() {
  for (;;) {
    pause();
  }
}

}  // namespace rocm_npu::test
