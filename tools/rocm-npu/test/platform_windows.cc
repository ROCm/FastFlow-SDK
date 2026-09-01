// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Implements portable CLI test helpers with Windows facilities.

#include "platform.h"

#include <windows.h>

#include <string>

namespace rocm_npu::test {
namespace {

std::wstring WidenAscii(const std::string& value) {
  return std::wstring(value.begin(), value.end());
}

}  // namespace

std::filesystem::path TestRoot(const std::string& name) {
  return std::filesystem::temp_directory_path() /
         std::filesystem::u8path("rocm-npu-" + name + "-\xc3\xa9-test");
}

std::filesystem::path PluginExecutable(const std::filesystem::path& directory,
                                       const std::string& command) {
  return directory / WidenAscii("rocm-npu-" + command + ".exe");
}

void SetEnvironmentPath(const std::string& name,
                        const std::filesystem::path& value) {
  _wputenv_s(WidenAscii(name).c_str(), value.c_str());
}

void ClearEnvironment(const std::string& name) {
  _wputenv_s(WidenAscii(name).c_str(), L"");
}

void MakeExecutable(const std::filesystem::path&) {}

bool SupportsIsolatedEmptyPluginPath() { return false; }

[[noreturn]] void WaitForSignal() {
  for (;;) {
    Sleep(INFINITE);
  }
}

}  // namespace rocm_npu::test
