// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Converts Windows wide arguments for the dispatch-test plugin.

#include <filesystem>
#include <string>
#include <vector>

#include "fake_plugin.h"

int wmain(int argc, wchar_t** argv) {
  std::vector<std::string> arguments;
  for (int index = 1; index < argc; ++index) {
    arguments.push_back(
        std::filesystem::path(std::wstring(argv[index])).u8string());
  }
  return FakePluginMain(arguments);
}
