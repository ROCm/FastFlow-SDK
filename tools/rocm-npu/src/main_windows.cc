// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Converts the Windows wide-character entry point for the portable CLI.

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "main.h"

int wmain(int argc, wchar_t** argv) {
  std::vector<std::string> arguments;
  arguments.reserve(static_cast<std::size_t>(argc > 0 ? argc - 1 : 0));
  for (int index = 1; index < argc; ++index) {
    arguments.push_back(
        std::filesystem::path(std::wstring(argv[index])).u8string());
  }
  return rocm_npu::RunMain(std::move(arguments));
}
