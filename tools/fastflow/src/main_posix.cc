// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Adapts the POSIX process entry point to the portable CLI routine.

#include <string>
#include <utility>
#include <vector>

#include "main.h"

int main(int argc, char** argv) {
  std::vector<std::string> arguments;
  arguments.reserve(static_cast<std::size_t>(argc > 0 ? argc - 1 : 0));
  for (int index = 1; index < argc; ++index) {
    arguments.emplace_back(argv[index]);
  }
  return fastflow_sdk::RunMain(std::move(arguments));
}
