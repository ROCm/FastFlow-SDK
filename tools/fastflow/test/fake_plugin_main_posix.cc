// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Adapts POSIX command-line arguments for the dispatch-test plugin.

#include <string>
#include <vector>

#include "fake_plugin.h"

int main(int argc, char** argv) {
  std::vector<std::string> arguments;
  for (int index = 1; index < argc; ++index) {
    arguments.emplace_back(argv[index]);
  }
  return FakePluginMain(arguments);
}
