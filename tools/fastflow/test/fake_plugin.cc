// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Implements a controllable external plugin used by dispatch tests.

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "fake_plugin.h"
#include "platform.h"

namespace {

std::string Environment(const char* name) {
  const char* value = std::getenv(name);
  return value == nullptr ? std::string() : std::string(value);
}

}  // namespace

int FakePluginMain(const std::vector<std::string>& arguments) {
  if (Environment("FASTFLOW_SDK_PLUGIN_API") != "1" ||
      Environment("FASTFLOW_SDK_CORE_VERSION").empty() ||
      Environment("FASTFLOW_SDK_CORE_EXECUTABLE").empty()) {
    std::cerr << "plugin environment is incomplete\n";
    return 99;
  }
  if (std::find(arguments.begin(), arguments.end(), "--verify-arguments") !=
          arguments.end() &&
      (std::find(arguments.begin(), arguments.end(), "space argument") ==
           arguments.end() ||
       std::find(arguments.begin(), arguments.end(), "unicode-\u00e9") ==
           arguments.end() ||
       std::find(arguments.begin(), arguments.end(), "quote\"argument") ==
           arguments.end() ||
       std::find(arguments.begin(), arguments.end(), "trailing\\") ==
           arguments.end())) {
    std::cerr << "plugin arguments were not forwarded exactly\n";
    return 97;
  }
  int exit_code = 0;
  for (const std::string& argument : arguments) {
    if (argument == "--require-json" &&
        Environment("FASTFLOW_SDK_OUTPUT_FORMAT") != "json") {
      std::cerr << "JSON output environment was not forwarded\n";
      return 98;
    }
    if (argument == "--require-runtime" &&
        Environment("FASTFLOW_SDK_RUNTIME_PATH").empty()) {
      std::cerr << "runtime path was not forwarded\n";
      return 96;
    }
    if (argument == "--wait-for-signal") {
      fastflow_sdk::test::WaitForSignal();
    }
    constexpr const char* prefix = "--exit=";
    if (argument.rfind(prefix, 0) == 0) {
      exit_code = std::stoi(argument.substr(7));
    }
    std::cout << "argument=" << argument << '\n';
  }
  std::string input;
  if (std::getline(std::cin, input)) {
    std::cout << "stdin=" << input << '\n';
  }
  std::cerr << "fake-plugin-stderr\n";
  return exit_code;
}
