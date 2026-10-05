// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Declares the platform-neutral FastFlow SDK CLI entry routine.

#pragma once

#include <string>
#include <vector>

namespace fastflow_sdk {

int RunMain(std::vector<std::string> arguments);

}  // namespace fastflow_sdk
