// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Declares the platform-neutral ROCm NPU CLI entry routine.

#pragma once

#include <string>
#include <vector>

namespace rocm_npu {

int RunMain(std::vector<std::string> arguments);

}  // namespace rocm_npu
