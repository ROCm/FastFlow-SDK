// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Declares platform-specific plugin signal-forwarding verification.

#pragma once

#include "rocm_npu/command.h"

namespace rocm_npu::test {

void VerifySignalForwarding(Invocation& invocation);

}  // namespace rocm_npu::test
