// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Declares platform-specific plugin signal-forwarding verification.

#pragma once

#include "fastflow_sdk/command.h"

namespace fastflow_sdk::test {

void VerifySignalForwarding(Invocation& invocation);

}  // namespace fastflow_sdk::test
