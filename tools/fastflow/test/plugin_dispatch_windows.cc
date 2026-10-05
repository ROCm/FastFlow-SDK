// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Provides the Windows dispatch-test hook where POSIX signals do not apply.

#include "plugin_dispatch_platform.h"

namespace fastflow_sdk::test {

void VerifySignalForwarding(Invocation&) {}

}  // namespace fastflow_sdk::test
