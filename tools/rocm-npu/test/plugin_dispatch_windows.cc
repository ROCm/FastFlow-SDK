// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Provides the Windows dispatch-test hook where POSIX signals do not apply.

#include "plugin_dispatch_platform.h"

namespace rocm_npu::test {

void VerifySignalForwarding(Invocation&) {}

}  // namespace rocm_npu::test
