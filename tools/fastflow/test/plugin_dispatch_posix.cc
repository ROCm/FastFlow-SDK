// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// Verifies POSIX signal forwarding from the CLI to a plugin process group.

#include "plugin_dispatch_platform.h"

#include <cassert>
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>

namespace fastflow_sdk::test {

void VerifySignalForwarding(Invocation& invocation) {
  const pid_t notifier = fork();
  assert(notifier >= 0);
  if (notifier == 0) {
    usleep(200000);
    kill(getppid(), SIGTERM);
    _exit(0);
  }
  invocation.arguments = {"--wait-for-signal"};
  const ExternalCommandResult signaled = RunExternalCommand(invocation);
  int notifier_status = 0;
  assert(waitpid(notifier, &notifier_status, 0) == notifier);
  assert(signaled.error.empty());
  assert(signaled.exit_code == 128 + SIGTERM);
}

}  // namespace fastflow_sdk::test
