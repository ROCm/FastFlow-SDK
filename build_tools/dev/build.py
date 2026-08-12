# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Developer build graph and command registry."""

from build_tools.dev import (
    compiler_build,
    fastflowlm_build,
    runtime_build,
)

BUILD_HANDLERS = {
    "llvm": compiler_build.cmd_build_llvm,
    "iron-tools": compiler_build.cmd_build_iron_tools,
    "hrx-xclbinutil": compiler_build.cmd_build_hrx_xclbinutil,
    "hrx": runtime_build.cmd_build_hrx,
    "hrx-cts": runtime_build.cmd_build_hrx_cts,
    "fastflowlm": fastflowlm_build.cmd_build_fastflowlm,
    "rocm-npu-cli": runtime_build.cmd_build_rocm_npu_cli,
}


def cmd_build_all(args):
    """Build the complete compiler-stack developer toolchain."""
    runtime_build.cmd_build_rocm_npu_cli(args)
    compiler_build.cmd_build_hrx_xclbinutil(args)
    compiler_build.cmd_build_llvm(args)
    compiler_build.cmd_build_iron_tools(args)
    runtime_build.cmd_build_hrx(args)


BUILD_HANDLERS["all"] = cmd_build_all
