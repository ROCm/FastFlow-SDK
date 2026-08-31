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
    "llvm-aie": compiler_build.cmd_build_llvm_aie,
    "iron-tools": compiler_build.cmd_build_iron_tools,
    "hrx-xclbinutil": compiler_build.cmd_build_hrx_xclbinutil,
    "hrx": runtime_build.cmd_build_hrx,
    "fastflowlm": fastflowlm_build.cmd_build_fastflowlm,
    # Diagnostic-only XRT-backed FastFlowLM build for HRX-vs-XRT benchmark
    # comparisons (qualify --bench --xrt-flm). Never packaged into wheels.
    "fastflowlm-xrt": fastflowlm_build.cmd_build_fastflowlm_xrt,
    "rocm-npu-cli": runtime_build.cmd_build_rocm_npu_cli,
}


def cmd_build_all(args):
    """Build the complete compiler-stack developer toolchain."""
    runtime_build.cmd_build_rocm_npu_cli(args)
    compiler_build.cmd_build_hrx_xclbinutil(args)
    compiler_build.cmd_build_llvm_aie(args)
    compiler_build.cmd_build_llvm(args)
    compiler_build.cmd_build_iron_tools(args)
    runtime_build.cmd_build_hrx(args)


BUILD_HANDLERS["all"] = cmd_build_all
