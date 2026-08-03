#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""rocm-npu-staging developer command router.

Builds the NPU stack together (everything except llvm-aie, which comes
from the Peano wheel). Mirrors the command-router shape of hrx's dev.py.

  python dev.py doctor [PROFILE]      # quick validation: compiler-stack (default), fastflowlm, or all
                                     #   --deep: replay every patch series in a temporary index
  python dev.py setup [--profile P]   # venv + python deps (incl. pip cmake/ninja) + fetch
                                     #   --iron-ops-tests: + what iron-ops' own tests need
                                     #     (torch, pytest); the NPU job only
  python dev.py fetch                 # apply temp patches + prefetch eudsl (verified) — offline-ready
  python dev.py build llvm            # LLVM+MLIR from projects/llvm-project (amd-llvm pin)
  python dev.py build iron-tools      # aie-opt/aie-translate against build/llvm
  python dev.py build hrx-xclbinutil  # the xclbin packager
  python dev.py build hrx             # the amdxdna HAL (IREE runtime)
  python dev.py build hrx-cts         # amdxdna HAL conformance binaries (need an NPU to run)
  python dev.py build fastflowlm      # HRX amdxdna + FastFlowLM app only
  python dev.py package-fastflowlm    # relocatable .zip/.tar.zst + manifest
  python dev.py package-fastflowlm-wheels # rocm-npu-runtime + rocm-npu-fastflowlm wheels
  python dev.py verify-fastflowlm-wheels  # pip install wheels into a clean venv and smoke
  python dev.py unpack-fastflowlm     # checksum, extract, and verify the bundle
  python dev.py package-compiler-product # relocatable rocm-npu-compiler product artifact
  python dev.py verify-compiler-product  # verify compiler product and relocation smoke
  python dev.py package-compiler-wheels  # compiler meta/core/CPython abi3 wheels
  python dev.py verify-compiler-wheels   # clean pip install, native closure, and e2e
  python dev.py build all             # everything above, in dependency order
  python dev.py test [DESIGN] [-- ...] # e2e build test (no NPU run); Linux + Windows
  python dev.py test-iron-ops [OP]    # build the iron-ops operators' xclbins (no NPU run)
                                     #   Linux + Windows; suite: tests/iron-ops/suite.py
  python dev.py submodules NAME...    # init (and optionally reset/clean) submodules
  python dev.py package-npu-prereqs   # stage + tar HRX and its CTS binaries (no NPU)
  python dev.py package-npu-test-package # prebuild XADX kernels + numeric cases (no NPU)
  python dev.py unpack-npu-prereqs    # extract that archive (codec-agnostic; no NPU)
  python dev.py verify-npu-prereqs    # assert that archive is self-contained + imports (no NPU)
  python dev.py npu-probe             # print NPU driver/fw versions + check libhrx open
  python dev.py test-hrx-cts [SUITE]  # amdxdna HAL conformance suites (needs an NPU)
  python dev.py test-npu-package      # ON-DEVICE execute-only XADX + numeric suite
  python dev.py test-npu [DESIGN]     # ON-DEVICE compile+run developer path (needs compiler)
                                     #   --device npu2|npu1   --chain (opt-in chained testbench)
  python dev.py test-fastflowlm       # pull qwen3:0.6b + run a 1k HRX benchmark
  python dev.py package-fastflowlm-kernels # build checksummed production-xclbin helper
  python dev.py test-fastflowlm-kernels # run Qwen3/LFM2 dequant+MM HRX stress
  python dev.py test-npu-iron-ops [OP] # ON-DEVICE e2e: an iron-ops operator, numerics checked (needs an NPU)
  python dev.py deps                  # print the dependency overview

Build-only: running on the NPU needs the amdxdna driver + an NPU (not here).
"""
import argparse
import sys

from build_tools import (
    compiler_product,
    compiler_wheels,
    fastflowlm_kernel_tests,
    fastflowlm_package,
    npu_prereqs,
    npu_test_package,
)
from build_tools.dev import build as dev_build
from build_tools.dev import doctor as dev_doctor
from build_tools.dev import iron as dev_iron
from build_tools.dev import patches as dev_patches
from build_tools.dev import setup as dev_setup
from build_tools.dev import sources as dev_sources
from build_tools.dev import testing as dev_testing
from build_tools.dev.profiles import doctor_profile_names, setup_profile_names
from build_tools.rocm_npu_wheel_common import PACKAGE_VERSION_ENV


def cmd_fetch(args, submodule_names=None):
    return dev_patches.cmd_fetch(args, submodule_names)

def cmd_doctor(args):
    return dev_doctor.cmd_doctor(args)

def cmd_setup(args):
    return dev_setup.cmd_setup(args)

def cmd_build_llvm(args):
    return dev_build.cmd_build_llvm(args)

def cmd_build_iron_tools(args):
    return dev_build.cmd_build_iron_tools(args)

def cmd_build_hrx_xclbinutil(args):
    return dev_build.cmd_build_hrx_xclbinutil(args)

def cmd_build_hrx(args):
    return dev_build.cmd_build_hrx(args)

def cmd_build_hrx_cts(args):
    return dev_build.cmd_build_hrx_cts(args)

def cmd_build_fastflowlm(args):
    return dev_build.cmd_build_fastflowlm(args)

def cmd_build_rocm_npu_cli(args):
    return dev_build.cmd_build_rocm_npu_cli(args)

def cmd_build_all(args):
    return dev_build.cmd_build_all(args)

def cmd_test(args):
    return dev_testing.cmd_test(args)

def cmd_test_fastflowlm(args):
    return dev_testing.cmd_test_fastflowlm(args)

def cmd_package_fastflowlm_kernels(args):
    return fastflowlm_kernel_tests.package_helper(args)

def cmd_verify_fastflowlm_kernels(args):
    return fastflowlm_kernel_tests.verify_helper(args)

def cmd_test_fastflowlm_kernels(args):
    return fastflowlm_kernel_tests.run_helper(args)

def cmd_test_iron_ops(args):
    return dev_testing.cmd_test_iron_ops(args)

def cmd_smoke(args):
    return dev_testing.cmd_smoke(args)

def cmd_submodules(args):
    return dev_sources.cmd_submodules(args)

def cmd_package_fastflowlm(args):
    return fastflowlm_package.cmd_package_fastflowlm(args)

def cmd_package_fastflowlm_wheels(args):
    return fastflowlm_package.cmd_package_fastflowlm_wheels(args)

def cmd_audit_fastflowlm_wheels(args):
    return fastflowlm_package.cmd_audit_fastflowlm_wheels(args)

def cmd_verify_fastflowlm_wheels(args):
    return fastflowlm_package.cmd_verify_fastflowlm_wheels(args)

def cmd_unpack_fastflowlm(args):
    return fastflowlm_package.cmd_unpack_fastflowlm(args)

def cmd_verify_fastflowlm(args):
    return fastflowlm_package.cmd_verify_fastflowlm(args)

def cmd_package_compiler_product(args):
    return compiler_product.cmd_package_compiler_product(args)

def cmd_unpack_compiler_product(args):
    return compiler_product.cmd_unpack_compiler_product(args)

def cmd_verify_compiler_product(args):
    return compiler_product.cmd_verify_compiler_product(args)

def cmd_package_compiler_wheels(args):
    return compiler_wheels.cmd_package_compiler_wheels(args)

def cmd_audit_compiler_wheels(args):
    return compiler_wheels.cmd_audit_compiler_wheels(args)

def cmd_verify_compiler_wheels(args):
    return compiler_wheels.cmd_verify_compiler_wheels(args)

def cmd_package_npu_prereqs(args):
    return npu_prereqs.cmd_package_npu_prereqs(args)

def cmd_unpack_npu_prereqs(args):
    return npu_prereqs.cmd_unpack_npu_prereqs(args)

def cmd_verify_npu_prereqs(args):
    return npu_prereqs.cmd_verify_npu_prereqs(args)

def cmd_package_npu_test_package(args):
    return npu_test_package.cmd_package_npu_test_package(args)

def cmd_unpack_npu_test_package(args):
    return npu_test_package.cmd_unpack_npu_test_package(args)

def cmd_verify_npu_test_package(args):
    return npu_test_package.cmd_verify_npu_test_package(args)

def cmd_test_npu_package(args):
    return dev_testing.cmd_test_npu_package(args)

def cmd_npu_probe(args):
    return dev_testing.cmd_npu_probe(args)

def cmd_test_hrx_cts(args):
    return dev_testing.cmd_test_hrx_cts(args)

def cmd_test_npu(args):
    return dev_testing.cmd_test_npu(args)

def cmd_test_npu_iron_ops(args):
    return dev_testing.cmd_test_npu_iron_ops(args)

def cmd_deps(args):
    return dev_sources.cmd_deps(args)



def argparse_positive_int(value):
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


_BUILD_HANDLERS = {
    "llvm": cmd_build_llvm,
    "iron-tools": cmd_build_iron_tools,
    "hrx-xclbinutil": cmd_build_hrx_xclbinutil,
    "hrx": cmd_build_hrx,
    "hrx-cts": cmd_build_hrx_cts,
    "fastflowlm": cmd_build_fastflowlm,
    "rocm-npu-cli": cmd_build_rocm_npu_cli,
    "all": cmd_build_all,
}


def _dispatch_build(args):
    return _BUILD_HANDLERS[args.what](args)


def _add_command(subparsers, name, handler, *, exit_on_return=False):
    parser = subparsers.add_parser(name)
    parser.set_defaults(
        _handler=handler,
        _exit_on_return=exit_on_return,
    )
    return parser


def dispatch(args):
    """Dispatches using metadata owned by the selected argparse subparser."""
    result = args._handler(args)
    if args._exit_on_return:
        raise SystemExit(result)
    return result


def create_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = _add_command(sub, "doctor", cmd_doctor, exit_on_return=True)
    d.add_argument("profile", nargs="?", choices=doctor_profile_names(),
                   default="compiler-stack", help="dependency closure to validate")
    d.add_argument(
        "--deep",
        action="store_true",
        help="replay every patch series in a temporary Git index",
    )
    s = _add_command(sub, "setup", cmd_setup)
    s.add_argument("--profile", choices=setup_profile_names(),
                   default="compiler-stack", help="dependency closure to prepare")
    s.add_argument("--iron-ops-tests", action="store_true",
                   help="also install what iron-ops' own tests need (torch, pytest); "
                        "the NPU job does, the build jobs do not")
    _add_command(sub, "fetch", cmd_fetch)
    b = _add_command(sub, "build", _dispatch_build)
    b.add_argument(
        "what", choices=sorted(_BUILD_HANDLERS)
    )
    b.add_argument("--fresh", action="store_true",
                   help="clear generated CMake state and outputs before configuring")
    t = _add_command(sub, "test", cmd_test)
    t.add_argument("design", nargs="?")
    t.add_argument("--compiler-prefix",
                   help="relocated rocm-npu-compiler product prefix to test")
    t.add_argument("design_args", nargs="*", help="extra aieargs for the design script")
    tf = _add_command(sub, "test-fastflowlm", cmd_test_fastflowlm)
    tf.add_argument("--bundle", help="run only from an extracted runtime bundle")
    tf.add_argument("--wheel-package", action="store_true",
                    help="run from installed rocm-npu-runtime/fastflowlm wheels")
    tf.add_argument(
        "--inference-iterations",
        type=argparse_positive_int,
        default=1,
        metavar="N",
        help="run N independent two-turn FLM inference processes before the benchmark",
    )
    pfk = _add_command(
        sub, "package-fastflowlm-kernels", cmd_package_fastflowlm_kernels
    )
    pfk.add_argument("--fresh", action="store_true",
                     help="clear generated CMake state before building")
    vfk = _add_command(
        sub, "verify-fastflowlm-kernels", cmd_verify_fastflowlm_kernels
    )
    vfk.add_argument("--dir", default=".artifact",
                     help="directory containing the helper archive")
    tfk = _add_command(
        sub, "test-fastflowlm-kernels", cmd_test_fastflowlm_kernels
    )
    tfk.add_argument("--dir", help="directory containing the helper archive")
    tfk.add_argument("--wheel-package", action="store_true",
                     help="use installed rocm-npu-runtime/fastflowlm wheels")
    tfk.add_argument("--iterations", type=argparse_positive_int, default=20,
                     help="resident round-robin iterations (default: 20)")
    tfk.add_argument("--readback-iterations", type=argparse_positive_int,
                     default=32,
                     help="host-mapping coherence rounds (default: 32)")
    to = _add_command(sub, "test-iron-ops", cmd_test_iron_ops)
    to.add_argument("operator", nargs="?",
        help="operator to build (default: every one in tests/iron-ops/suite.py)")
    to.add_argument("--compiler-prefix",
                    help="relocated rocm-npu-compiler product prefix to test")
    to.add_argument("--device", choices=sorted(dev_iron.NPU_TARGETS),
                    help="override the NPU generation the suite pins for the operator")
    _add_command(sub, "smoke", cmd_smoke, exit_on_return=True)
    sm = _add_command(sub, "submodules", cmd_submodules)
    sm.add_argument("names", nargs="+")
    sm.add_argument("--reset", action="store_true", help="force-reset tracked files")
    sm.add_argument("--clean", nargs="*", default=[],
                    help="also remove untracked+ignored files (needed where a patch adds them)")
    sm.add_argument("--shallow", action="store_true", help="--depth 1 --jobs 4")
    _add_command(sub, "package-npu-prereqs", cmd_package_npu_prereqs)
    _add_command(sub, "package-npu-test-package", cmd_package_npu_test_package)
    _add_command(sub, "package-fastflowlm", cmd_package_fastflowlm)
    pfw = _add_command(
        sub, "package-fastflowlm-wheels", cmd_package_fastflowlm_wheels
    )
    pfw.add_argument("--version", help=f"PEP 440 package version (or ${PACKAGE_VERSION_ENV})")
    afw = _add_command(
        sub, "audit-fastflowlm-wheels", cmd_audit_fastflowlm_wheels
    )
    afw.add_argument("--dir", default=str(fastflowlm_package.FASTFLOWLM_ARTIFACTS),
                     help="directory containing the produced rocm-npu wheels")
    vfw = _add_command(
        sub, "verify-fastflowlm-wheels", cmd_verify_fastflowlm_wheels
    )
    vfw.add_argument("--dir", default=str(fastflowlm_package.FASTFLOWLM_ARTIFACTS),
                     help="directory containing the produced rocm-npu wheels")
    vfw.add_argument("--venv", default=str(fastflowlm_package.FASTFLOWLM_WHEEL_VENV),
                     help="fresh venv to create for wheel install testing")
    _add_command(
        sub, "package-compiler-product", cmd_package_compiler_product
    )
    pcw = _add_command(
        sub, "package-compiler-wheels", cmd_package_compiler_wheels
    )
    pcw.add_argument("--version",
                     help=f"PEP 440 package version (or ${PACKAGE_VERSION_ENV})")
    pcw.add_argument(
        "--product",
        help="reuse an existing validated compiler product root or parent directory",
    )
    pcw.add_argument(
        "--binding",
        metavar="PYTHON_ROOT",
        help="installed stable-ABI iron-tools Python root",
    )
    pcw.add_argument(
        "--no-meta",
        action="store_true",
        help="omit the platform-neutral meta wheel (Windows release job)",
    )
    acw = _add_command(
        sub, "audit-compiler-wheels", cmd_audit_compiler_wheels
    )
    acw.add_argument("--dir", default=str(compiler_wheels.COMPILER_WHEEL_ARTIFACTS))
    vcw = _add_command(
        sub, "verify-compiler-wheels", cmd_verify_compiler_wheels
    )
    vcw.add_argument("--dir", default=str(compiler_wheels.COMPILER_WHEEL_ARTIFACTS))
    vcw.add_argument("--version")
    vcw.add_argument("--venv", default=str(compiler_wheels.COMPILER_WHEEL_VENV))
    vcw.add_argument(
        "--consumer-root",
        help="export or reuse a hard-linked compiler tree for cross-minor ABI checks",
    )
    vcw.add_argument(
        "--consumer-only",
        action="store_true",
        help="skip static/core/e2e checks and validate bindings against --consumer-root",
    )
    vcw.add_argument(
        "--consumer-e2e",
        action="store_true",
        help="also run the compiler e2e with a shared cross-minor consumer tree",
    )
    uf = _add_command(sub, "unpack-fastflowlm", cmd_unpack_fastflowlm)
    uf.add_argument("--dir", default=".artifact", help="directory containing the downloaded archive")
    vf = _add_command(sub, "verify-fastflowlm", cmd_verify_fastflowlm)
    vf.add_argument("--dir", default=str(fastflowlm_package.FASTFLOWLM_EXTRACTED),
                    help="extracted bundle root or its parent directory")
    uc = _add_command(
        sub, "unpack-compiler-product", cmd_unpack_compiler_product
    )
    uc.add_argument("--dir", default=".artifact", help="directory containing the downloaded archive")
    uc.add_argument("--output-dir", default=str(compiler_product.COMPILER_PRODUCT_EXTRACTED),
                    help="physical extraction root (use a short path on Windows)")
    vc = _add_command(
        sub, "verify-compiler-product", cmd_verify_compiler_product
    )
    vc.add_argument("--dir", default=str(compiler_product.COMPILER_PRODUCT_STAGE),
                    help="compiler product root or its parent directory")
    vc.add_argument("--relocate", action="store_true",
                    help="copy the product to another prefix and run the off-device e2e")
    u = _add_command(sub, "unpack-npu-prereqs", cmd_unpack_npu_prereqs)
    u.add_argument("--dir", default=".artifact", help="where download-artifact put it")
    v = _add_command(
        sub,
        "verify-npu-prereqs",
        cmd_verify_npu_prereqs,
        exit_on_return=True,
    )
    v.add_argument("--extracted", action="store_true",
                   help="check an already-unpacked tree instead of the archive")
    unt = _add_command(
        sub, "unpack-npu-test-package", cmd_unpack_npu_test_package
    )
    unt.add_argument("--dir", default=".artifact", help="where download-artifact put it")
    vnt = _add_command(
        sub, "verify-npu-test-package", cmd_verify_npu_test_package
    )
    vnt.add_argument("--extracted", action="store_true",
                     help="check the unpacked package instead of the archive")
    rnt = _add_command(sub, "test-npu-package", cmd_test_npu_package)
    rnt.add_argument("--case", dest="cases", action="append",
                     help="run only this packaged case; may be repeated")
    rnt.add_argument("--lifecycle-rounds", type=argparse_positive_int, default=5,
                     help="load/dispatch/release rounds per case (default: 5)")
    rnt.add_argument("--resident-rounds", type=argparse_positive_int, default=10,
                     help="resident round-robin rounds (default: 10)")
    np = _add_command(
        sub, "npu-probe", cmd_npu_probe, exit_on_return=True
    )
    np.add_argument("--bundle", help="probe libHRX from an extracted runtime bundle")
    np.add_argument("--runtime-package", action="store_true",
                    help="probe libHRX from an installed rocm-npu-runtime wheel")
    c = _add_command(
        sub, "test-hrx-cts", cmd_test_hrx_cts, exit_on_return=True
    )
    c.add_argument("suite", nargs="?", choices=dev_build.HRX_CTS_TESTS)
    tno = _add_command(
        sub, "test-npu-iron-ops", cmd_test_npu_iron_ops
    )
    tno.add_argument("operator", nargs="?",
        help="operator to run on the NPU (default: every one in tests/iron-ops/suite.py)")
    tno.add_argument("--compiler-prefix",
                     help="relocated rocm-npu-compiler product prefix to test")
    tn = _add_command(sub, "test-npu", cmd_test_npu)
    tn.add_argument("design", nargs="?")
    tn.add_argument("--compiler-prefix",
                    help="relocated rocm-npu-compiler product prefix to test")
    tn.add_argument("--device", choices=sorted(dev_iron.NPU_TARGETS), default="npu2",
                    help="NPU generation to build the design for (default: npu2/Strix)")
    tn.add_argument("--chain", action="store_true",
                    help="run the chained testbench instead of the single-dispatch gate")
    _add_command(sub, "deps", cmd_deps)
    return p


def main():
    dispatch(create_parser().parse_args(sys.argv[1:]))


if __name__ == "__main__":
    main()
