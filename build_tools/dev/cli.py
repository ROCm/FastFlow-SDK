# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""ROCm NPU developer build, test, setup, and packaging commands.

Run ``python dev.py --help`` for the complete command inventory.
"""

import argparse
import sys

from build_tools import (
    compiler_product,
    compiler_wheels,
    fastflowlm_package,
    fastflowlm_xadx_assets,
    hrx_runtime,
    npu_prereqs,
    npu_test_package,
    rocm_npu_qualification_wheel,
)
from build_tools import (
    iron as dev_iron,
)
from build_tools.dev import build as dev_build
from build_tools.dev import device_testing, host_testing
from build_tools.dev import doctor as dev_doctor
from build_tools.dev import patches as dev_patches
from build_tools.dev import setup as dev_setup
from build_tools.dev import sources as dev_sources
from build_tools.dev.profiles import doctor_profile_names, setup_profile_names
from build_tools.package_version import PACKAGE_VERSION_ENV


def argparse_positive_int(value):
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _dispatch_build(args):
    return dev_build.BUILD_HANDLERS[args.what](args)


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
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    d = _add_command(sub, "doctor", dev_doctor.cmd_doctor, exit_on_return=True)
    d.add_argument(
        "profile",
        nargs="?",
        choices=doctor_profile_names(),
        default="compiler-stack",
        help="dependency closure to validate",
    )
    d.add_argument(
        "--deep",
        action="store_true",
        help="replay every patch series in a temporary Git index",
    )
    s = _add_command(sub, "setup", dev_setup.cmd_setup)
    s.add_argument(
        "--profile",
        choices=setup_profile_names(),
        default="compiler-stack",
        help="dependency closure to prepare",
    )
    s.add_argument(
        "--iron-ops-tests",
        action="store_true",
        help="also install what iron-ops' own tests need (torch, pytest); "
        "the NPU job does, the build jobs do not",
    )
    _add_command(sub, "fetch", dev_patches.cmd_fetch)
    b = _add_command(sub, "build", _dispatch_build)
    b.add_argument("what", choices=sorted(dev_build.BUILD_HANDLERS))
    b.add_argument(
        "--fresh",
        action="store_true",
        help="clear generated CMake state and outputs before configuring",
    )
    b.add_argument(
        "--include-tests",
        action="store_true",
        help="when building hrx, build and run hermetic amdxdna host unit tests",
    )
    b.add_argument(
        "--include-cts",
        action="store_true",
        help="when building hrx, also build on-device amdxdna CTS binaries "
        "(run them with test-hrx-cts)",
    )
    t = _add_command(sub, "test", host_testing.cmd_test)
    t.add_argument("design", nargs="?")
    t.add_argument(
        "--compiler-prefix", help="relocated rocm-npu-compiler product prefix to test"
    )
    t.add_argument("design_args", nargs="*", help="extra aieargs for the design script")
    tf = _add_command(sub, "test-fastflowlm", host_testing.cmd_test_fastflowlm)
    tf.add_argument("--bundle", help="run only from an extracted runtime bundle")
    tf.add_argument(
        "--wheel-package",
        action="store_true",
        help="run from installed rocm-npu-runtime/fastflowlm wheels",
    )
    tf.add_argument(
        "--inference-iterations",
        type=argparse_positive_int,
        default=1,
        metavar="N",
        help="run N independent two-turn FLM inference processes before the benchmark",
    )
    gfx = _add_command(
        sub, "generate-fastflowlm-xadx", fastflowlm_xadx_assets.cmd_generate_xadx_assets
    )
    gfx.add_argument(
        "--fresh",
        action="store_true",
        help="clear generated CMake state before building",
    )
    gfx.add_argument(
        "--dir",
        default=str(fastflowlm_xadx_assets.XADX_STAGE),
        help="generated XADX asset directory",
    )
    to = _add_command(sub, "test-iron-ops", host_testing.cmd_test_iron_ops)
    to.add_argument(
        "operator",
        nargs="?",
        help="operator to build (default: every one in build_tools/iron.py)",
    )
    to.add_argument(
        "--compiler-prefix", help="relocated rocm-npu-compiler product prefix to test"
    )
    to.add_argument(
        "--device",
        choices=sorted(dev_iron.NPU_TARGETS),
        help="override the NPU generation the suite pins for the operator",
    )
    _add_command(sub, "smoke", host_testing.cmd_smoke, exit_on_return=True)
    sm = _add_command(sub, "submodules", dev_sources.cmd_submodules)
    sm.add_argument("names", nargs="+")
    sm.add_argument("--reset", action="store_true", help="force-reset tracked files")
    sm.add_argument(
        "--clean",
        nargs="*",
        default=[],
        help="also remove untracked+ignored files (needed where a patch adds them)",
    )
    sm.add_argument("--shallow", action="store_true", help="--depth 1 --jobs 4")
    _add_command(sub, "package-npu-prereqs", npu_prereqs.cmd_package_npu_prereqs)
    _add_command(
        sub, "package-npu-test-package", npu_test_package.cmd_package_npu_test_package
    )
    _add_command(sub, "package-fastflowlm", fastflowlm_package.cmd_package_fastflowlm)
    pfw = _add_command(
        sub,
        "package-fastflowlm-wheels",
        fastflowlm_package.cmd_package_fastflowlm_wheels,
    )
    pfw.add_argument(
        "--version", help=f"PEP 440 package version (or ${PACKAGE_VERSION_ENV})"
    )
    pqw = _add_command(
        sub,
        "package-qualification-wheel",
        rocm_npu_qualification_wheel.cmd_package_qualification_wheel,
    )
    pqw.add_argument(
        "--version", help=f"PEP 440 package version (or ${PACKAGE_VERSION_ENV})"
    )
    pqw.add_argument("--dir", default="build/artifacts", help="wheel output directory")
    pqw.add_argument(
        "--xadx-assets",
        required=True,
        help="verified generated FastFlowLM XADX asset directory",
    )
    afw = _add_command(
        sub, "audit-fastflowlm-wheels", fastflowlm_package.cmd_audit_fastflowlm_wheels
    )
    afw.add_argument(
        "--dir",
        default=str(fastflowlm_package.FASTFLOWLM_ARTIFACTS),
        help="directory containing the produced rocm-npu wheels",
    )
    vfw = _add_command(
        sub, "verify-fastflowlm-wheels", fastflowlm_package.cmd_verify_fastflowlm_wheels
    )
    vfw.add_argument(
        "--dir",
        default=str(fastflowlm_package.FASTFLOWLM_ARTIFACTS),
        help="directory containing the produced rocm-npu wheels",
    )
    vfw.add_argument(
        "--venv",
        default=str(fastflowlm_package.FASTFLOWLM_WHEEL_VENV),
        help="fresh venv to create for wheel install testing",
    )
    _add_command(
        sub, "package-compiler-product", compiler_product.cmd_package_compiler_product
    )
    pcw = _add_command(
        sub, "package-compiler-wheels", compiler_wheels.cmd_package_compiler_wheels
    )
    pcw.add_argument(
        "--version", help=f"PEP 440 package version (or ${PACKAGE_VERSION_ENV})"
    )
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
        sub, "audit-compiler-wheels", compiler_wheels.cmd_audit_compiler_wheels
    )
    acw.add_argument("--dir", default=str(compiler_wheels.COMPILER_WHEEL_ARTIFACTS))
    vcw = _add_command(
        sub, "verify-compiler-wheels", compiler_wheels.cmd_verify_compiler_wheels
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
    uf = _add_command(
        sub, "unpack-fastflowlm", fastflowlm_package.cmd_unpack_fastflowlm
    )
    uf.add_argument(
        "--dir", default=".artifact", help="directory containing the downloaded archive"
    )
    vf = _add_command(
        sub, "verify-fastflowlm", fastflowlm_package.cmd_verify_fastflowlm
    )
    vf.add_argument(
        "--dir",
        default=str(fastflowlm_package.FASTFLOWLM_EXTRACTED),
        help="extracted bundle root or its parent directory",
    )
    uc = _add_command(
        sub, "unpack-compiler-product", compiler_product.cmd_unpack_compiler_product
    )
    uc.add_argument(
        "--dir", default=".artifact", help="directory containing the downloaded archive"
    )
    uc.add_argument(
        "--output-dir",
        default=str(compiler_product.COMPILER_PRODUCT_EXTRACTED),
        help="physical extraction root (use a short path on Windows)",
    )
    vc = _add_command(
        sub, "verify-compiler-product", compiler_product.cmd_verify_compiler_product
    )
    vc.add_argument(
        "--dir",
        default=str(compiler_product.COMPILER_PRODUCT_STAGE),
        help="compiler product root or its parent directory",
    )
    vc.add_argument(
        "--relocate",
        action="store_true",
        help="copy the product to another prefix and run the off-device e2e",
    )
    u = _add_command(sub, "unpack-npu-prereqs", npu_prereqs.cmd_unpack_npu_prereqs)
    u.add_argument("--dir", default=".artifact", help="where download-artifact put it")
    v = _add_command(
        sub,
        "verify-npu-prereqs",
        npu_prereqs.cmd_verify_npu_prereqs,
        exit_on_return=True,
    )
    v.add_argument(
        "--extracted",
        action="store_true",
        help="check an already-unpacked tree instead of the archive",
    )
    unt = _add_command(
        sub, "unpack-npu-test-package", npu_test_package.cmd_unpack_npu_test_package
    )
    unt.add_argument(
        "--dir", default=".artifact", help="where download-artifact put it"
    )
    vnt = _add_command(
        sub, "verify-npu-test-package", npu_test_package.cmd_verify_npu_test_package
    )
    vnt.add_argument(
        "--extracted",
        action="store_true",
        help="check the unpacked package instead of the archive",
    )
    rnt = _add_command(sub, "test-npu-package", device_testing.cmd_test_npu_package)
    rnt.add_argument(
        "--case",
        dest="cases",
        action="append",
        help="run only this packaged case; may be repeated",
    )
    rnt.add_argument(
        "--lifecycle-rounds",
        type=argparse_positive_int,
        default=5,
        help="load/dispatch/release rounds per case (default: 5)",
    )
    rnt.add_argument(
        "--resident-rounds",
        type=argparse_positive_int,
        default=10,
        help="resident round-robin rounds (default: 10)",
    )
    np = _add_command(
        sub, "npu-probe", device_testing.cmd_npu_probe, exit_on_return=True
    )
    np.add_argument("--bundle", help="probe libHRX from an extracted runtime bundle")
    np.add_argument(
        "--runtime-package",
        action="store_true",
        help="probe libHRX from an installed rocm-npu-runtime wheel",
    )
    c = _add_command(
        sub,
        "test-hrx-cts",
        device_testing.cmd_test_hrx_cts,
        exit_on_return=True,
    )
    c.add_argument("suite", nargs="?", choices=hrx_runtime.HRX_CTS_TESTS)
    tno = _add_command(sub, "test-npu-iron-ops", device_testing.cmd_test_npu_iron_ops)
    tno.add_argument(
        "operator",
        nargs="?",
        help="operator to run on the NPU (default: every one in build_tools/iron.py)",
    )
    tno.add_argument(
        "--compiler-prefix", help="relocated rocm-npu-compiler product prefix to test"
    )
    tnj = _add_command(sub, "test-npu-jit", device_testing.cmd_test_npu_jit)
    tnj.add_argument(
        "--build-only",
        action="store_true",
        help="compile the smoke design but do not dispatch it (no NPU needed)",
    )
    tnj.add_argument(
        "--compiler-prefix", help="relocated rocm-npu-compiler product prefix to test"
    )
    tn = _add_command(sub, "test-npu", device_testing.cmd_test_npu)
    tn.add_argument("design", nargs="?")
    tn.add_argument(
        "--compiler-prefix", help="relocated rocm-npu-compiler product prefix to test"
    )
    tn.add_argument(
        "--device",
        choices=sorted(dev_iron.NPU_TARGETS),
        default="npu2",
        help="NPU generation to build the design for (default: npu2/Strix)",
    )
    tn.add_argument(
        "--chain",
        action="store_true",
        help="run the chained testbench instead of the single-dispatch gate",
    )
    _add_command(sub, "deps", dev_sources.cmd_deps)
    return p


def main():
    dispatch(create_parser().parse_args(sys.argv[1:]))


if __name__ == "__main__":
    main()
