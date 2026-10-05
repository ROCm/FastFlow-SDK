# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""On-device NPU test and probe commands.

CI NPU e2e uses ``npu-probe``, ``test-hrx-cts``, and ``test-npu-package``.
``test-npu-iron-ops`` is a local extra (IRON pytest under HRX); it is not a
workflow step. The README compile+run example is ``test-npu-jit`` on the
dedicated examples job.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

from build_tools import (
    environment,
    hrx_runtime,
    iron,
    linux_npu,
    npu_test_package,
    paths,
)
from build_tools.windows_npu import probe_runtime as probe_windows_npu_runtime

HRX_CTS_TIMEOUT_SECONDS = 30 * 60
NPU_TEST_PACKAGE_TIMEOUT_SECONDS = 10 * 60


def _select_hrx_runtime(env):
    """Select HRX for IRON. IRON_RUNTIME is the older name for NPU_RUNTIME."""
    env["IRON_RUNTIME"] = "hrx"
    env["NPU_RUNTIME"] = "hrx"
    return hrx_runtime.add_hrx_install_env(env)


def cmd_test_npu_package(args):
    """Dispatch the prebuilt XADX package produced by package-npu-test-package.

    This is the CI NPU operator gate: no Peano, no iron-ops pytest.
    """
    try:
        npu_test_package.verify_npu_test_package(
            npu_test_package.NPU_TEST_PACKAGE_EXTRACTED
        )
    except RuntimeError as error:
        sys.exit(str(error))
    env = hrx_runtime.add_hrx_install_env(os.environ.copy())
    command = [
        environment.venv_py(),
        "-u",
        paths.ROOT / "tests" / "npu" / "run_test_package.py",
        npu_test_package.NPU_TEST_PACKAGE_EXTRACTED,
    ]
    for case in args.cases or []:
        command += ["--case", case]
    command += [
        "--lifecycle-rounds",
        str(args.lifecycle_rounds),
        "--resident-rounds",
        str(args.resident_rounds),
    ]
    try:
        result = environment.sh(
            command,
            env=env,
            check=False,
            timeout=NPU_TEST_PACKAGE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        sys.exit(
            "test-npu-package timed out after "
            f"{NPU_TEST_PACKAGE_TIMEOUT_SECONDS} seconds; "
            "the last [npu-test] line identifies the blocked phase"
        )
    sys.exit(result.returncode)


def cmd_npu_probe(args):
    if paths.is_windows():
        library = hrx_runtime.libhrx_path(args.bundle, args.runtime_package)
        if not library:
            print("npu-probe: hrx.dll was not found")
            return 1
        return probe_windows_npu_runtime(library)
    rc = 0
    linux_npu.print_npu_driver_info()
    print("== NPU device / libhrx probe ==")
    devnode_rc, devnodes = linux_npu.probe_npu_devnodes()
    rc |= devnode_rc
    if not devnodes:
        print(
            "  [MISS] skipping hrx_gpu_initialize because no usable "
            "NPU devnode was found"
        )
        print("npu-probe: FAIL")
        return 1
    rc |= hrx_runtime.probe_libhrx(args.bundle, args.runtime_package)
    print("npu-probe: " + ("ok" if rc == 0 else "FAIL"))
    return rc


def _gtest_counts(output):
    passed = skipped = 0
    for line in output.splitlines():
        match = re.match(r"\[\s+PASSED\s+\] (\d+) test", line)
        if match:
            passed = int(match.group(1))
        match = re.match(r"\[\s+SKIPPED\s+\] (\d+) test", line)
        if match:
            skipped = int(match.group(1))
    return passed, skipped


def cmd_test_hrx_cts(args):
    cts = paths.HRX_CMAKE_BUILD.joinpath(*hrx_runtime.HRX_CTS_DIR)
    names = [args.suite] if getattr(args, "suite", None) else hrx_runtime.HRX_CTS_TESTS
    executables = {}
    for name in names:
        candidate = cts / paths.executable_name(name)
        if not candidate.is_file():
            sys.exit(
                f"CTS binary missing ({candidate}) — "
                "run: python dev.py build hrx --include-cts"
            )
        executables[name] = candidate
    failed, counts = [], {}
    for name in names:
        print(f"\n=== hrx CTS: {name} ===", flush=True)
        timed_out = False
        try:
            result = environment.run(
                [executables[name]],
                capture_output=True,
                text=True,
                check=False,
                timeout=HRX_CTS_TIMEOUT_SECONDS,
            )
            stdout, stderr = result.stdout, result.stderr
            returncode = result.returncode
        except subprocess.TimeoutExpired as error:
            timed_out = True
            stdout = error.stdout or ""
            stderr = error.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            returncode = 124
        print(stdout, end="")
        if stderr:
            print(stderr, end="", file=sys.stderr)
        if timed_out:
            print(
                f"{name}: timed out after {HRX_CTS_TIMEOUT_SECONDS} seconds",
                file=sys.stderr,
            )
        counts[name] = _gtest_counts(stdout)
        if returncode != 0:
            failed.append(name)
        elif counts[name][0] == 0:
            print(f"{name}: every case skipped; the suite asserts nothing")
            failed.append(name)
    print("\nhrx CTS summary:")
    for name in names:
        passed, skipped = counts.get(name, (0, 0))
        print(f"  {name:24s} {passed:3d} passed  {skipped:3d} skipped")
    total_passed = sum(count[0] for count in counts.values())
    total_skipped = sum(count[1] for count in counts.values())
    if failed:
        print(f"\nhrx CTS: FAIL ({', '.join(failed)})")
        return 1
    print(
        f"\nhrx CTS: PASS ({total_passed} passed, {total_skipped} "
        f"skipped across {len(names)} suites)"
    )
    return 0


def _resolved_compiler_prefix(compiler_prefix=None):
    if compiler_prefix:
        return Path(compiler_prefix).resolve()
    return iron.compiler_prefix_from_env()


def _compiler_python_root(compiler_prefix=None):
    prefix = _resolved_compiler_prefix(compiler_prefix)
    return prefix / "python" if prefix else paths.IRON_TOOLS_BUILD / "python"


def _hrxruntime_discovery(compiler_prefix=None):
    roots = [_compiler_python_root(compiler_prefix)]
    installed = iron.installed_aie_python_root()
    if installed is not None and installed not in roots:
        roots.append(installed)
    for root in roots:
        candidate = (
            root / "aie" / "utils" / "hostruntime" / "hrxruntime" / "discovery.py"
        )
        if candidate.is_file():
            return candidate
    return None


def _require_hrx_runtime(compiler_prefix=None):
    if _hrxruntime_discovery(compiler_prefix) is None:
        sys.exit(
            "hrxruntime missing from the aie package — rebuild iron-tools "
            "or install fastflow-sdk-compiler-bindings"
        )
    if not hrx_runtime.libhrx_path():
        library_name = "hrx.dll" if paths.is_windows() else "libhrx.so"
        sys.exit(f"{library_name} not found — run: python dev.py build hrx")


def _toolchain_aie_opt(compiler_prefix=None):
    prefix = _resolved_compiler_prefix(compiler_prefix)
    name = paths.executable_name("aie-opt")
    if prefix:
        return prefix / "bin" / name
    return paths.IRON_TOOLS_BUILD / "bin" / name


def _compiler_env(compiler_prefix=None):
    if compiler_prefix or _toolchain_aie_opt(compiler_prefix).is_file():
        return iron.iron_env(compiler_prefix)
    return os.environ.copy()


def cmd_test_npu(args):
    design = getattr(args, "design", None) or "vector_scalar_add"
    design_dir = paths.PROJ / "iron-tools" / "programming_examples" / "basic" / design
    if not design_dir.is_dir():
        sys.exit(f"{design}: no such design under programming_examples/basic")
    _require_hrx_runtime(getattr(args, "compiler_prefix", None))
    env = _select_hrx_runtime(iron.iron_env(getattr(args, "compiler_prefix", None)))
    target = getattr(args, "device", None) or "npu2"
    try:
        artifacts = iron.build_design(
            design_dir, iron.NPU_TARGETS[target]["device"], [], env
        )
    except iron.E2EError as error:
        sys.exit(f"test-npu design build failed: {error}")
    iron.require_txn_devgen(artifacts.insts, target)
    if getattr(args, "chain", False):
        if not (design_dir / "test_chain_hrx.py").exists():
            sys.exit(
                f"{design}: no chained testbench (only vector_scalar_add ships one)"
            )
        testbench = [environment.venv_py(), "test_chain_hrx.py"]
    else:
        testbench = [
            environment.venv_py(),
            paths.ROOT / "tests" / "npu" / "single_dispatch.py",
        ]
    environment.sh(
        testbench
        + [
            "--xclbin",
            str(artifacts.xclbin),
            "--instr",
            str(artifacts.insts),
            "--kernel",
            "MLIR_AIE",
        ],
        cwd=design_dir,
        env=env,
    )
    print(f"test-npu: ok ({design} ran on the NPU via hrx)")


def cmd_test_npu_jit(args):
    build_only = getattr(args, "build_only", False)
    compiler_prefix = getattr(args, "compiler_prefix", None)
    if not build_only:
        _require_hrx_runtime(compiler_prefix)
    env = _compiler_env(compiler_prefix)
    command = [
        environment.venv_py(),
        paths.ROOT / "tests" / "npu" / "jit_smoke.py",
    ]
    if build_only:
        command.append("--build-only")
    else:
        env = _select_hrx_runtime(env)
    environment.sh(command, env=env)
    print(
        "test-npu-jit: ok "
        + (
            "(compiler + JIT, no dispatch)"
            if build_only
            else "(ran on the NPU via hrx)"
        )
    )


def npu_iron_ops_pytest_selection(operators, ops_dir: Path):
    """Dirs and -k clauses for operators that have an HRX pytest filter.

    ``hrx_pytest_filter is None`` means local test-npu-iron-ops must not
    collect that operator's test.py. An empty -k would otherwise run every
    test in the directory, including XRT fused-ELF cases that fail on HRX.
    CI does not call this helper.
    """
    test_dirs = []
    filters = []
    ran = []
    skipped = []
    for operator in operators:
        test_dir = ops_dir / "iron" / "operators" / operator.name
        if operator.hrx_pytest_filter is None:
            skipped.append(operator.name)
            continue
        if not (test_dir / "test.py").is_file():
            raise RuntimeError(
                f"{operator.name}: hrx_pytest_filter is set but "
                f"{test_dir / 'test.py'} is missing"
            )
        ran.append(operator.name)
        test_dirs.append(str(test_dir))
        filters.append(operator.hrx_pytest_filter)
    return ran, test_dirs, filters, skipped


def cmd_test_npu_iron_ops(args):
    """Run iron-ops' own pytest on a machine with an NPU (local, not CI).

    Recompiles inside each ``run_test``. Needs ``setup --iron-ops-tests``.
    """
    try:
        operators = iron.select(getattr(args, "operator", None))
    except KeyError as error:
        sys.exit(error.args[0])
    compiler_prefix = getattr(args, "compiler_prefix", None)
    _require_hrx_runtime(compiler_prefix)
    ops_dir = paths.PROJ / "iron-ops"
    try:
        ran, test_dirs, filters, skipped = npu_iron_ops_pytest_selection(
            operators, ops_dir
        )
    except RuntimeError as error:
        sys.exit(str(error))
    for name in skipped:
        print(
            f"{name}: compile-only (no hrx_pytest_filter); skipped by test-npu-iron-ops"
        )
    if not ran:
        names = ", ".join(operator.name for operator in operators)
        print(f"test-npu-iron-ops: ok (nothing to run on the NPU: {names})")
        return
    pytest = paths.venv_bin() / paths.executable_name("pytest")
    if not pytest.exists():
        sys.exit(
            "pytest not in the venv — install iron-ops test deps: "
            "python dev.py setup --iron-ops-tests"
        )
    env = _select_hrx_runtime(iron.iron_ops_env(compiler_prefix))
    pytest_filter = ["-k", " or ".join(f"({value})" for value in filters)]
    rc = subprocess.run(
        [
            str(environment.venv_py()),
            "-m",
            "pytest",
            *test_dirs,
            "-m",
            "not extensive",
            *pytest_filter,
            "--iterations",
            "1",
            "--csv-output",
            str(paths.OUT / "iron-ops.csv"),
        ],
        cwd=ops_dir,
        env=env,
    ).returncode
    names = ", ".join(ran)
    if rc != 0:
        sys.exit(f"test-npu-iron-ops: FAIL ({names})")
    print(f"test-npu-iron-ops: ok ({names} ran on the NPU via hrx, numerics verified)")
