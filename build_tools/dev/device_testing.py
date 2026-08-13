# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""On-device NPU test and probe commands."""

import os
import re
import subprocess
import sys
from pathlib import Path

from build_tools import environment, hrx_runtime, iron, linux_npu, npu_test_package, paths
from build_tools.windows_npu import probe_runtime as probe_windows_npu_runtime

HRX_CTS_TIMEOUT_SECONDS = 30 * 60


def cmd_test_npu_package(args):
    try:
        npu_test_package.verify_npu_test_package(
            npu_test_package.NPU_TEST_PACKAGE_EXTRACTED
        )
    except RuntimeError as error:
        sys.exit(str(error))
    env = hrx_runtime.add_hrx_install_env(os.environ.copy())
    command = [
        environment.venv_py(),
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
    sys.exit(environment.sh(command, env=env, check=False).returncode)


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
    names = (
        [args.suite]
        if getattr(args, "suite", None)
        else hrx_runtime.HRX_CTS_TESTS
    )
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


def _compiler_python_root(compiler_prefix=None):
    value = compiler_prefix or os.environ.get("ROCM_NPU_COMPILER_PREFIX")
    prefix = Path(value).resolve() if value else None
    return prefix / "python" if prefix else paths.IRON_TOOLS_BUILD / "python"


def _require_hrx_runtime(compiler_prefix=None):
    python_root = _compiler_python_root(compiler_prefix)
    discovery = (
        python_root / "aie" / "utils" / "hostruntime" / "hrxruntime" / "discovery.py"
    )
    if not discovery.exists():
        sys.exit(
            "hrxruntime missing from the built aie package — "
            "rebuild against a current iron-tools "
            "(run: python dev.py build iron-tools)"
        )
    if not hrx_runtime.libhrx_path():
        library_name = "hrx.dll" if paths.is_windows() else "libhrx.so"
        sys.exit(f"{library_name} not found — run: python dev.py build hrx")


def cmd_test_npu(args):
    design = getattr(args, "design", None) or "vector_scalar_add"
    design_dir = paths.PROJ / "iron-tools" / "programming_examples" / "basic" / design
    if not design_dir.is_dir():
        sys.exit(f"{design}: no such design under programming_examples/basic")
    _require_hrx_runtime(getattr(args, "compiler_prefix", None))
    env = iron.iron_env(getattr(args, "compiler_prefix", None))
    env["IRON_RUNTIME"] = "hrx"
    hrx_runtime.add_hrx_install_env(env)
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
                f"{design}: no chained testbench " "(only vector_scalar_add ships one)"
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


def cmd_test_npu_iron_ops(args):
    try:
        operators = iron.select(getattr(args, "operator", None))
    except KeyError as error:
        sys.exit(error.args[0])
    compiler_prefix = getattr(args, "compiler_prefix", None)
    _require_hrx_runtime(compiler_prefix)
    pytest = paths.venv_bin() / paths.executable_name("pytest")
    if not pytest.exists():
        sys.exit(
            "pytest not in the venv — the NPU job installs the iron-ops "
            "test deps: python dev.py setup --iron-ops-tests"
        )
    ops_dir = paths.PROJ / "iron-ops"
    test_dirs = []
    for operator in operators:
        test_dir = ops_dir / "iron" / "operators" / operator.name
        if not (test_dir / "test.py").exists():
            sys.exit(
                f"{operator.name}: no test.py under " "projects/iron-ops/iron/operators"
            )
        test_dirs.append(str(test_dir))
    env = iron.iron_ops_env(compiler_prefix)
    env["IRON_RUNTIME"] = "hrx"
    hrx_runtime.add_hrx_install_env(env)
    filters = [
        operator.hrx_pytest_filter
        for operator in operators
        if operator.hrx_pytest_filter
    ]
    pytest_filter = (
        ["-k", " or ".join(f"({value})" for value in filters)] if filters else []
    )
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
    names = ", ".join(operator.name for operator in operators)
    if rc != 0:
        sys.exit(f"test-npu-iron-ops: FAIL ({names})")
    print(
        f"test-npu-iron-ops: ok ({names} ran on the NPU via hrx, " "numerics verified)"
    )
