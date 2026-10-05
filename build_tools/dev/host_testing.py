# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Host-side developer test and smoke commands."""

import subprocess
import sys
from pathlib import Path

from build_tools import environment, fastflowlm_package, hrx_runtime, iron, paths


def cmd_test(args):
    if not environment.venv_py().exists():
        sys.exit(f"no venv at {paths.VENV} (run: python dev.py setup)")
    command = [
        environment.venv_py(),
        paths.ROOT / "tests" / "e2e" / "run_e2e.py",
    ]
    if getattr(args, "compiler_prefix", None):
        command += ["--compiler-prefix", args.compiler_prefix]
    if args.design:
        command.append(args.design)
    if args.design_args:
        command += ["--", *args.design_args]
    sys.exit(environment.sh(command, check=False).returncode)


def _require_fastflowlm_wheel_packages():
    missing = []
    for module in ("fastflow_sdk_runtime", "fastflow_sdk_fastflowlm"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        executable = paths.venv_bin(
            fastflowlm_package.FASTFLOWLM_WHEEL_VENV
        ) / paths.executable_name("python")
        sys.exit(
            "test-fastflowlm --wheel-package requires running dev.py with a "
            "Python environment where the fastflow-sdk-runtime and "
            "fastflow-sdk-fastflowlm wheels are installed; missing "
            f"{', '.join(missing)}. Run `python dev.py verify-fastflowlm-wheels` "
            f"first, then use `{executable} dev.py test-fastflowlm --wheel-package`."
        )


def cmd_test_fastflowlm(args):
    script = paths.ROOT / "tests" / "fastflowlm" / "run_fastflowlm_hrx_smoke.py"
    env = environment.cmake_env()
    if args.bundle:
        env["FASTFLOWLM_BUNDLE"] = str(Path(args.bundle).resolve())
    elif args.wheel_package:
        _require_fastflowlm_wheel_packages()
        env["FASTFLOWLM_INSTALLED_PACKAGE"] = "1"
    else:
        hrx_runtime.add_hrx_install_env(env)
    env.setdefault("FLM_MODEL_PATH", str(paths.FASTFLOWLM_MODEL_CACHE))
    command = [
        sys.executable,
        script,
        "--inference-iterations",
        str(args.inference_iterations),
    ]
    sys.exit(environment.sh(command, env=env, check=False).returncode)


def cmd_test_iron_ops(args):
    if not environment.venv_py().exists():
        sys.exit(f"no venv at {paths.VENV} (run: python dev.py setup)")
    command = [
        environment.venv_py(),
        paths.ROOT / "tests" / "iron-ops" / "build_operator.py",
    ]
    if getattr(args, "operator", None):
        command += ["--operator", args.operator]
    if getattr(args, "device", None):
        command += ["--device", args.device]
    command += ["--jobs", str(getattr(args, "jobs", 0) or 0)]
    sys.exit(
        environment.sh(
            command,
            env=iron.iron_ops_env(getattr(args, "compiler_prefix", None)),
            check=False,
        ).returncode
    )


def cmd_smoke(args):
    rc = 0
    xclbinutil_build = (
        paths.PROJ / "iron-tools" / "third_party" / "hrx-xclbinutil" / "bld"
    )
    if (xclbinutil_build / "CMakeCache.txt").exists():
        print("== hrx-xclbinutil smoke (ctest) ==")
        rc |= subprocess.run(
            [
                "ctest",
                "--test-dir",
                str(xclbinutil_build),
                "--output-on-failure",
            ],
            env=environment.cmake_env(),
        ).returncode
    else:
        print(
            "smoke: hrx-xclbinutil not built (run: python dev.py build hrx-xclbinutil)"
        )
        rc = 1
    aie_opt = paths.IRON_TOOLS_BUILD / "bin" / paths.executable_name("aie-opt")
    if aie_opt.exists():
        print("== iron-tools smoke (aie-opt) ==")
        environment.sh([aie_opt, "--version"])
        result = subprocess.run(
            [str(aie_opt)],
            input=b"module { aie.device(npu2) {} }\n",
            capture_output=True,
        )
        ok = result.returncode == 0
        print(f"  aie-opt parses aie.device(npu2): {'ok' if ok else 'FAIL'}")
        if not ok:
            print(result.stderr.decode()[:300])
            rc = 1
    else:
        print("smoke: aie-opt not built (run: python dev.py build iron-tools)")
        rc = 1
    print("smoke: PASS" if rc == 0 else "smoke: FAIL")
    return rc
