# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Platform-neutral staging for runtime and FastFlowLM wheels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from build_tools.artifact import reset_directory
from build_tools.rocm_npu_wheel_common import (
    FASTFLOWLM_DISTRIBUTION,
    FASTFLOWLM_PACKAGE,
    RUNTIME_DISTRIBUTION,
    RUNTIME_PACKAGE,
    fastflowlm_artifact_name,
    runtime_artifact_name,
)
from build_tools.wheel_payload import (
    copy_license as _copy_license,
)
from build_tools.wheel_payload import (
    copy_license_as as _copy_license_as,
)
from build_tools.wheel_payload import (
    copy_relative_tree as _copy_relative_tree,
)
from build_tools.wheel_payload import (
    write_file as _write_file,
)
from build_tools.wheel_payload import (
    write_package_manifest as _write_package_manifest,
)


@dataclass(frozen=True)
class RuntimeWheelTrees:
    runtime_tree: Path
    runtime_root: Path
    fastflowlm_tree: Path
    fastflowlm_root: Path


def _write_runtime_module(package_root: Path) -> None:
    _write_file(package_root / "__init__.py", '''"""Package-local HRX runtime helpers."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path

_dll_directory_handle = None
_runtime_library = None


def package_root() -> Path:
    return Path(__file__).resolve().parent


def binary_dir() -> Path:
    return package_root() / "bin"


def rocm_npu_path() -> Path:
    path = binary_dir() / ("rocm-npu.exe" if os.name == "nt" else "rocm-npu")
    if not path.is_file():
        raise FileNotFoundError(f"rocm-npu executable was not found: {path}")
    return path


def libhrx_path() -> Path:
    if os.name == "nt":
        path = binary_dir() / "hrx.dll"
        if path.is_file():
            return path
        raise FileNotFoundError(f"HRX runtime was not found: {path}")
    direct = package_root() / "lib" / "libhrx.so"
    if direct.is_file():
        return direct
    candidates = sorted((package_root() / "lib").glob("libhrx.so*"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"libhrx.so was not found under {package_root() / 'lib'}")


def initialize_process():
    global _dll_directory_handle, _runtime_library
    if os.name != "nt":
        return ctypes.CDLL(
            str(libhrx_path()), mode=getattr(ctypes, "RTLD_GLOBAL", 0)
        )
    if _dll_directory_handle is None:
        _dll_directory_handle = os.add_dll_directory(str(binary_dir()))
    if _runtime_library is None:
        _runtime_library = ctypes.WinDLL(str(libhrx_path()))
    return _runtime_library


def self_test(*, load: bool = False) -> Path:
    path = libhrx_path()
    if load:
        initialize_process()
    return path


def main() -> int:
    environment = os.environ.copy()
    scripts = str(Path(sys.executable).parent)
    existing = environment.get("ROCM_NPU_PLUGIN_PATH")
    environment["ROCM_NPU_PLUGIN_PATH"] = (
        scripts if not existing else scripts + os.pathsep + existing
    )
    return subprocess.run(
        [str(rocm_npu_path()), *sys.argv[1:]],
        env=environment,
        check=False,
    ).returncode
''')
    _write_file(package_root / "__main__.py", '''from . import main

raise SystemExit(main())
''')


def _write_fastflowlm_module(package_root: Path) -> None:
    _write_file(package_root / "__init__.py", '''"""Package-local FastFlowLM runtime helpers."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def package_root() -> Path:
    return Path(__file__).resolve().parent


def flm_path() -> Path:
    path = package_root() / "bin" / ("flm.exe" if os.name == "nt" else "flm")
    if not path.is_file():
        raise FileNotFoundError(f"FastFlowLM executable was not found: {path}")
    return path


def config_path() -> Path:
    path = package_root() / "share" / "flm" / "model_list.json"
    if not path.is_file():
        raise FileNotFoundError(f"FastFlowLM model list was not found: {path}")
    return path


def xclbin_path() -> Path:
    path = package_root() / "share" / "flm"
    if not (path / "xclbins").is_dir():
        raise FileNotFoundError(f"FastFlowLM xclbins were not found: {path / 'xclbins'}")
    return path


def model_info_path() -> Path:
    path = package_root() / "share" / "flm" / "model_info.json"
    if not path.is_file():
        raise FileNotFoundError(f"FastFlowLM model info was not found: {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    executable = str(flm_path())
    args = list(sys.argv[1:] if argv is None else argv)
    env = dict(os.environ)
    env.setdefault("HRX_GPU_DRIVER", "amdxdna")
    env.setdefault("FLM_CONFIG_PATH", str(config_path()))
    env.setdefault("FLM_XCLBIN_PATH", str(xclbin_path()))
    env.setdefault("FLM_MODELINFO_PATH", str(model_info_path()))
    if os.name == "nt":
        from rocm_npu_runtime import binary_dir as runtime_binary_dir
        env["PATH"] = os.pathsep.join(
            (str(package_root() / "bin"), str(runtime_binary_dir()), env.get("PATH", ""))
        )
        return subprocess.call([executable, *args], env=env)
    os.execvpe(executable, [executable, *args], env)
    raise RuntimeError("unreachable")
''')
    _write_file(package_root / "__main__.py", '''from __future__ import annotations

from . import main


if __name__ == "__main__":
    main()
''')


def prepare_runtime_wheel_trees(
    source_bundle: Path, destination: Path
) -> RuntimeWheelTrees:
    """Creates and stages the platform-neutral runtime product trees."""
    reset_directory(destination, destination.parent)
    trees = RuntimeWheelTrees(
        runtime_tree=destination / "runtime",
        runtime_root=destination / "runtime" / RUNTIME_PACKAGE,
        fastflowlm_tree=destination / "fastflowlm",
        fastflowlm_root=destination / "fastflowlm" / FASTFLOWLM_PACKAGE,
    )
    trees.runtime_root.mkdir(parents=True)
    trees.fastflowlm_root.mkdir(parents=True)
    _write_runtime_module(trees.runtime_root)
    _write_fastflowlm_module(trees.fastflowlm_root)
    _copy_relative_tree(
        source_bundle, "share/rocm-npu/self-test", trees.runtime_root
    )
    _copy_relative_tree(source_bundle, "share/flm", trees.fastflowlm_root)

    _copy_license_as(
        source_bundle,
        "licenses/rocm-npu-staging-LICENSE",
        trees.runtime_root,
        "licenses/rocm-npu-LICENSE",
    )
    _copy_license(source_bundle, "licenses/hrx-LICENSE", trees.runtime_root)
    _copy_license(
        source_bundle, "licenses/nlohmann-json-LICENSE.MIT", trees.runtime_root
    )
    _copy_license(
        source_bundle, "licenses/THIRD_PARTY_NOTICES.md", trees.runtime_root
    )
    _copy_license_as(
        source_bundle,
        "licenses/rocm-npu-staging-LICENSE",
        trees.fastflowlm_root,
        "licenses/rocm-npu-LICENSE",
    )
    _copy_license(
        source_bundle,
        "licenses/FastFlowLM-LICENSE_RUNTIME.txt",
        trees.fastflowlm_root,
    )
    _copy_license(
        source_bundle, "licenses/FastFlowLM-TERMS.md", trees.fastflowlm_root
    )
    return trees


def write_runtime_wheel_manifests(
    trees: RuntimeWheelTrees,
    source_manifest: dict[str, object],
    *,
    version: str,
    runtime_wheel_tag: str,
    fastflowlm_wheel_tag: str,
    strip_report: dict[str, int],
    binary_dependencies: dict[str, list[str]],
) -> None:
    """Writes the shared runtime/FastFlowLM package-manifest contracts."""
    common = {
        "schema_version": 1,
        "version": version,
        "platform": source_manifest.get("platform", {}),
        "sources": source_manifest.get("sources", {}),
        "patches": source_manifest.get("patches", []),
        "dependencies": source_manifest.get("dependencies", {}),
        "toolchain": source_manifest.get("toolchain", {}),
        "source_artifact": source_manifest.get("artifact"),
        "source_product": source_manifest.get("product"),
        "strip": strip_report,
    }
    _write_package_manifest(
        trees.runtime_root,
        {
            **common,
            "product": RUNTIME_DISTRIBUTION,
            "artifact": runtime_artifact_name(),
            "components": source_manifest.get("components", []),
            "wheel_tag": runtime_wheel_tag,
            "binary_dependencies": {
                name: needed
                for name, needed in binary_dependencies.items()
                if name.startswith(f"{RUNTIME_PACKAGE}/")
            },
        },
    )
    _write_package_manifest(
        trees.fastflowlm_root,
        {
            **common,
            "product": FASTFLOWLM_DISTRIBUTION,
            "artifact": fastflowlm_artifact_name(),
            "components": [],
            "wheel_tag": fastflowlm_wheel_tag,
            "requires": [f"{RUNTIME_DISTRIBUTION} =={version}"],
            "binary_dependencies": {
                name: needed
                for name, needed in binary_dependencies.items()
                if name.startswith(f"{FASTFLOWLM_PACKAGE}/")
            },
        },
    )
