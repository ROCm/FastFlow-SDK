# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""FastFlowLM application build command."""

import json
import os
import shutil
import sys
from pathlib import Path

from build_tools import environment, paths
from build_tools.artifact import directory_state, git_source_state
from build_tools.dev import patches
from build_tools.fastflowlm_deps import dependency_errors, dependency_paths

_QUALIFICATION_PYTHON = paths.ROOT / "tools" / "rocm-npu-qualification" / "python"
if str(_QUALIFICATION_PYTHON) not in sys.path:
    sys.path.insert(0, str(_QUALIFICATION_PYTHON))

from rocm_npu_qualification import host_xrt

_directory_has_xrt_runtime = host_xrt.directory_has_xrt_runtime
_has_xrt_headers = host_xrt.has_xrt_headers
_host_xrt_prefix = host_xrt.host_xrt_prefix
_host_xrt_prefix_candidates = host_xrt.host_xrt_prefix_candidates
_is_host_xrt_prefix = host_xrt.is_host_xrt_prefix
_windows_install_roots = host_xrt.windows_install_roots

BUILD_JOBS = os.environ.get("RNS_BUILD_JOBS") or str(os.cpu_count() or 8)
FRESH_BUILD_MARKER = ".rns-fresh-build.json"


def preset_cache_variables(source: Path) -> dict[str, str]:
    """Return ``common-default`` cache variables from FastFlowLM CMakePresets.json."""
    path = source / "CMakePresets.json"
    try:
        presets = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        sys.exit(f"unable to read FastFlowLM CMakePresets.json ({path}): {error}")
    for preset in presets.get("configurePresets") or []:
        if preset.get("name") == "common-default":
            cache = preset.get("cacheVariables") or {}
            return {str(key): str(value) for key, value in cache.items()}
    sys.exit(f"missing common-default cacheVariables in {path}")


def _require_fastflowlm_deps(env):
    configured = dependency_paths(paths.ROOT)
    missing = dependency_errors(
        configured,
        cargo_available=bool(environment.which("cargo", env)),
        rustc_version=environment.rustc_version(env),
    )
    if missing:
        sys.exit("missing FastFlowLM prerequisite(s):\n  - " + "\n  - ".join(missing))
    return configured


def _prepare_fresh_cmake_tree(path, expected):
    if path.resolve() != expected.resolve():
        raise RuntimeError(f"refusing to clear unexpected build directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def _host_xrt_lib_dir(prefix: Path) -> Path:
    """Return the directory CMake should pass as ``XRT_LIB_DIR``.

    Linux FHS installs keep ``libxrt_coreutil.so`` under ``prefix/lib``.
    Windows trees keep import libs and DLLs in ``prefix/lib``, ``prefix/bin``,
    or directly in the prefix.
    """
    for candidate in (prefix / "lib", prefix / "bin", prefix):
        if _directory_has_xrt_runtime(candidate):
            return candidate
    lib = prefix / "lib"
    return lib if lib.is_dir() else prefix


def _host_xrt_include_dir(prefix, *, windows=None):
    """Return the directory that contains ``xrt/xrt_bo.h``, or ``None``.

    Headers are taken from the discovered prefix (``include/`` or an XRT
    source layout under it). Linux also probes the distro ``xrt`` package.
    A runtime-only tree without SDK headers is not a compile-time install.
    """
    selected_windows = paths.is_windows() if windows is None else windows
    candidates = []
    if prefix is not None:
        candidates.append(Path(prefix) / "include")
        candidates.append(Path(prefix) / "src" / "runtime_src" / "core" / "include")
    if not selected_windows:
        candidates.extend((Path("/usr/include/xrt"), Path("/usr/include")))
    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        if _has_xrt_headers(candidate):
            return resolved
    return None


def _missing_host_xrt_message(*, windows=None):
    selected_windows = paths.is_windows() if windows is None else windows
    extra = (
        ", or put xrt_coreutil.dll on PATH"
        if selected_windows
        else " or install to /opt/xilinx/xrt"
    )
    return (
        "missing host XRT runtime: set XILINX_XRT to an XRT install"
        + extra
        + " (the XRT FastFlowLM build links the host install; it does "
        "not bundle a relocatable XRT prefix)"
    )


def _host_xrt_pkgconfig_dir(prefix):
    """Return the directory containing a real host ``xrt.pc``, or ``None``.

    Distro ``xrt.pc`` is often broken (``prefix=/usr//usr``). A file next to
    the runtime prefix is a real pin; a synthesized stub with ``Version: 2.0``
    is not. CMake is given ``-DXRT_INCLUDE_DIR`` / ``-DXRT_LIB_DIR`` from the
    discovered install, so a stub is unnecessary.
    """
    host_pc = Path(prefix) / "share" / "pkgconfig" / "xrt.pc"
    return host_pc.parent if host_pc.is_file() else None


def cmd_build_fastflowlm(args):
    _build_fastflowlm(args, backend="hrx")


def cmd_build_fastflowlm_xrt(args):
    _build_fastflowlm(args, backend="xrt")


def _build_fastflowlm(args, *, backend):
    """Configure, build and install FastFlowLM for the selected NPU backend.

    ``backend="hrx"`` (default) is the shipped product: it links the HRX
    amdxdna runtime and installs an FHS ``bin/flm`` tree consumed by the wheels.
    ``backend="xrt"`` is a diagnostic-only comparison build (``FLM_USE_HRX=OFF``):
    it links the discovered host XRT install (``$XILINX_XRT``, platform default
    roots, or ``PATH``), installs a matching FHS ``bin/flm`` tree, and is never
    packaged into the HRX wheels. The two use entirely separate
    CMake/build/install trees so they never share state.
    """
    if backend not in ("hrx", "xrt"):
        raise ValueError(f"unknown FastFlowLM backend: {backend}")
    source = paths.PROJ / "FastFlowLM" / "src"
    if not (source / "CMakeLists.txt").exists():
        sys.exit(
            "missing projects/FastFlowLM "
            "(run: git submodule update --init --recursive "
            "projects/FastFlowLM)"
        )
    cache = preset_cache_variables(source)
    flm_version = cache.get("FLM_VERSION")
    npu_version = cache.get("NPU_VERSION")
    if not flm_version or not npu_version:
        sys.exit(
            "FastFlowLM CMakePresets.json common-default must set "
            "FLM_VERSION and NPU_VERSION"
        )
    use_hrx = backend == "hrx"
    build_dir = paths.FASTFLOWLM_BUILD if use_hrx else paths.FASTFLOWLM_XRT_BUILD
    install_dir = paths.FASTFLOWLM_INSTALL if use_hrx else paths.FASTFLOWLM_XRT_INSTALL
    env = environment.cmake_env()
    dependencies = _require_fastflowlm_deps(env)
    host_xrt = None
    xrt_include = None
    if (
        use_hrx
        and not paths.is_windows()
        and not environment.have_header("libdrm/drm.h")
    ):
        sys.exit(
            "missing libdrm/drm.h: FastFlowLM's amdxdna headers require "
            "libdrm (Ubuntu: sudo apt-get install libdrm-dev; "
            "Fedora: sudo dnf install libdrm-devel)"
        )
    if not use_hrx:
        host_xrt = _host_xrt_prefix()
        if host_xrt is None:
            sys.exit(_missing_host_xrt_message())
        xrt_include = _host_xrt_include_dir(host_xrt)
        if xrt_include is None:
            sys.exit(
                "missing XRT headers (xrt/xrt_bo.h and "
                "xrt/experimental/xrt_ext.h): the runtime prefix at "
                f"{host_xrt} is not a full SDK. Install matching XRT "
                "development headers or set XILINX_XRT to an install that "
                "contains include/xrt/"
            )
        print(f"  host XRT: {host_xrt}")
        print(f"  XRT headers: {xrt_include}")
        # Prefer a real xrt.pc next to the runtime prefix. Do not synthesize a
        # stub (that invented Version: 2.0) and do not put distro pkg-config
        # first: CMake is given -DXRT_INCLUDE_DIR / -DXRT_LIB_DIR instead.
        host_pc_dir = _host_xrt_pkgconfig_dir(host_xrt)
        if host_pc_dir is not None:
            env = dict(env)
            env["PKG_CONFIG_PATH"] = os.pathsep.join(
                entry
                for entry in (str(host_pc_dir), env.get("PKG_CONFIG_PATH", ""))
                if entry
            )
    patches.apply_patches({"FastFlowLM"})
    fresh = getattr(args, "fresh", False)
    if fresh:
        patches.validate_declared_patch_state("FastFlowLM", paths.PROJ / "FastFlowLM")
        _prepare_fresh_cmake_tree(build_dir, build_dir)
        # The in-source FastFlowLM/src/build tree is the HRX helper's CMake
        # scratch dir. Do not wipe it for an XRT --fresh: the backends already
        # use separate out-of-tree build dirs.
        if use_hrx:
            shutil.rmtree(source / "build", ignore_errors=True)
    backend_flags: list[str] = []
    if use_hrx:
        hrx_config = paths.HRX_INSTALL / "lib" / "cmake" / "hrx" / "hrx-config.cmake"
        if not hrx_config.exists():
            sys.exit(
                f"missing installed HRX package ({hrx_config}) - "
                "run: python dev.py build hrx"
            )
        backend_flags = [
            "-DFLM_USE_HRX=ON",
            f"-DCMAKE_PREFIX_PATH={environment.cmake_path(paths.HRX_INSTALL)}",
        ]
    else:
        # XRT diagnostic build: no HRX prerequisite. Link the host XRT runtime
        # and the matching development headers. Do not enable FLM_PORTABLE_BUILD:
        # that bundles libxrt into the prefix and is the wrong product for a
        # host-XRT comparison. Explicit -DXRT_* dirs satisfy CMake; a stub
        # xrt.pc is not written.
        backend_flags = [
            "-DFLM_USE_HRX=OFF",
            f"-DXRT_INCLUDE_DIR={environment.cmake_path(xrt_include)}",
            f"-DXRT_LIB_DIR={environment.cmake_path(_host_xrt_lib_dir(host_xrt))}",
        ]
    print(f"  FLM_VERSION={flm_version} NPU_VERSION={npu_version} (CMakePresets.json)")
    cc, cxx = environment.compilers()
    environment.sh(
        [
            environment.cmake_bin(),
            *(["--fresh"] if fresh else []),
            "-G",
            "Ninja",
            "-S",
            environment.cmake_path(source),
            "-B",
            environment.cmake_path(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            *backend_flags,
            "-DCMAKE_INSTALL_LIBDIR=lib",
            f"-DFLM_VERSION={flm_version}",
            f"-DNPU_VERSION={npu_version}",
            f"-DCMAKE_TOOLCHAIN_FILE={environment.cmake_path(Path(dependencies['toolchain']))}",
            f"-DVCPKG_TARGET_TRIPLET={dependencies['triplet']}",
            f"-DVCPKG_INSTALLED_DIR={environment.cmake_path(Path(dependencies['install_root']))}",
            f"-DVCPKG_MANIFEST_DIR={environment.cmake_path(paths.ROOT / '.github' / 'vcpkg' / 'fastflowlm')}",
            "-DVCPKG_MANIFEST_INSTALL=OFF",
            f"-DCMAKE_INSTALL_PREFIX={environment.cmake_path(install_dir)}",
            f"-DCMAKE_XCLBIN_PREFIX={environment.cmake_path(source)}",
            *environment.cmake_compiler_flags(cc, cxx),
        ],
        env=env,
    )
    environment.sh(
        [
            environment.cmake_bin(),
            "--build",
            environment.cmake_path(build_dir),
            *(["--clean-first"] if fresh else []),
            "-j",
            BUILD_JOBS,
        ],
        env=env,
    )
    shutil.rmtree(install_dir, ignore_errors=True)
    environment.sh(
        [
            environment.cmake_bin(),
            "--install",
            environment.cmake_path(build_dir),
        ],
        env=env,
    )
    installed = install_dir / "bin" / paths.executable_name("flm")
    if not installed.exists():
        sys.exit(f"FastFlowLM install did not produce {installed}")
    if not use_hrx and not paths.is_windows():
        # FHS XRT does not bundle vcpkg SONAMEs (that was portable-build
        # territory). Point RUNPATH at the vcpkg lib dir used to link flm so
        # the diagnostic binary can start without a fake XRT prefix.
        from build_tools.native_dependencies import require_patchelf, set_elf_rpath

        vcpkg_lib = Path(dependencies["triplet_root"]) / "lib"
        rpath_entries = ["$ORIGIN/../lib"]
        if vcpkg_lib.is_dir():
            rpath_entries.append(str(vcpkg_lib.resolve()))
        set_elf_rpath(installed, ":".join(rpath_entries), patchelf=require_patchelf())
    if fresh:
        marker: dict = {
            "source": git_source_state(paths.PROJ / "FastFlowLM"),
            "install": directory_state(
                install_dir,
                exclude_names=(FRESH_BUILD_MARKER,),
            ),
        }
        if use_hrx:
            marker["hrx"] = json.loads(
                (paths.HRX_INSTALL / FRESH_BUILD_MARKER).read_text(encoding="utf-8")
            )
        (install_dir / FRESH_BUILD_MARKER).write_text(
            json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"build fastflowlm ({backend}): ok ->", installed)
