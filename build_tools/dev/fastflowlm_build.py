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

BUILD_JOBS = os.environ.get("RNS_BUILD_JOBS") or str(os.cpu_count() or 8)
FRESH_BUILD_MARKER = ".rns-fresh-build.json"


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


def _finalize_portable_wrapper(install_prefix):
    """Activate the portable wrapper as the entrypoint (`flm`).

    A portable FastFlowLM install lays the real binary and its bundled-runtime
    wrapper flat at the prefix root (``flm`` + ``flm-wrapper.sh``); the wrapper
    is what sets ``XILINX_XRT``/``LD_LIBRARY_PATH`` before exec'ing the real
    binary. Upstream's packaging step performs this rename, so we mirror it here
    so that invoking ``<prefix>/flm`` runs the wrapper (matching how the HRX
    build exposes ``bin/flm``).
    """
    wrapper = install_prefix / "flm-wrapper.sh"
    if not wrapper.exists():
        return
    real = install_prefix / "flm"
    if real.exists():
        real.replace(install_prefix / "flm-real")
    wrapper.replace(real)
    real.chmod(0o755)


def _index_shared_libraries(directory):
    """Map every shared-library filename in ``directory`` to its path."""
    index = {}
    if directory.is_dir():
        for path in sorted(directory.glob("*.so*")):
            index.setdefault(path.name, path)
    return index


def _stage_xdna_plugin(lib_dir):
    """Copy the XRT XDNA driver plugin into ``lib_dir`` when discoverable.

    XRT loads the device driver plugin from ``$XILINX_XRT/lib``; the portable
    wrapper points ``XILINX_XRT`` at the bundle root, so staging the plugin (and
    its version symlink) here lets the tree drive the NPU without a separate
    host XRT install. Best-effort: probe ``$XILINX_XRT`` then the conventional
    ``/opt/xilinx/xrt``. Returns the staged real plugin, or ``None`` if none is
    found (the host XRT runtime must then provide it).
    """
    from build_tools.native_dependencies import require_patchelf, set_elf_rpath

    search_roots = []
    env_root = os.environ.get("XILINX_XRT")
    if env_root:
        search_roots.append(Path(env_root))
    search_roots.append(Path("/opt/xilinx/xrt"))
    for base in search_roots:
        source_lib = base / "lib"
        matches = (
            sorted(source_lib.glob("libxrt_driver_xdna.so*"))
            if source_lib.is_dir()
            else []
        )
        if not matches:
            continue
        real = None
        for match in matches:
            destination = lib_dir / match.name
            if destination.exists() or destination.is_symlink():
                destination.unlink()
            if match.is_symlink():
                destination.symlink_to(os.path.basename(os.readlink(match)))
            else:
                shutil.copy2(match, destination)
                set_elf_rpath(destination, "$ORIGIN", patchelf=require_patchelf())
                real = destination
        return real or (lib_dir / matches[-1].name)
    return None


def _mirror_xrt_multiarch(lib_dir):
    """Mirror bundled ``libxrt*.so*`` into ``lib/x86_64-linux-gnu/``.

    XRT constructs driver/plugin paths as ``$XILINX_XRT/lib/x86_64-linux-gnu/``
    and the portable wrapper points ``XILINX_XRT`` at the bundle root.
    Pre-populating this directory at build time (rather than leaning on the
    wrapper's guarded first-run mirror, which is skipped if the directory
    already exists) keeps the tree correct and self-contained across reruns and
    ensures the staged XDNA driver plugin is discoverable.
    """
    multiarch = lib_dir / "x86_64-linux-gnu"
    multiarch.mkdir(parents=True, exist_ok=True)
    for lib in sorted(lib_dir.glob("libxrt*.so*")):
        link = multiarch / lib.name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(Path("..") / lib.name)


def _stage_xrt_runtime_closure(install_dir, *, root):
    """Make the portable XRT tree self-contained.

    Upstream's portable CMake install bundles ``libxrt`` plus a best-effort set
    of third-party libraries, but it can copy *system* versions that mismatch
    the exact ``DT_NEEDED`` strings the ``flm`` binary was linked against (which
    come from the vcpkg toolchain, e.g. boost 1.91 vs a system 1.83). Walk the
    real ELF closure and copy any missing, non-system dependency from the vcpkg
    install tree into ``<prefix>/lib`` under its exact needed name; then
    best-effort stage the XRT XDNA driver plugin. System libraries built by the
    same host toolchain (libstdc++, libgcc_s, ...) are intentionally left to the
    host loader.
    """
    from build_tools.fastflowlm_artifact import _is_linux_system_runtime
    from build_tools.native_dependencies import (
        elf_needed,
        is_elf,
        require_patchelf,
        set_elf_rpath,
    )

    patchelf = require_patchelf()
    lib_dir = install_dir / "lib"
    lib_dir.mkdir(exist_ok=True)
    vcpkg_lib = Path(dependency_paths(root)["triplet_root"]) / "lib"
    candidates = _index_shared_libraries(vcpkg_lib)
    provided = {path.name for path in lib_dir.glob("*.so*")}
    queue = [install_dir / "flm-real", *sorted(lib_dir.glob("*.so*"))]
    staged = []
    while queue:
        path = queue.pop(0)
        if not path.is_file() or path.is_symlink() or not is_elf(path):
            continue
        for needed in elf_needed(path, patchelf=patchelf):
            if needed in provided or _is_linux_system_runtime(needed):
                continue
            source = candidates.get(needed)
            if source is None:
                # A genuine host/system library (libstdc++, libgcc_s, ...): the
                # XRT build is only ever run on a host with a matching toolchain,
                # so leave these to the system loader rather than bundling.
                continue
            destination = lib_dir / needed
            shutil.copy2(source, destination)
            set_elf_rpath(destination, "$ORIGIN", patchelf=patchelf)
            provided.add(needed)
            staged.append(needed)
            queue.append(destination)
    plugin = _stage_xdna_plugin(lib_dir)
    _mirror_xrt_multiarch(lib_dir)
    summary = f"staged {len(staged)} vcpkg dependency(ies)"
    summary += (
        f"; XDNA plugin {plugin.name}" if plugin else "; no XDNA plugin found"
    )
    print(f"  XRT runtime closure: {summary}")


def cmd_build_fastflowlm(args):
    _build_fastflowlm(args, backend="hrx")


def cmd_build_fastflowlm_xrt(args):
    _build_fastflowlm(args, backend="xrt")


def _build_fastflowlm(args, *, backend):
    """Configure, build and install FastFlowLM for the selected NPU backend.

    ``backend="hrx"`` (default) is the shipped product: it links the HRX
    amdxdna runtime and installs an FHS ``bin/flm`` tree consumed by the wheels.
    ``backend="xrt"`` is a diagnostic-only comparison build (``FLM_USE_HRX=OFF``
    + ``FLM_PORTABLE_BUILD=ON``): it links XRT, bundles its runtime libraries,
    installs a flat portable tree, and is never packaged into the HRX wheels.
    The two use entirely separate CMake/build/install trees so they never share
    state.
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
    use_hrx = backend == "hrx"
    build_dir = paths.FASTFLOWLM_BUILD if use_hrx else paths.FASTFLOWLM_XRT_BUILD
    install_dir = (
        paths.FASTFLOWLM_INSTALL if use_hrx else paths.FASTFLOWLM_XRT_INSTALL
    )
    env = environment.cmake_env()
    dependencies = _require_fastflowlm_deps(env)
    if not paths.is_windows() and not environment.have_header("libdrm/drm.h"):
        sys.exit(
            "missing libdrm/drm.h: FastFlowLM's amdxdna headers require "
            "libdrm (Ubuntu: sudo apt-get install libdrm-dev; "
            "Fedora: sudo dnf install libdrm-devel)"
        )
    patches.apply_patches({"FastFlowLM"})
    fresh = getattr(args, "fresh", False)
    if fresh:
        patches.validate_declared_patch_state("FastFlowLM", paths.PROJ / "FastFlowLM")
        _prepare_fresh_cmake_tree(build_dir, build_dir)
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
        # XRT diagnostic build: no HRX prerequisite. FLM_PORTABLE_BUILD bundles
        # the XRT runtime libraries next to the binary; _stage_xrt_runtime_closure
        # then completes the tree with the vcpkg dependency closure and the XDNA
        # driver plugin (libxrt_driver_xdna.so.2) so it is self-contained and can
        # reach the device without a host XRT install.
        backend_flags = [
            "-DFLM_USE_HRX=OFF",
            "-DFLM_PORTABLE_BUILD=ON",
        ]
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
            "-DFLM_VERSION=0.9.45",
            "-DNPU_VERSION=32.0.203.304",
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
    if use_hrx:
        installed = install_dir / "bin" / paths.executable_name("flm")
    else:
        _finalize_portable_wrapper(install_dir)
        if not paths.is_windows():
            _stage_xrt_runtime_closure(install_dir, root=paths.ROOT)
        installed = install_dir / paths.executable_name("flm")
    if not installed.exists():
        sys.exit(f"FastFlowLM install did not produce {installed}")
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
