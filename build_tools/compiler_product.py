# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Relocatable FastFlow SDK compiler product staging and verification."""

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from build_tools import environment, iron, paths
from build_tools.archive import (
    create_deterministic_tar_gz,
    create_deterministic_tar_zst,
    extract_archive_atomic,
)
from build_tools.artifact import (
    file_inventory,
    file_record,
    git_revision,
    git_source_state,
    reset_directory,
    sha256_sidecar_path,
    verify_sha256_sidecar,
    write_sha256_sidecar,
)
from build_tools.compiler_native import (
    linux_system_runtime,
    relocate_linux_compiler_rpaths,
    windows_system_runtime,
)
from build_tools.fastflowlm_artifact import verify_bundle
from build_tools.native_dependencies import (
    elf_needed,
    is_loadable_elf,
    is_pe,
    pe_imports,
)

ROOT = paths.ROOT
PROJ = paths.PROJ
BUILD = paths.BUILD
IRON_TOOLS_INSTALL = paths.IRON_TOOLS_INSTALL
COMPILER_PRODUCT_STAGE = paths.BUILD / "fastflow-sdk-compiler-product"
COMPILER_PRODUCT_EXTRACTED = paths.BUILD / "fastflow-sdk-compiler-extracted"
COMPILER_PRODUCT_RELOCATED = paths.BUILD / "fastflow-sdk-compiler-relocated"
COMPILER_ARTIFACTS = paths.BUILD / "artifacts"
COMPILER_PRODUCT_ZSTD_LEVEL = 6
COMPILER_PRODUCT_ZSTD_WINDOW_LOG = 27
PEANO_WHEEL = "llvm-aie==21.0.0.2026061501+acc2a72c"
PEANO_INDEX = "https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly"
NPU_PREREQ_TOOLS = [
    "aiecc",
    "aie-opt",
    "aie-translate",
    "bootgen",
    "target_model",
    "target_model_rtti",
    "register_database",
]
COMPILER_PRODUCT_LICENSES = (
    "fastflow-sdk-LICENSE",
    "iron-tools-LICENSE",
    "hrx-xclbinutil-LICENSE",
    "llvm-LICENSE.txt",
    "aie-api-LICENSE",
    "aie-runtime-LICENSE.txt",
    "bootgen-LICENSE",
    "aiebu-LICENSE",
    "aiebu-ELFIO-LICENSE.txt",
    "aiebu-cxxopts-LICENSE",
    "aiebu-zstd-LICENSE",
    "aiebu-aie-runtime-LICENSE.txt",
    "aiebu-boost-LICENSE_1_0.txt",
)
IS_WINDOWS = paths.is_windows()
_EXE = ".exe" if IS_WINDOWS else ""
sh = environment.sh
venv_py = environment.venv_py


def _tool(path):
    """`path`, or `path.exe` on Windows. Fails loudly rather than silently skipping."""
    for cand in (path, path.with_suffix(".exe")):
        if cand.exists():
            return cand
    sys.exit(f"missing from the build tree: {path}")


def _stage(src, dst):
    """Copy dereferenced: the archive must hold real files, never symlinks.

    MLIR's AddMLIRPython.cmake materializes the aie Python package as absolute symlinks
    into projects/llvm-project, projects/iron-tools and build/llvm. build/llvm is not
    shipped and the consumer checks out no llvm-project, so archiving the links produces
    an artifact that only works on the machine that built it. copytree/copy2 follow
    symlinks by default and raise on a dangling one."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, symlinks=False, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst, follow_symlinks=True)


def _git_tracked_diff_sha256(repo, *pathspecs):
    """SHA-256 of `git diff HEAD` scoped to optional pathspecs."""
    command = ["git", "-C", str(repo), "diff", "HEAD"]
    if pathspecs:
        command.extend(["--", *pathspecs])
    diff = subprocess.run(command, check=True, capture_output=True).stdout
    return hashlib.sha256(diff).hexdigest()


def _hrx_xclbinutil_source_state():
    """Provenance for hrx-xclbinutil: nested git repo, or vendored in iron-tools."""
    source = PROJ / "iron-tools" / "third_party" / "hrx-xclbinutil"
    if (source / ".git").exists():
        return git_source_state(source)
    iron = PROJ / "iron-tools"
    iron_state = git_source_state(iron)
    prefix = "third_party/hrx-xclbinutil/"
    changed_files = [
        item
        for item in iron_state["changed_files"]
        if item["path"].replace("\\", "/").startswith(prefix)
    ]
    return {
        "revision": git_revision(iron),
        "dirty": bool(changed_files),
        "vendored_in": "iron-tools",
        "path": "third_party/hrx-xclbinutil",
        "tracked_diff_sha256": _git_tracked_diff_sha256(
            iron, "third_party/hrx-xclbinutil"
        ),
        "changed_files": changed_files,
    }


def _platform_machine(machine=None):
    machine = (machine or platform.machine() or "unknown").lower()
    return "x86_64" if machine == "amd64" else machine


def _compiler_product_name(system=None, machine=None):
    return f"fastflow-sdk-compiler-{(system or platform.system()).lower()}-{_platform_machine(machine)}"


def _assert_no_symlinks(root):
    links = [p for p in [root, *root.rglob("*")] if p.is_symlink()]
    if links:
        raise RuntimeError(
            "compiler product contains symlinks:\n  "
            + "\n  ".join(str(p) for p in links[:20])
        )


_BUILD_METADATA_NAMES = {
    "__pycache__",
    "CMakeFiles",
    "CMakeCache.txt",
    "cmake_install.cmake",
    "CTestTestfile.cmake",
    "build.ninja",
    "rules.ninja",
    ".ninja_deps",
    ".ninja_log",
    "install_manifest.txt",
}
_BUILD_INTERMEDIATE_SUFFIXES = {".obj", ".pdb", ".ilk", ".pyc", ".pyo"}


def _is_build_metadata(path):
    path = Path(path)
    return (
        any(
            part == "CMakeFiles" or part.startswith("objects-") or part.endswith("Tmp")
            for part in path.parts
        )
        or path.name in _BUILD_METADATA_NAMES
        or path.suffix.lower() in _BUILD_INTERMEDIATE_SUFFIXES
    )


def _stage_runtime_tree(src, dst):
    """Copy a mixed runtime build tree without CMake/Ninja intermediates."""
    shutil.copytree(
        src,
        dst,
        symlinks=False,
        dirs_exist_ok=True,
        ignore=lambda _directory, names: [
            name for name in names if _is_build_metadata(Path(name))
        ],
    )


def _peano_prefix():
    peano = Path(sysconfig.get_paths()["purelib"]) / "llvm-aie"
    if not peano.is_dir():
        raise RuntimeError(
            f"Peano is not installed in this Python environment: {peano}\n"
            "run: python dev.py setup"
        )
    return peano


def _stage_compiler_openssl_license(product, copied):
    if not any(name.casefold().startswith(("libssl", "libcrypto")) for name in copied):
        return
    openssl_root = Path(os.environ.get("OPENSSL_ROOT_DIR", "/usr"))
    license_candidates = (
        openssl_root / "share" / "openssl" / "copyright",
        openssl_root.parent / "share" / "openssl" / "copyright",
        Path("/usr/share/licenses/openssl/LICENSE.txt"),
        Path("/usr/share/doc/openssl/copyright"),
    )
    license_path = next((path for path in license_candidates if path.is_file()), None)
    if license_path:
        _stage(
            license_path,
            product / "licenses" / "openssl-copyright",
        )


def _stage_compiler_linux_native_dependencies(product):
    """Bundles non-manylinux shared libraries required by the compiler product."""
    patchelf = shutil.which("patchelf")
    if not patchelf:
        raise RuntimeError("patchelf is required to close Linux compiler dependencies")
    roots = []
    openssl_root = os.environ.get("OPENSSL_ROOT_DIR")
    if openssl_root:
        root = Path(openssl_root)
        roots.extend((root, root / "lib", root / "lib64"))
    roots.extend(
        Path(path)
        for path in (
            "/usr/local/lib64",
            "/usr/local/lib",
            "/usr/lib64",
            "/usr/lib/x86_64-linux-gnu",
            "/lib64",
            "/lib/x86_64-linux-gnu",
        )
    )
    candidates = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.glob("*.so*"):
            if path.is_file():
                candidates.setdefault(path.name, path)

    provided = {
        path.name
        for path in product.rglob("*")
        if path.is_file() and is_loadable_elf(path)
    }
    copied = {}
    queue = [
        path for path in product.rglob("*") if path.is_file() and is_loadable_elf(path)
    ]
    while queue:
        path = queue.pop(0)
        needed_libraries = elf_needed(path, patchelf=patchelf)
        for needed in needed_libraries:
            if needed in provided or linux_system_runtime(needed):
                continue
            source = candidates.get(needed)
            if source is None:
                relative = path.relative_to(product).as_posix()
                raise RuntimeError(
                    "compiler product dependency is not available from standard "
                    f"Linux library roots: {relative} -> {needed}"
                )
            destination = product / "lib" / needed
            _stage(source, destination)
            provided.add(needed)
            copied[needed] = str(source)
            queue.append(destination)
    _stage_compiler_openssl_license(product, copied)
    return copied


def _stage_compiler_native_dependencies(product):
    """Closes non-system native imports required by the compiler product."""
    if not IS_WINDOWS:
        return _stage_compiler_linux_native_dependencies(product)

    candidate_roots = []
    for variable in ("OPENSSL_ROOT_DIR", "VCPKG_INSTALLED_DIR"):
        value = os.environ.get(variable)
        if value and Path(value).is_dir():
            candidate_roots.append(Path(value))
    candidates = {
        path.name.casefold(): path
        for root in candidate_roots
        for path in root.rglob("*.dll")
        if path.is_file()
    }
    provided = {
        path.name.casefold()
        for path in product.rglob("*")
        if path.is_file() and is_pe(path)
    }
    copied = {}
    queue = [path for path in product.rglob("*") if path.is_file() and is_pe(path)]
    while queue:
        path = queue.pop(0)
        for needed in pe_imports(path):
            folded = needed.casefold()
            if re.fullmatch(r"python3[0-9]{1,2}\.dll", folded):
                relative = path.relative_to(product).as_posix()
                raise RuntimeError(
                    "compiler product imports a versioned CPython DLL instead of "
                    f"the stable-ABI python3.dll: {relative} -> {needed}"
                )
            if folded in provided or windows_system_runtime(needed):
                continue
            source = candidates.get(folded)
            if source is None:
                relative = path.relative_to(product).as_posix()
                raise RuntimeError(
                    "compiler product dependency is not available from its "
                    f"configured native dependency roots: {relative} -> {needed}"
                )
            destination = product / "bin" / source.name
            _stage(source, destination)
            provided.add(folded)
            copied[needed] = str(source)
            queue.append(destination)

    _stage_compiler_openssl_license(product, copied)
    return copied


def _verify_compiler_native_dependencies(product):
    if not IS_WINDOWS:
        return {}

    binaries = [path for path in product.rglob("*") if path.is_file() and is_pe(path)]
    provided = {path.name.casefold() for path in binaries}
    graph = {}
    missing = []
    versioned_python_imports = []
    for path in binaries:
        imports = pe_imports(path)
        relative = path.relative_to(product).as_posix()
        graph[relative] = imports
        for name in imports:
            folded = name.casefold()
            if re.fullmatch(r"python3[0-9]{1,2}\.dll", folded):
                versioned_python_imports.append(f"{relative} -> {name}")
            if folded not in provided and not windows_system_runtime(name):
                missing.append(f"{relative} -> {name}")
    if versioned_python_imports:
        raise RuntimeError(
            "compiler product imports a versioned CPython DLL instead of "
            "the stable-ABI python3.dll:\n  "
            + "\n  ".join(sorted(versioned_python_imports))
        )
    if missing:
        raise RuntimeError(
            "compiler product Windows dependency closure is incomplete:\n  "
            + "\n  ".join(sorted(missing))
        )
    return graph


def _write_product_manifest(bundle, manifest):
    manifest_path = bundle / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()
    manifest = dict(manifest, files=file_inventory(bundle))
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # _create_compiler_product_archive verifies this inventory immediately
    # before archiving. Avoid hashing the full product twice back-to-back here.
    return manifest


def stage_product():
    it_build = BUILD / "iron-tools"
    it = IRON_TOOLS_INSTALL
    if not it.is_dir():
        raise RuntimeError(
            "iron-tools is not installed - run: python dev.py build iron-tools"
        )

    reset_directory(COMPILER_PRODUCT_STAGE, BUILD)
    product = COMPILER_PRODUCT_STAGE / _compiler_product_name()
    product.mkdir(parents=True)

    required_tools = []
    for tool_name in NPU_PREREQ_TOOLS:
        tool = _tool(it_build / "bin" / tool_name)
        relative = Path("bin") / tool.name
        _stage(tool, product / relative)
        required_tools.append(relative.as_posix())

    xclbinutil = _tool(ROOT / "tools-bin" / "xclbinutil")
    relative = Path("bin") / xclbinutil.name
    _stage(xclbinutil, product / relative)
    required_tools.append(relative.as_posix())

    # The IRON JIT external-kernel path needs LLVM's objcopy because GNU
    # binutils objcopy cannot parse AIEngine ELF objects. Iron-tools installs
    # it into its bin directory, so the relocated product must ship it too.
    objcopy = _tool(it / "bin" / "llvm-objcopy")
    relative = Path("bin") / objcopy.name
    _stage(objcopy, product / relative)
    required_tools.append(relative.as_posix())

    aiebu_asm = _tool(paths.AIEBU_INSTALL / "bin" / "aiebu-asm")
    relative = Path("bin") / aiebu_asm.name
    _stage(aiebu_asm, product / relative)
    required_tools.append(relative.as_posix())

    installed_python = it / "python"
    if not installed_python.is_dir():
        raise RuntimeError(f"missing from iron-tools install: {installed_python}")
    _stage_runtime_tree(installed_python, product / "python")
    installed_include = it / "include"
    if not installed_include.is_dir():
        raise RuntimeError(f"missing from iron-tools install: {installed_include}")
    _stage(installed_include, product / "include")
    build_include = it_build / "include"
    if not build_include.is_dir():
        raise RuntimeError(f"missing from iron-tools build: {build_include}")
    _stage_runtime_tree(build_include, product / "include")
    installed_lib = it / "lib"
    if not installed_lib.is_dir():
        raise RuntimeError(f"missing from iron-tools install: {installed_lib}")
    _stage_runtime_tree(installed_lib, product / "lib")
    # These data/header trees are generated directly into the configured build
    # prefix by upstream install rules. They contain runtime inputs, not CMake
    # object state; the Python package above must always come from the clean
    # `cmake --install` tree.
    for dirname in ("runtime_lib", "aie_runtime_lib"):
        source = it_build / dirname
        if not source.is_dir():
            raise RuntimeError(f"missing from iron-tools build: {source}")
        _stage_runtime_tree(source, product / dirname)

    aie_api = PROJ / "iron-tools" / "third_party" / "aie_api"
    if not aie_api.is_dir():
        raise RuntimeError(f"missing aie_api checkout: {aie_api}")
    _stage(aie_api, product / "third_party" / "aie_api")
    _stage(_peano_prefix(), product / "peano" / "llvm-aie")

    native_dependencies = _stage_compiler_native_dependencies(product)
    relocated_rpaths = {} if IS_WINDOWS else relocate_linux_compiler_rpaths(product)

    license_sources = (
        (ROOT / "LICENSE", "fastflow-sdk-LICENSE"),
        (PROJ / "iron-tools" / "LICENSE", "iron-tools-LICENSE"),
        # hrx-xclbinutil declares Apache-2.0 in its checked-in CMakeLists.txt,
        # but the copy vendored under mlir-aie's third_party does not carry a
        # standalone LICENSE. Stage this repository's identical Apache-2.0
        # terms under the component-specific name.
        (ROOT / "LICENSE", "hrx-xclbinutil-LICENSE"),
        (PROJ / "llvm-project" / "llvm" / "LICENSE.TXT", "llvm-LICENSE.txt"),
        (
            PROJ / "iron-tools" / "third_party" / "aie_api" / "LICENSE",
            "aie-api-LICENSE",
        ),
        (
            PROJ / "iron-tools" / "third_party" / "aie-rt" / "license.txt",
            "aie-runtime-LICENSE.txt",
        ),
        (
            PROJ / "iron-tools" / "third_party" / "bootgen" / "LICENSE",
            "bootgen-LICENSE",
        ),
        (PROJ / "aiebu" / "LICENSE", "aiebu-LICENSE"),
        (
            PROJ / "aiebu" / "src" / "cpp" / "ELFIO" / "LICENSE.txt",
            "aiebu-ELFIO-LICENSE.txt",
        ),
        (
            PROJ / "aiebu" / "src" / "cpp" / "cxxopts" / "LICENSE",
            "aiebu-cxxopts-LICENSE",
        ),
        (
            PROJ / "aiebu" / "src" / "cpp" / "zstd" / "LICENSE",
            "aiebu-zstd-LICENSE",
        ),
        (
            PROJ / "aiebu" / "src" / "cpp" / "aie-rt" / "license.txt",
            "aiebu-aie-runtime-LICENSE.txt",
        ),
        (
            PROJ / "iron-tools" / "LICENSES" / "BSL-1.0.txt",
            "aiebu-boost-LICENSE_1_0.txt",
        ),
    )
    for source, name in license_sources:
        if source.is_file():
            _stage(source, product / "licenses" / name)

    _assert_no_symlinks(product)
    manifest = _write_product_manifest(
        product,
        {
            "schema_version": 1,
            "product": "fastflow-sdk-compiler",
            "artifact": product.name,
            "platform": {"system": platform.system(), "machine": platform.machine()},
            "required_tools": required_tools,
            "peano": {
                "wheel": PEANO_WHEEL,
                "index": PEANO_INDEX,
                "path": "peano/llvm-aie",
            },
            "sources": {
                "fastflow-sdk": git_source_state(ROOT),
                "llvm-project": git_source_state(PROJ / "llvm-project"),
                "aiebu": git_source_state(PROJ / "aiebu"),
                "iron-tools": git_source_state(PROJ / "iron-tools"),
                "hrx-xclbinutil": _hrx_xclbinutil_source_state(),
            },
            "dependencies": {
                "requirements": [
                    file_record(ROOT, ROOT / "requirements" / "compiler-stack.txt"),
                    file_record(ROOT, ROOT / "requirements" / "compiler-stack.lock"),
                ],
                "native_runtime_files": native_dependencies,
                "relocated_elf_rpaths": relocated_rpaths,
            },
            "toolchain": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "build_image": os.environ.get("RNS_BUILD_IMAGE"),
                "runner_os": os.environ.get("RUNNER_OS"),
                "runner_arch": os.environ.get("RUNNER_ARCH"),
            },
        },
    )
    return product, manifest


def resolve_product_root(location):
    location = Path(location)
    if (location / "manifest.json").is_file():
        return location
    if not location.is_dir():
        raise RuntimeError(f"compiler product directory not found: {location}")
    candidates = [
        p for p in location.iterdir() if p.is_dir() and (p / "manifest.json").is_file()
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one compiler product under {location}")
    return candidates[0]


def _create_compiler_product_archive(product, output_directory):
    verify_product_inventory(product)
    output_directory.mkdir(parents=True, exist_ok=True)
    archive = (
        output_directory / f"{product.name}{'.tar.zst' if IS_WINDOWS else '.tar.gz'}"
    )
    for stale in output_directory.glob(f"{product.name}.*"):
        if stale != archive and stale != sha256_sidecar_path(archive):
            stale.unlink()
    if IS_WINDOWS:
        create_deterministic_tar_zst(
            product,
            archive,
            arcname=product.name,
            level=COMPILER_PRODUCT_ZSTD_LEVEL,
            compression_parameters={
                "enable_ldm": True,
                "window_log": COMPILER_PRODUCT_ZSTD_WINDOW_LOG,
                "threads": -1,
                "write_checksum": True,
            },
        )
    else:
        create_deterministic_tar_gz(
            product, archive, arcname=product.name, compresslevel=6
        )
    write_sha256_sidecar(archive)
    return archive


def _find_compiler_product_archive(directory):
    directory = Path(directory)
    archives = sorted(
        [
            *directory.glob("fastflow-sdk-compiler-*.zip"),
            *directory.glob("fastflow-sdk-compiler-*.tar.gz"),
            *directory.glob("fastflow-sdk-compiler-*.tar.zst"),
        ]
    )
    if len(archives) != 1:
        raise RuntimeError(
            f"expected exactly one fastflow-sdk-compiler artifact in {directory}, "
            f"found {len(archives)}"
        )
    archive = archives[0]
    verify_sha256_sidecar(archive)
    return archive


def verify_product_inventory(product):
    manifest = verify_bundle(product)
    if manifest.get("product") != "fastflow-sdk-compiler":
        raise RuntimeError(f"not a fastflow-sdk-compiler product: {product}")
    artifact = str(manifest.get("artifact", ""))
    if "staging" in artifact:
        raise RuntimeError(
            f"compiler product artifact name must not include staging: {artifact}"
        )
    build_metadata = [
        record["path"]
        for record in manifest.get("files", [])
        if _is_build_metadata(record["path"])
    ]
    if build_metadata:
        raise RuntimeError(
            f"compiler product contains build-system metadata: {build_metadata[0]}"
        )
    _assert_no_symlinks(product)

    required_files = [
        "python/aie/ir.py",
        "include/aie_api/aie.hpp",
        "aie_runtime_lib/AIE2P/vec_math.h",
        *manifest.get("required_tools", []),
    ]
    for relative in required_files:
        if not (product / relative).is_file():
            raise RuntimeError(f"compiler product is missing file: {relative}")
    for relative in ("peano/llvm-aie", "third_party/aie_api", "lib"):
        if not (product / relative).is_dir():
            raise RuntimeError(f"compiler product is missing directory: {relative}")
    for name in COMPILER_PRODUCT_LICENSES:
        if not (product / "licenses" / name).is_file():
            raise RuntimeError(f"compiler product is missing license: {name}")
    _verify_compiler_native_dependencies(product)
    return manifest


def _compiler_product_env(prefix):
    try:
        return iron.toolchain_env(iron.require_xclbinutil(prefix), prefix)
    except iron.E2EError as e:
        raise RuntimeError(str(e)) from e


def _smoke_compiler_product(prefix):
    manifest = verify_product_inventory(prefix)
    env = _compiler_product_env(prefix)
    aieopt = prefix / "bin" / f"aie-opt{_EXE}"
    sh([aieopt, "--version"], env=env)
    p = subprocess.run(
        [str(aieopt)],
        input=b"module { aie.device(npu2) {} }\n",
        env=env,
        capture_output=True,
    )
    if p.returncode != 0:
        detail = p.stderr.decode(errors="replace")[:1000]
        raise RuntimeError(f"aie-opt relocation smoke failed: {detail}")
    p = subprocess.run(
        [str(venv_py()), "-c", "import aie.dialects.transform.interpreter"],
        env=env,
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(
            f"compiler Python extension import failed: {p.stderr[-1000:]}"
        )
    return manifest


def _relocate_compiler_product(product):
    reset_directory(COMPILER_PRODUCT_RELOCATED, BUILD)
    relocated = COMPILER_PRODUCT_RELOCATED / product.name
    _stage(product, relocated)
    _smoke_compiler_product(relocated)
    env = dict(os.environ, FASTFLOW_SDK_COMPILER_PREFIX=str(relocated))
    rc = sh(
        [
            venv_py(),
            ROOT / "tests" / "e2e" / "run_e2e.py",
            "--compiler-prefix",
            relocated,
        ],
        env=env,
        check=False,
    ).returncode
    if rc != 0:
        raise RuntimeError(f"relocated compiler product e2e failed: {relocated}")
    return relocated


def cmd_package_compiler_product(args):
    """Stage and archive the source-built compiler product under a relocatable prefix."""
    try:
        product, manifest = stage_product()
        archive = _create_compiler_product_archive(product, COMPILER_ARTIFACTS)
    except RuntimeError as error:
        sys.exit(str(error))
    print("package compiler product: ok")
    print("  product: ", product)
    print("  archive: ", archive)
    print(f"  files:   {len(manifest['files'])}")


def cmd_unpack_compiler_product(args):
    try:
        archive = _find_compiler_product_archive(args.dir)
        output_directory = Path(args.output_dir)
        product = extract_archive_atomic(
            archive,
            output_directory,
            allowed_root=output_directory.parent,
            verify_root=verify_product_inventory,
        )
    except RuntimeError as error:
        sys.exit(str(error))
    print("unpack compiler product: ok ->", product)


def cmd_verify_compiler_product(args):
    try:
        product = resolve_product_root(args.dir)
        manifest = _smoke_compiler_product(product)
        if args.relocate:
            relocated = _relocate_compiler_product(product)
            print("  relocated:", relocated)
    except RuntimeError as error:
        sys.exit(str(error))
    print(f"verify compiler product: ok ({len(manifest['files'])} files) -> {product}")
