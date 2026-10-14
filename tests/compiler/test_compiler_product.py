# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import os
import platform
import struct
import subprocess
import sys
import sysconfig
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

pytestmark = [pytest.mark.compiler_stack, pytest.mark.presubmit]


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "tests" / "e2e"),
    str(ROOT / "tests" / "compiler"),
]

import run_e2e
import verify_compiler_bindings

from build_tools import (
    compiler_product,
    compiler_wheels,
    hrx_runtime,
    iron,
    npu_prereqs,
    npu_test_cases,
    npu_test_package,
    paths,
)
from build_tools.archive import extract_archive_atomic
from build_tools.artifact import sha256_file
from build_tools.dev import compiler_build as build


def write_file(path: Path, data: bytes = b"x", executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    if executable and os.name != "nt":
        path.chmod(0o755)


def make_product(
    root: Path,
    artifact: str = "fastflow-sdk-compiler-linux-x86_64",
    include_build_metadata: bool = False,
) -> Path:
    product = root / artifact
    required_tools = []
    for name in [
        *compiler_product.NPU_PREREQ_TOOLS,
        "aiebu-asm",
        "xclbinutil",
        "llvm-nm",
        "llvm-objcopy",
    ]:
        tool = f"{name}{compiler_product._EXE}"
        write_file(product / "bin" / tool, executable=True)
        required_tools.append(f"bin/{tool}")
    for relative in (
        "python/aie/ir.py",
        "include/aie_api/aie.hpp",
        "aie_runtime_lib/AIE2P/vec_math.h",
        "peano/llvm-aie/README.txt",
        "third_party/aie_api/README.txt",
        "lib/.keep",
        *(f"licenses/{name}" for name in compiler_product.COMPILER_PRODUCT_LICENSES),
    ):
        write_file(product / relative)
    if include_build_metadata:
        write_file(product / "python" / "CMakeFiles" / "target.dir" / "object.obj")

    files = []
    for path in sorted(p for p in product.rglob("*") if p.is_file()):
        files.append(
            {
                "path": path.relative_to(product).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    (product / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "product": "fastflow-sdk-compiler",
                "artifact": artifact,
                "required_tools": required_tools,
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    return product


def test_product_name_is_public_and_platform_normalized():
    assert (
        compiler_product._compiler_product_name("Linux", "AMD64")
        == "fastflow-sdk-compiler-linux-x86_64"
    )
    assert "staging" not in compiler_product._compiler_product_name("Windows", "x86_64")


def test_iron_tools_build_does_not_resolve_virtualenv_python(tmp_path):
    root = tmp_path
    venv_python = root / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_bytes(b"")

    bdir = root / "iron-tools"
    install_dir = root / "iron-tools-install"
    original_resolve = Path.resolve

    def reject_python_resolve(path, *args, **kwargs):
        if path == venv_python:
            raise AssertionError("virtualenv Python path must not be resolved")
        return original_resolve(path, *args, **kwargs)

    with (
        mock.patch.object(paths, "BUILD", root),
        mock.patch.object(build.patches, "eudsl_cache_ok", return_value=True),
        mock.patch.object(
            build.environment,
            "compilers",
            return_value=("cc", "c++"),
        ),
        mock.patch.object(build.environment, "cmake_compiler_flags", return_value=[]),
        mock.patch.object(build.environment, "cmake_ccache_flags", return_value=[]),
        mock.patch.object(build.environment, "cmake_env", return_value={}),
        mock.patch.object(build.environment, "cmake_bin", return_value="cmake"),
        mock.patch.object(build.environment, "sh") as run,
        mock.patch.object(Path, "resolve", reject_python_resolve),
    ):
        build.build_iron_tools(venv_python, bdir, install_dir)

    configure_command = run.call_args_list[0].args[0]
    assert (
        f"-DPython3_EXECUTABLE={build.environment.cmake_path(venv_python.absolute())}"
        in configure_command
    )
    assert (
        f"-DPython_EXECUTABLE={build.environment.cmake_path(venv_python.absolute())}"
        in configure_command
    )
    assert "-DMLIR_ENABLE_PYTHON_STABLE_ABI=ON" in configure_command
    assert "-DAIE_ENABLE_PYTHON_PASSES=OFF" in configure_command
    assert "-DAIE_ENABLE_XRT_PYTHON_BINDINGS=OFF" in configure_command


def test_binding_conformance_discovers_abi3_extension_modules():
    aie_root = Path("site-packages") / "aie"
    assert (
        verify_compiler_bindings._extension_module_name(
            aie_root,
            aie_root / "_mlir_libs" / "_mlir.pyd",
        )
        == "aie._mlir_libs._mlir"
    )
    assert (
        verify_compiler_bindings._extension_module_name(
            aie_root,
            aie_root / "_mlir_libs" / "_aie.abi3.so",
        )
        == "aie._mlir_libs._aie"
    )


def test_wheel_verification_runs_binding_conformance_with_current_minor():
    with mock.patch.object(compiler_wheels.subprocess, "run") as run:
        compiler_wheels._verify_installed_compiler_bindings(Path("venv-python"))

    command = run.call_args.args[0]
    assert command[0] == Path("venv-python")
    assert str(command[1]).endswith("verify_compiler_bindings.py")
    assert command[-1] == f"{sys.version_info.major}.{sys.version_info.minor}"


def test_install_compiler_wheels_installs_core_and_bindings(tmp_path):
    core = tmp_path / "fastflow_sdk_compiler_core-0.1.0-py3-none-any.whl"
    bindings = tmp_path / "fastflow_sdk_compiler_bindings-0.1.0-py3-none-any.whl"
    meta = tmp_path / "fastflow_sdk_compiler-0.1.0-py3-none-any.whl"
    ops = tmp_path / "fastflow_sdk_ops-0.1.0-py3-none-any.whl"
    for wheel in (core, bindings, meta, ops):
        wheel.write_bytes(b"wheel")
    args = SimpleNamespace(dir=str(tmp_path))
    with (
        mock.patch.object(
            compiler_wheels.environment, "venv_py", return_value="python"
        ),
        mock.patch.object(compiler_wheels.environment, "python_install") as install,
    ):
        compiler_wheels.cmd_install_compiler_wheels(args)

    wheels_call, requirements_call = install.call_args_list
    assert wheels_call.args[:3] == ("python", "--no-index", "--no-deps")
    assert wheels_call.args[3:] == (str(core), str(bindings))
    assert str(meta) not in wheels_call.args
    assert str(ops) not in wheels_call.args
    assert requirements_call.args == (
        "python",
        *compiler_wheels.COMPILER_BINDINGS_RUNTIME_REQUIREMENTS,
    )


def test_install_compiler_wheels_requires_exactly_one_distribution_wheel(tmp_path):
    args = SimpleNamespace(dir=str(tmp_path))
    with pytest.raises(SystemExit, match="expected one wheel per distribution"):
        compiler_wheels.cmd_install_compiler_wheels(args)


def test_consumer_only_verification_skips_static_wheel_scans():
    args = SimpleNamespace(consumer_only=True)
    with (
        mock.patch.object(
            compiler_wheels, "_verify_compiler_wheel_consumer"
        ) as consumer,
        mock.patch.object(compiler_wheels, "_compiler_wheel_files") as static_scan,
    ):
        compiler_wheels.cmd_verify_compiler_wheels(args)

    consumer.assert_called_once_with(args)
    static_scan.assert_not_called()


def test_consumer_root_exports_only_cross_minor_compiler_packages(tmp_path):
    root = tmp_path
    site_packages = root / "site-packages"
    write_file(site_packages / "aie" / "__init__.py", b"")
    write_file(site_packages / "aie" / "__pycache__" / "bad.cpython-312.pyc")
    write_file(site_packages / "fastflow_sdk_compiler" / "__init__.py", b"")
    write_file(site_packages / "fastflow_sdk_compiler_core-1.0.dist-info" / "METADATA")
    write_file(site_packages / "numpy" / "__init__.py", b"")
    destination = root / "build" / "consumer"
    with (
        mock.patch.object(compiler_wheels, "BUILD", root / "build"),
        mock.patch.object(
            compiler_wheels,
            "_compiler_site_packages",
            return_value=site_packages,
        ),
    ):
        compiler_wheels._export_compiler_consumer_root(Path("python"), destination)

    assert (destination / "aie" / "__init__.py").is_file()
    assert (destination / "fastflow_sdk_compiler" / "__init__.py").is_file()
    assert not (destination / "aie" / "__pycache__").exists()
    assert not (destination / "numpy").exists()


def test_compiler_site_packages_uses_platform_purelib():
    assert compiler_wheels._compiler_site_packages(Path(sys.executable)) == Path(
        sysconfig.get_path("purelib")
    )


def test_latest_minor_consumer_can_run_one_e2e(tmp_path):
    root = tmp_path
    consumer_root = root / "consumer"
    write_file(consumer_root / "aie" / "__init__.py", b"")
    write_file(consumer_root / "fastflow_sdk_compiler" / "__init__.py", b"")
    args = SimpleNamespace(
        consumer_root=str(consumer_root),
        consumer_e2e=True,
        venv=str(root / "build" / "venv"),
    )
    with (
        mock.patch.object(compiler_wheels, "BUILD", root / "build"),
        mock.patch.object(compiler_wheels.environment, "python_install"),
        mock.patch.object(compiler_wheels, "_verify_installed_compiler_bindings"),
        mock.patch.object(compiler_wheels.subprocess, "run") as run,
    ):
        compiler_wheels._verify_compiler_wheel_consumer(args)

    e2e = run.call_args_list[-1]
    assert str(e2e.args[0][1]).endswith("run_e2e.py")
    assert e2e.kwargs["env"]["FASTFLOW_SDK_COMPILER_PREFIX"] == str(
        consumer_root / "fastflow_sdk_compiler"
    )
    assert e2e.kwargs["env"]["FASTFLOW_SDK_CMAKE"] == compiler_wheels.cmake_bin()


def test_compiler_product_can_extract_to_explicit_short_root(tmp_path):
    short_root = tmp_path / "rnc"
    archive = tmp_path / "compiler.zip"
    with (
        mock.patch.object(
            compiler_product,
            "_find_compiler_product_archive",
            return_value=archive,
        ),
        mock.patch.object(
            compiler_product,
            "extract_archive_atomic",
            return_value=short_root / "fastflow-sdk-compiler-windows-x86_64",
        ) as extract,
    ):
        compiler_product.cmd_unpack_compiler_product(
            SimpleNamespace(dir=tmp_path, output_dir=str(short_root))
        )

    extract.assert_called_once_with(
        archive,
        short_root,
        allowed_root=short_root.parent,
        verify_root=compiler_product.verify_product_inventory,
    )


def test_inventory_accepts_complete_product(tmp_path):
    product = make_product(tmp_path)
    manifest = compiler_product.verify_product_inventory(product)
    assert manifest["product"] == "fastflow-sdk-compiler"


def test_inventory_rejects_staging_artifact_name(tmp_path):
    product = make_product(tmp_path, "fastflow-sdk-staging-compiler-linux-x86_64")
    with pytest.raises(RuntimeError, match="must not include staging"):
        compiler_product.verify_product_inventory(product)


def test_inventory_rejects_build_system_metadata(tmp_path):
    product = make_product(tmp_path, include_build_metadata=True)
    with pytest.raises(RuntimeError, match="build-system metadata"):
        compiler_product.verify_product_inventory(product)


def test_runtime_staging_excludes_build_intermediates(tmp_path):
    root = tmp_path
    source = root / "source"
    write_file(source / "lib" / "runtime.lib")
    write_file(source / "lib" / "crt0.o")
    write_file(source / "include" / "runtime.h")
    write_file(source / "CMakeFiles" / "target.dir" / "runtime.obj")
    write_file(source / "lib" / "objects-Release" / "obj.AIECAPI" / "Dialects.cpp.o")
    write_file(source / "lib" / "stray.pdb")
    write_file(source / "__pycache__" / "module.cpython-312.pyc")
    write_file(
        source
        / "runtime_lib"
        / "xaiengineTmp"
        / "x86_64"
        / "tmp"
        / "xaiengine-cache-Release.cmake"
    )

    destination = root / "destination"
    compiler_product._stage_runtime_tree(source, destination)

    assert (destination / "lib" / "runtime.lib").is_file()
    assert (destination / "lib" / "crt0.o").is_file()
    assert (destination / "include" / "runtime.h").is_file()
    assert not (destination / "CMakeFiles").exists()
    assert not (destination / "lib" / "objects-Release").exists()
    assert not (destination / "lib" / "stray.pdb").exists()
    assert not (destination / "__pycache__").exists()
    assert not (destination / "runtime_lib" / "xaiengineTmp").exists()


def test_ci_runs_one_deep_doctor_and_uses_content_addressed_caches():
    workflows = ROOT / ".github" / "workflows"
    linux = (workflows / "build_fastflow_sdk_linux.yml").read_text()
    windows = (workflows / "build_fastflow_sdk_windows.yml").read_text()

    assert "dev.py doctor compiler-stack --deep" in linux
    assert "dev.py doctor compiler-stack --deep" not in windows
    assert "key: ccache-${{ runner.os }}-${{ github.run_id }}" not in linux + windows
    assert (
        "key: vcpkg-${{ runner.os }}-${{ hashFiles("
        "'.github/vcpkg/triplets/*.cmake') }}-${{ github.run_id }}" not in windows
    )
    cache_inputs = (
        "hashFiles('manifests/versions.toml', 'patches/**/*.patch', "
        "'dev.py', 'build_tools/**/*.py')"
    )
    assert cache_inputs in linux
    assert cache_inputs in windows


def test_linux_ci_installs_aiebu_static_link_dependencies():
    workflow = (
        ROOT / ".github" / "workflows" / "build_fastflow_sdk_linux.yml"
    ).read_text()

    assert "--enablerepo=powertools" in workflow
    assert "glibc-static" in workflow


def test_ci_builds_hrx_unit_tests_without_installing_test_artifacts():
    workflows = ROOT / ".github" / "workflows"
    linux = (workflows / "build_fastflow_sdk_linux.yml").read_text()
    windows = (workflows / "build_fastflow_sdk_windows.yml").read_text()
    runtime_build = (ROOT / "build_tools" / "dev" / "runtime_build.py").read_text()

    assert "dev.py build all --include-tests" in linux
    assert "dev.py build all --include-tests" in windows
    assert '"-DHRX_INSTALL_TESTS=OFF"' in runtime_build
    assert '"HrxPublicDist"' in runtime_build
    assert "libhrx_tools_hrx-info" in runtime_build
    assert "LIBHRX_BUILD_TOOLS" not in runtime_build


def test_windows_ci_retries_transient_chocolatey_llvm_lookup_failures():
    workflow = (
        ROOT / ".github" / "workflows" / "build_fastflow_sdk_windows.yml"
    ).read_text()

    assert "*NuGetResolverInputException*" in workflow
    assert '*"503 (Service Unavailable)"*' in workflow
    assert "Transient Chocolatey LLVM lookup failure" in workflow


def test_npu_prerequisites_exclude_separate_compiler_product(tmp_path):
    root = tmp_path
    hrx_install = root / "hrx-install"
    write_file(hrx_install / ("bin/hrx.dll" if os.name == "nt" else "lib/libhrx.so"))
    write_file(hrx_install / "include/hrx/hrx_runtime.h")
    write_file(hrx_install / "include/hrx/hrx_amdxdna.h")
    cts_source = root / "cts"
    for name in hrx_runtime.HRX_CTS_TESTS:
        write_file(cts_source / f"{name}{compiler_product._EXE}", executable=True)

    stage = root / "stage"
    npu_prereqs._stage_npu_runtime_prereqs(stage, hrx_install, cts_source)

    assert (stage / "build/hrx-install/include/hrx/hrx_runtime.h").is_file()
    cts_stage = stage.joinpath(
        "projects", "hrx", "build", "cmake", *hrx_runtime.HRX_CTS_DIR
    )
    for name in hrx_runtime.HRX_CTS_TESTS:
        assert (cts_stage / f"{name}{compiler_product._EXE}").is_file()
    assert not (stage / "build/iron-tools").exists()
    assert not (stage / "tools-bin").exists()


def test_npu_test_package_contains_only_xadx_and_numeric_cases(tmp_path):
    stage = tmp_path
    case_records = []
    names = (
        "vector_scalar_add",
        "vector_scalar_add_chain4",
        "add_relu_fused",
        "elementwise_add",
        "relu",
        "gemv",
        "strided_copy",
    )
    for name in names:
        source = {"kind": "test", "name": name}
        if name == "vector_scalar_add_chain4":
            source.update(name="vector_scalar_add", run_count=4)
        case_records.append(
            npu_test_package.write_case(
                stage,
                npu_test_cases.generate_case(name),
                b"\x00\x00\x00\x00XADX",
                source,
                required_features=(
                    (npu_test_package.FULL_ELF_LOAD_PDI_FEATURE,)
                    if name in ("add_relu_fused", "strided_copy")
                    else ()
                ),
            )
        )

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    system = platform.system().lower()
    npu_test_package.write_manifest(stage, case_records, revision, system, "x86_64")
    manifest = npu_test_package.verify_tree(
        stage,
        names,
        revision,
        system,
    )

    assert len(manifest["cases"]) == 7
    fused = next(case for case in manifest["cases"] if case["name"] == "add_relu_fused")
    assert fused["required_features"] == [npu_test_package.FULL_ELF_LOAD_PDI_FEATURE]
    assert [buffer["name"] for buffer in fused["buffers"]] == [
        "inputs",
        "output",
        "scratch",
    ]
    dynamic = next(case for case in manifest["cases"] if case["name"] == "strided_copy")
    assert [dispatch["name"] for dispatch in dynamic["dispatches"]] == [
        "input_offset_0",
        "input_offset_1024",
    ]
    assert [
        (stage / dispatch["constants"]).read_bytes()
        for dispatch in dynamic["dispatches"]
    ] == [
        (0).to_bytes(4, "little", signed=True),
        (1024).to_bytes(4, "little", signed=True),
    ]
    generated = npu_test_cases.generate_case("strided_copy")
    assert [
        (stage / dispatch["expected"]["output"]).read_bytes()
        for dispatch in dynamic["dispatches"]
    ] == [
        dispatch["expected"]["output"].tobytes() for dispatch in generated["dispatches"]
    ]
    assert dynamic["required_features"] == [npu_test_package.FULL_ELF_LOAD_PDI_FEATURE]
    packaged = {path.suffix for path in stage.rglob("*") if path.is_file()}
    assert ".xclbin" not in packaged
    assert ".mlir" not in packaged


def test_windows_tar_zst_is_deterministic_and_extracts_valid_files(tmp_path):
    root = tmp_path
    product = make_product(root, artifact="fastflow-sdk-compiler-windows-x86_64")
    with mock.patch.object(compiler_product, "IS_WINDOWS", True):
        first = compiler_product._create_compiler_product_archive(
            product, root / "first"
        )
        second = compiler_product._create_compiler_product_archive(
            product, root / "second"
        )

    assert first.name.endswith(".tar.zst")
    assert first.read_bytes() == second.read_bytes()
    assert first.with_name(first.name + ".sha256").is_file()
    extracted = extract_archive_atomic(
        first,
        root / "extracted",
        allowed_root=root,
        verify_root=compiler_product.verify_product_inventory,
    )
    assert (extracted / "python" / "aie" / "ir.py").read_bytes() == b"x"
    compiler_product.verify_product_inventory(extracted)


def test_product_smoke_reports_python_extension_import_failure():
    aieopt_result = mock.Mock(returncode=0, stderr=b"")
    import_result = mock.Mock(
        returncode=1, stderr="ImportError: DLL filename is too long"
    )
    with (
        mock.patch.object(
            compiler_product, "verify_product_inventory", return_value={"files": []}
        ),
        mock.patch.object(compiler_product, "_compiler_product_env", return_value={}),
        mock.patch.object(compiler_product, "sh"),
        mock.patch.object(
            compiler_product.subprocess,
            "run",
            side_effect=[aieopt_result, import_result],
        ) as run,
        pytest.raises(RuntimeError, match="DLL filename is too long"),
    ):
        compiler_product._smoke_compiler_product(Path("compiler"))

    assert (
        run.call_args_list[1].args[0][-1] == "import aie.dialects.transform.interpreter"
    )


def test_windows_compiler_product_stages_non_system_dll_closure(tmp_path):
    root = tmp_path
    product = root / "product"
    dependency_root = root / "openssl"
    system_root = root / "windows"
    write_file(product / "bin" / "bootgen.exe", b"MZbootgen")
    write_file(
        product / "python" / "aie" / "_mlir_libs" / "nanobind-abi3-mlir.dll",
        b"MZnanobind",
    )
    write_file(
        product / "python" / "aie" / "_mlir_libs" / "MLIRPythonSupport-mlir.dll",
        b"MZsupport",
    )
    write_file(
        dependency_root / "bin" / "libcrypto-3-x64.dll",
        b"MZcrypto",
    )
    write_file(
        system_root / "System32" / "kernel32.dll",
        b"MZsystem",
    )
    imports = {
        "bootgen.exe": ["libcrypto-3-x64.dll"],
        "libcrypto-3-x64.dll": ["kernel32.dll"],
        "nanobind-abi3-mlir.dll": ["python3.dll"],
        "MLIRPythonSupport-mlir.dll": ["python3.dll"],
    }
    with (
        mock.patch.object(compiler_product, "IS_WINDOWS", True),
        mock.patch.dict(
            os.environ,
            {
                "OPENSSL_ROOT_DIR": str(dependency_root),
                "SystemRoot": str(system_root),
            },
            clear=False,
        ),
        mock.patch.object(
            compiler_product,
            "pe_imports",
            side_effect=lambda path: imports[Path(path).name],
        ),
    ):
        copied = compiler_product._stage_compiler_native_dependencies(product)
    assert (product / "bin" / "libcrypto-3-x64.dll").is_file()
    assert "libcrypto-3-x64.dll" in copied
    assert not (product / "bin" / "python3.dll").exists()


def test_windows_compiler_product_rejects_versioned_cpython_runtime(tmp_path):
    product = tmp_path / "product"
    support = product / "python" / "aie" / "_mlir_libs" / "MLIRPythonSupport-mlir.dll"
    write_file(support, b"MZsupport")
    with (
        mock.patch.object(compiler_product, "IS_WINDOWS", True),
        mock.patch.object(
            compiler_product,
            "pe_imports",
            return_value=["python312.dll"],
        ),
        pytest.raises(RuntimeError, match="versioned CPython DLL"),
    ):
        compiler_product._stage_compiler_native_dependencies(product)


def test_windows_product_verifier_accepts_stable_abi_runtime(tmp_path):
    product = tmp_path / "product"
    support = product / "python" / "MLIRPythonSupport-mlir.dll"
    write_file(support, b"MZsupport")
    with (
        mock.patch.object(compiler_product, "IS_WINDOWS", True),
        mock.patch.object(compiler_product, "pe_imports", return_value=["python3.dll"]),
    ):
        graph = compiler_product._verify_compiler_native_dependencies(product)
    assert graph["python/MLIRPythonSupport-mlir.dll"] == ["python3.dll"]


def test_windows_product_verifier_rejects_versioned_cpython_runtime(tmp_path):
    product = tmp_path / "product"
    support = product / "python" / "MLIRPythonSupport-mlir.dll"
    write_file(support, b"MZsupport")
    with (
        mock.patch.object(compiler_product, "IS_WINDOWS", True),
        mock.patch.object(
            compiler_product,
            "pe_imports",
            return_value=["python312.dll"],
        ),
        pytest.raises(RuntimeError, match="versioned CPython DLL"),
    ):
        compiler_product._verify_compiler_native_dependencies(product)


def test_linux_compiler_product_bundles_non_manylinux_dependencies(tmp_path):
    root = tmp_path
    product = root / "product"
    openssl = root / "openssl"

    def write_elf(path):
        header = bytearray(120)
        header[:4] = b"\x7fELF"
        header[4] = 2
        header[5] = 1
        struct.pack_into("<H", header, 16, 3)
        struct.pack_into("<Q", header, 32, 64)
        struct.pack_into("<H", header, 52, 64)
        struct.pack_into("<H", header, 54, 56)
        struct.pack_into("<H", header, 56, 1)
        struct.pack_into("<I", header, 64, 2)
        write_file(path, bytes(header))

    write_elf(product / "bin" / "bootgen")
    write_elf(openssl / "lib" / "libssl.so.1.1")
    write_elf(openssl / "lib" / "libcrypto.so.1.1")
    write_file(
        openssl / "share" / "openssl" / "copyright",
        b"OpenSSL license",
    )
    needed = {
        "bootgen": ["libssl.so.1.1", "libz.so.1"],
        "libssl.so.1.1": ["libcrypto.so.1.1"],
        "libcrypto.so.1.1": ["libc.so.6"],
    }

    def run(command, **kwargs):
        path = Path(command[-1])
        output = "\n".join(needed[path.name])
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=output + "\n",
        )

    with (
        mock.patch.object(compiler_product, "IS_WINDOWS", False),
        mock.patch.dict(
            os.environ,
            {"OPENSSL_ROOT_DIR": str(openssl)},
            clear=False,
        ),
        mock.patch.object(
            compiler_product.shutil,
            "which",
            return_value="/usr/bin/patchelf",
        ),
        mock.patch(
            "build_tools.native_dependencies.subprocess.run",
            side_effect=run,
        ),
    ):
        copied = compiler_product._stage_compiler_native_dependencies(product)

    assert set(copied) == {"libssl.so.1.1", "libcrypto.so.1.1"}
    assert (product / "lib" / "libssl.so.1.1").is_file()
    assert (product / "lib" / "libcrypto.so.1.1").is_file()
    assert (product / "licenses" / "openssl-copyright").is_file()


@pytest.mark.skipif(os.name == "nt", reason="Linux archive format test")
def test_linux_archive_is_tar_gz_without_zstd(tmp_path):
    root = tmp_path
    product = make_product(root / "stage")
    with mock.patch("build_tools.fastflowlm_artifact.shutil.which", lambda name: None):
        archive = compiler_product._create_compiler_product_archive(
            product, root / "artifacts"
        )

    assert archive.name.endswith(".tar.gz")
    assert archive.with_name(archive.name + ".sha256").is_file()
    extracted = extract_archive_atomic(
        archive,
        root / "extracted",
        allowed_root=root,
        verify_root=compiler_product.verify_product_inventory,
    )
    compiler_product.verify_product_inventory(extracted)
    assert os.access(extracted / "bin" / "xclbinutil", os.X_OK)


def test_relocated_toolchain_env_uses_product_paths(tmp_path):
    prefix = tmp_path
    write_file(prefix / "bin" / f"aie-opt{run_e2e.EXE}", executable=True)
    write_file(prefix / "bin" / f"aiebu-asm{run_e2e.EXE}", executable=True)
    write_file(prefix / "bin" / f"xclbinutil{run_e2e.EXE}", executable=True)
    write_file(prefix / "python" / "aie" / "ir.py")
    write_file(prefix / "python" / "probe.py", b"VALUE = 1\n")
    (prefix / "peano" / "llvm-aie").mkdir(parents=True)
    (prefix / "third_party" / "aie_api").mkdir(parents=True)
    (prefix / "lib").mkdir()

    env = run_e2e.toolchain_env(prefix / "bin" / f"xclbinutil{run_e2e.EXE}", prefix)

    assert env["PYTHONPATH"] == str(prefix / "python")
    assert env["PEANO_INSTALL_DIR"] == str(prefix / "peano" / "llvm-aie")
    assert env["AIEOPT_DIR"] == str(prefix / "third_party" / "aie_api")
    assert env["FASTFLOW_SDK_COMPILER_PREFIX"] == str(prefix)
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    subprocess.run([sys.executable, "-c", "import probe"], env=env, check=True)
    assert not (prefix / "python" / "__pycache__").exists()
    if os.name == "nt":
        assert str(prefix / "lib") in env["PATH"].split(os.pathsep)
    else:
        assert env["LD_LIBRARY_PATH"].split(os.pathsep)[0] == str(prefix / "lib")


def test_relocated_toolchain_env_uses_installed_wheel_bindings(tmp_path):
    root = tmp_path
    site_packages = root / "site-packages"
    prefix = site_packages / "fastflow_sdk_compiler"
    write_file(prefix / "bin" / f"aie-opt{run_e2e.EXE}", executable=True)
    write_file(prefix / "bin" / f"aiebu-asm{run_e2e.EXE}", executable=True)
    write_file(prefix / "bin" / f"xclbinutil{run_e2e.EXE}", executable=True)
    write_file(site_packages / "aie" / "ir.py")
    (prefix / "peano" / "llvm-aie").mkdir(parents=True)
    (prefix / "third_party" / "aie_api").mkdir(parents=True)
    (prefix / "lib").mkdir()

    with mock.patch.object(
        iron,
        "installed_aie_python_root",
        return_value=site_packages,
    ):
        env = run_e2e.toolchain_env(prefix / "bin" / f"xclbinutil{run_e2e.EXE}", prefix)

    assert env["PYTHONPATH"] == str(site_packages)


def test_hrx_xclbinutil_source_state_uses_nested_git_when_present(
    tmp_path, monkeypatch
):
    source = tmp_path / "iron-tools" / "third_party" / "hrx-xclbinutil"
    source.mkdir(parents=True)
    (source / ".git").mkdir()
    nested = {
        "revision": "nestedsha",
        "dirty": False,
        "tracked_diff_sha256": "00" * 32,
        "changed_files": [],
    }

    def fake_git_source_state(path):
        assert path == source
        return nested

    monkeypatch.setattr(compiler_product, "PROJ", tmp_path)
    monkeypatch.setattr(compiler_product, "git_source_state", fake_git_source_state)
    assert compiler_product._hrx_xclbinutil_source_state() == nested


def test_hrx_xclbinutil_source_state_scopes_vendored_diff(tmp_path, monkeypatch):
    iron = tmp_path / "iron-tools"
    vendored = iron / "third_party" / "hrx-xclbinutil"
    vendored.mkdir(parents=True)
    whole_tree_hash = "ab" * 32
    scoped_hash = "cd" * 32
    iron_state = {
        "revision": "ironsha",
        "dirty": True,
        "tracked_diff_sha256": whole_tree_hash,
        "changed_files": [
            {"path": "python/requirements.txt", "sha256": "aa"},
            {"path": "third_party/hrx-xclbinutil/README", "sha256": "bb"},
        ],
    }

    def fake_git_source_state(path):
        assert path == iron
        return iron_state

    monkeypatch.setattr(compiler_product, "PROJ", tmp_path)
    monkeypatch.setattr(compiler_product, "git_source_state", fake_git_source_state)
    monkeypatch.setattr(compiler_product, "git_revision", lambda path: "ironsha")
    monkeypatch.setattr(
        compiler_product,
        "_git_tracked_diff_sha256",
        lambda repo, *pathspecs: scoped_hash,
    )

    state = compiler_product._hrx_xclbinutil_source_state()
    assert state["vendored_in"] == "iron-tools"
    assert state["path"] == "third_party/hrx-xclbinutil"
    assert state["revision"] == "ironsha"
    assert state["dirty"] is True
    assert state["changed_files"] == [
        {"path": "third_party/hrx-xclbinutil/README", "sha256": "bb"}
    ]
    assert state["tracked_diff_sha256"] == scoped_hash
    assert state["tracked_diff_sha256"] != whole_tree_hash


def test_hrx_xclbinutil_source_state_clean_when_vendored_tree_unchanged(
    tmp_path, monkeypatch
):
    iron = tmp_path / "iron-tools"
    (iron / "third_party" / "hrx-xclbinutil").mkdir(parents=True)
    empty_hash = "ee" * 32
    iron_state = {
        "revision": "ironsha",
        "dirty": True,
        "tracked_diff_sha256": "ff" * 32,
        "changed_files": [{"path": "python/requirements.txt", "sha256": "aa"}],
    }
    monkeypatch.setattr(compiler_product, "PROJ", tmp_path)
    monkeypatch.setattr(compiler_product, "git_source_state", lambda path: iron_state)
    monkeypatch.setattr(compiler_product, "git_revision", lambda path: "ironsha")
    monkeypatch.setattr(
        compiler_product,
        "_git_tracked_diff_sha256",
        lambda repo, *pathspecs: empty_hash,
    )

    state = compiler_product._hrx_xclbinutil_source_state()
    assert state["dirty"] is False
    assert state["changed_files"] == []
    assert state["tracked_diff_sha256"] == empty_hash
