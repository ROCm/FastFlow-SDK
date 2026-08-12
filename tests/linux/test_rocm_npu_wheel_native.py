# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import io
import subprocess
import sys
import tempfile
import types
import zipfile
from contextlib import contextmanager, redirect_stdout
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from unittest import mock

import pytest

from build_tools import fastflowlm_package
from build_tools import paths
from build_tools.rocm_npu_linux_wheels import (
    FASTFLOWLM_DISTRIBUTION,
    FASTFLOWLM_LINUX_WHEEL_TAG,
    FASTFLOWLM_PACKAGE,
    LINUX_WHEEL_TAG,
    RUNTIME_DISTRIBUTION,
    RUNTIME_PACKAGE,
    _auditwheel_policy_diagnostics,
    _copy_hrx_soname,
    _copy_runtime_native_closure,
    _patch_and_verify_linux_products,
    _write_runtime_module,
    auditwheel_policy,
    build_wheel,
    find_rocm_npu_wheels,
    fastflowlm_wheel_tag,
    validate_auditwheel_policy,
    verify_wheels_do_not_have_text_path_references,
    wheel_tag,
)
from build_tools.rocm_npu_wheel_common import verify_wheel_checksums
from build_tools.rocm_npu_wheel_common import COMPILER_BINDINGS_DISTRIBUTION


pytestmark = [
    pytest.mark.compiler_stack,
    pytest.mark.fastflowlm,
    pytest.mark.linux,
]


class TestRocmNpuLinuxWheelNative:
    def test_linux_workflow_uses_verify_wheel_venv(self):
        workflow = (
            Path(__file__).resolve().parents[2] /
            ".github" / "workflows" / "fastflowlm_hrx_linux.yml"
        ).read_text(encoding="utf-8")
        venv = fastflowlm_package.FASTFLOWLM_WHEEL_VENV.relative_to(
            paths.ROOT
        ).as_posix()
        expected_python = f"{venv}/bin/python"

        assert (
            f"run: {expected_python} dev.py npu-probe --runtime-package"
            in workflow
        )
        assert f"{venv}/bin/rocm-npu qualify run" in workflow
        assert "--fast" in workflow
        assert "--model-cache build/FastFlowLM-models" in workflow
        assert "test-fastflowlm --wheel-package" not in workflow
        assert (
            "image: ghcr.io/rocm/therock_build_manylinux_x86_64@sha256:"
            in workflow
        )
        assert "--device=/dev/accel/accel0" in workflow
        assert "--ulimit memlock=-1:-1" in workflow
        assert "name: Install Python venv support" not in workflow
        assert "python3-venv" not in workflow
        assert "python3-pip" not in workflow
        assert "rocm-npu-fastflowlm-wheel-venv" not in workflow

    def test_runtime_launcher_preserves_venv_scripts_directory(self, tmp_path):
        package = tmp_path / "rocm_npu_runtime"
        _write_runtime_module(package)

        source = (package / "__init__.py").read_text(encoding="utf-8")
        assert "scripts = str(Path(sys.executable).parent)" in source
        assert "Path(sys.executable).resolve()" not in source

    def test_build_wheel_writes_metadata_record_and_checksum(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tree = root / "tree"
            package = tree / "rocm_npu_runtime"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
            executable = package / "bin" / "tool"
            executable.parent.mkdir()
            executable.write_text("#!/bin/sh\n", encoding="utf-8")

            wheel = build_wheel(
                tree,
                RUNTIME_DISTRIBUTION,
                "0.0.0.dev1",
                root / "dist",
                summary="test wheel",
                tag=LINUX_WHEEL_TAG,
            )

            assert (
                wheel.name
                == f"rocm_npu_runtime-0.0.0.dev1-{LINUX_WHEEL_TAG}.whl"
            )
            assert wheel.with_name(wheel.name + ".sha256").is_file()

            with zipfile.ZipFile(wheel) as archive:
                names = set(archive.namelist())
                dist_info = "rocm_npu_runtime-0.0.0.dev1.dist-info"
                assert "rocm_npu_runtime/__init__.py" in names
                assert f"{dist_info}/METADATA" in names
                assert f"{dist_info}/WHEEL" in names
                assert f"{dist_info}/RECORD" in names
                metadata = archive.read(f"{dist_info}/METADATA").decode("utf-8")
                assert "Name: rocm-npu-runtime\n" in metadata
                wheel_metadata = archive.read(f"{dist_info}/WHEEL").decode("utf-8")
                assert f"Tag: {LINUX_WHEEL_TAG}\n" in wheel_metadata
                record = archive.read(f"{dist_info}/RECORD").decode("utf-8")
                assert f"{dist_info}/RECORD,," in record
                mode = archive.getinfo(
                    "rocm_npu_runtime/__init__.py").external_attr >> 16
                assert mode & 0o777 == 0o644
                mode = archive.getinfo("rocm_npu_runtime/bin/tool").external_attr >> 16
                assert mode & 0o777 == 0o755

    def test_build_wheel_writes_console_script_entry_points(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wheel = build_wheel(
                self._package_tree(root, "rocm_npu_fastflowlm"),
                FASTFLOWLM_DISTRIBUTION,
                "0.0.0.dev1",
                root / "dist",
                summary="fastflowlm",
                console_scripts={"flm": "rocm_npu_fastflowlm:main"},
                tag=LINUX_WHEEL_TAG,
            )

            with zipfile.ZipFile(wheel) as archive:
                dist_info = "rocm_npu_fastflowlm-0.0.0.dev1.dist-info"
                entry_points = archive.read(
                    f"{dist_info}/entry_points.txt").decode("utf-8")
                assert (
                    entry_points
                    == "[console_scripts]\nflm = rocm_npu_fastflowlm:main\n"
                )
                record = archive.read(f"{dist_info}/RECORD").decode("utf-8")
                assert f"{dist_info}/entry_points.txt,sha256=" in record

    def test_find_rocm_npu_wheels_requires_one_of_each_product(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = build_wheel(
                self._package_tree(root, "rocm_npu_runtime"),
                RUNTIME_DISTRIBUTION,
                "0.0.0.dev1",
                root / "dist",
                summary="runtime",
                tag=LINUX_WHEEL_TAG,
            )
            fastflowlm = build_wheel(
                self._package_tree(root, "rocm_npu_fastflowlm"),
                FASTFLOWLM_DISTRIBUTION,
                "0.0.0.dev1",
                root / "dist",
                summary="fastflowlm",
                requires=("rocm-npu-runtime ==0.0.0.dev1",),
                tag=LINUX_WHEEL_TAG,
            )

            wheels = find_rocm_npu_wheels(root / "dist")
            assert wheels[RUNTIME_DISTRIBUTION] == runtime
            assert wheels[FASTFLOWLM_DISTRIBUTION] == fastflowlm
            verify_wheel_checksums(wheels)

    def test_path_reference_check_fails_on_text_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "tree" / "rocm_npu_runtime"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text(
                f"PATH = {root!s}\n", encoding="utf-8")
            wheel = build_wheel(
                root / "tree",
                RUNTIME_DISTRIBUTION,
                "0.0.0.dev1",
                root / "dist",
                summary="runtime",
                tag=LINUX_WHEEL_TAG,
            )

            with pytest.raises(
                RuntimeError,
                match="wheel text payload contains local checkout/build paths",
            ):
                verify_wheels_do_not_have_text_path_references(
                    {RUNTIME_DISTRIBUTION: wheel}, (root,))

    def test_path_reference_check_warns_on_elf_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "tree" / "rocm_npu_runtime" / "lib"
            package.mkdir(parents=True)
            (package / "libpayload.so").write_bytes(
                b"\x7fELF" + bytes(str(root), "utf-8"))
            wheel = build_wheel(
                root / "tree",
                RUNTIME_DISTRIBUTION,
                "0.0.0.dev1",
                root / "dist",
                summary="runtime",
                tag=LINUX_WHEEL_TAG,
            )

            output = io.StringIO()
            with redirect_stdout(output):
                verify_wheels_do_not_have_text_path_references(
                    {RUNTIME_DISTRIBUTION: wheel}, (root,))
            assert (
                "wheel path audit: 1 native member(s) contain local path strings"
                in output.getvalue()
            )

    def test_linux_wheel_tag(self):
        assert wheel_tag("Linux") == LINUX_WHEEL_TAG
        assert fastflowlm_wheel_tag("Linux") == FASTFLOWLM_LINUX_WHEEL_TAG
        with pytest.raises(RuntimeError, match="x86-64"):
            wheel_tag("Linux", "arm64")

    def test_auditwheel_policy_matches_product_tags(self):
        assert (
            validate_auditwheel_policy(
                RUNTIME_DISTRIBUTION,
                "manylinux_2_27_x86_64",
            )
            == "manylinux_2_27_x86_64"
        )
        assert (
            validate_auditwheel_policy(
                FASTFLOWLM_DISTRIBUTION,
                "linux_x86_64",
            )
            == "linux_x86_64"
        )
        assert (
            validate_auditwheel_policy(
                COMPILER_BINDINGS_DISTRIBUTION,
                "manylinux_2_28_x86_64",
            )
            == "manylinux_2_28_x86_64"
        )
        with pytest.raises(RuntimeError, match="not manylinux_2_28"):
            validate_auditwheel_policy(
                RUNTIME_DISTRIBUTION,
                "linux_x86_64",
            )

    def test_auditwheel_policy_diagnostics_identify_limiting_constraints(self):
        class Policy:
            def __init__(self, name, priority, symbol_versions=None):
                self.name = name
                self.priority = priority
                self.symbol_versions = symbol_versions or {}

            def __lt__(self, other):
                return self.priority < other.priority

        linux = Policy("linux_x86_64", 0)
        target = Policy(
            "manylinux_2_28_x86_64",
            10,
            {"GLIBC": frozenset({"2.28"})},
        )
        highest = Policy("manylinux_2_39_x86_64", 20)
        policies = types.SimpleNamespace(
            get_policy_by_name=lambda name: {
                target.name: target,
            }[name]
        )
        wheel_info = types.SimpleNamespace(
            policies=policies,
            overall_policy=linux,
            ref_policy=linux,
            sym_policy=linux,
            ucs_policy=highest,
            pyfpe_policy=highest,
            blacklist_policy=highest,
            machine_policy=target,
            external_refs={
                target.name: types.SimpleNamespace(
                    libs={"libmissing.so.1": None},
                    blacklist={"libz.so.1": ["inflate"]},
                )
            },
            versioned_symbols={"libc.so.6": {"GLIBC_2.28", "GLIBC_2.34"}},
        )

        diagnostics = _auditwheel_policy_diagnostics(wheel_info, target.name)

        assert "external references: linux_x86_64 (LIMITING)" in diagnostics
        assert "versioned symbols: linux_x86_64 (LIMITING)" in diagnostics
        assert "machine ISA: manylinux_2_28_x86_64 (ok)" in diagnostics
        assert "libmissing.so.1: <unresolved>" in diagnostics
        assert "libz.so.1: inflate" in diagnostics
        assert "GLIBC_2.34" in diagnostics
        assert "    GLIBC_2.28" not in diagnostics

    def test_auditwheel_policy_uses_pinned_structured_analyzer(self):
        wheel = Path("example-py3-none-linux_x86_64.whl")
        analyze = mock.Mock(
            return_value=types.SimpleNamespace(
                overall_policy=types.SimpleNamespace(name="linux_x86_64")
            )
        )
        auditwheel = types.ModuleType("auditwheel")
        auditwheel.__path__ = []
        error = types.ModuleType("auditwheel.error")
        error.NonPlatformWheelError = type("NonPlatformWheelError", (Exception,), {})
        error.WheelToolsError = type("WheelToolsError", (Exception,), {})
        wheel_abi = types.ModuleType("auditwheel.wheel_abi")
        wheel_abi.analyze_wheel_abi = analyze
        wheeltools = types.ModuleType("auditwheel.wheeltools")
        wheeltools.get_wheel_architecture = mock.Mock(return_value="x86_64")
        wheeltools.get_wheel_libc = mock.Mock(return_value="glibc")
        modules = {
            "auditwheel": auditwheel,
            "auditwheel.error": error,
            "auditwheel.wheel_abi": wheel_abi,
            "auditwheel.wheeltools": wheeltools,
        }
        with (
            mock.patch(
                "build_tools.rocm_npu_linux_wheels.distribution_version",
                return_value="6.7.0",
            ),
            mock.patch.dict(sys.modules, modules),
        ):
            assert (
                auditwheel_policy(
                    wheel, exclude=frozenset({"libAIEAggregateCAPI.so"})
                )
                == "linux_x86_64"
            )
        analyze.assert_called_once_with(
            "glibc",
            "x86_64",
            wheel,
            frozenset({"libAIEAggregateCAPI.so"}),
            disable_isa_ext_check=False,
            allow_graft=False,
        )

    def test_auditwheel_policy_rejects_unpinned_api_version(self):
        with mock.patch(
            "build_tools.rocm_npu_linux_wheels.distribution_version",
            return_value="6.8.0",
        ):
            with pytest.raises(RuntimeError, match="expected 6.7.0"):
                auditwheel_policy(Path("example.whl"))

    def test_auditwheel_policy_requires_pinned_tool(self):
        with mock.patch(
            "build_tools.rocm_npu_linux_wheels.distribution_version",
            side_effect=PackageNotFoundError,
        ):
            with pytest.raises(
                RuntimeError, match="auditwheel 6.7.0 is required"
            ):
                auditwheel_policy(Path("example.whl"))

    def test_runtime_native_closure_copies_needed_bundle_libraries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "bundle"
            runtime_root = root / RUNTIME_PACKAGE
            self._write_elf(source / "lib" / "libhrx.so")
            self._write_elf(source / "lib" / "libdep.so")
            self._write_elf(runtime_root / "lib" / "libhrx.so")

            with self._mock_patchelf(
                needed={"libhrx.so": ["libdep.so"], "libdep.so": []},
            ):
                _copy_runtime_native_closure(source, runtime_root)

            assert (runtime_root / "lib" / "libdep.so").is_file()

    def test_runtime_stages_only_hrx_soname(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "bundle"
            runtime_root = root / RUNTIME_PACKAGE
            for name in ("libhrx.so", "libhrx.so.0", "libhrx.so.0.1.0"):
                self._write_elf(source / "lib" / name)

            with self._mock_patchelf(
                needed={},
                sonames={
                    "libhrx.so": "libhrx.so.0",
                    "libhrx.so.0": "libhrx.so.0",
                    "libhrx.so.0.1.0": "libhrx.so.0",
                },
            ):
                soname = _copy_hrx_soname(source, runtime_root)

            assert soname == "libhrx.so.0"
            assert [
                path.name for path in (runtime_root / "lib").iterdir()
            ] == ["libhrx.so.0"]

    def test_runtime_native_closure_copies_needed_soname_filename(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "bundle"
            runtime_root = root / RUNTIME_PACKAGE
            self._write_elf(source / "lib" / "libfftw3.so.3")
            self._write_elf(runtime_root / "lib" / "libhrx.so")

            with self._mock_patchelf(
                needed={
                    "libhrx.so": ["libfftw3.so.3.6.9"],
                    "libfftw3.so.3": [],
                    "libfftw3.so.3.6.9": [],
                },
                sonames={"libfftw3.so.3": "libfftw3.so.3.6.9"},
            ):
                license_names = _copy_runtime_native_closure(source, runtime_root)

            assert (runtime_root / "lib" / "libfftw3.so.3.6.9").is_file()
            assert license_names == {"libfftw3.so.3", "libfftw3.so.3.6.9"}

    def test_linux_dependency_closure_is_checked_per_product(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime_root = root / RUNTIME_PACKAGE
            fastflowlm_root = root / FASTFLOWLM_PACKAGE
            self._write_elf(runtime_root / "lib" / "libhrx.so")
            self._write_elf(fastflowlm_root / "lib" / "libdep.so")

            with self._mock_patchelf(
                needed={"libhrx.so": ["libdep.so"], "libdep.so": []},
            ):
                with pytest.raises(
                    RuntimeError,
                    match="rocm_npu_runtime/lib/libhrx.so -> libdep.so",
                ):
                    _patch_and_verify_linux_products([runtime_root, fastflowlm_root])

    def test_linux_dependency_closure_requires_needed_filename(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime_root = root / RUNTIME_PACKAGE
            self._write_elf(runtime_root / "lib" / "libhrx.so")
            self._write_elf(runtime_root / "lib" / "libfftw3.so.3")

            with self._mock_patchelf(
                needed={
                    "libhrx.so": ["libfftw3.so.3.6.9"],
                    "libfftw3.so.3": [],
                },
                sonames={"libfftw3.so.3": "libfftw3.so.3.6.9"},
            ):
                with pytest.raises(
                    RuntimeError,
                    match=(
                        "rocm_npu_runtime/lib/libhrx.so -> libfftw3.so.3.6.9"
                    ),
                ):
                    _patch_and_verify_linux_products([runtime_root])

    def test_fastflowlm_product_can_depend_on_runtime_product(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime_root = root / RUNTIME_PACKAGE
            fastflowlm_root = root / FASTFLOWLM_PACKAGE
            self._write_elf(runtime_root / "lib" / "libhrx.so")
            self._write_elf(fastflowlm_root / "bin" / "flm")

            with self._mock_patchelf(
                needed={"libhrx.so": [], "flm": ["libhrx.so"]},
            ):
                graph = _patch_and_verify_linux_products([runtime_root, fastflowlm_root])

            assert graph[f"{FASTFLOWLM_PACKAGE}/bin/flm"] == ["libhrx.so"]

    def _package_tree(self, root, name):
        tree = root / f"tree-{name}"
        package = tree / name
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        return tree

    def _write_elf(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x7fELFfake")

    @contextmanager
    def _mock_patchelf(self, *, needed, sonames=None):
        sonames = sonames or {}

        def fake_which(name):
            return "/bin/patchelf" if name == "patchelf" else None

        def fake_run(command, **kwargs):
            normalized = [Path(command[0]).name, *[str(arg) for arg in command[1:]]]
            if normalized[1] == "--set-rpath":
                return subprocess.CompletedProcess(command, 0)
            path = Path(normalized[2])
            if normalized[1] == "--print-rpath":
                return subprocess.CompletedProcess(command, 0, stdout="$ORIGIN\n")
            if normalized[1] == "--print-soname":
                return subprocess.CompletedProcess(
                    command, 0, stdout=f"{sonames.get(path.name, path.name)}\n")
            if normalized[1] == "--print-needed":
                output = "\n".join(needed.get(path.name, []))
                return subprocess.CompletedProcess(
                    command, 0, stdout=(output + "\n") if output else "")
            raise AssertionError(f"unexpected command: {command!r}")

        with mock.patch("build_tools.rocm_npu_linux_wheels.shutil.which", fake_which), \
             mock.patch("build_tools.rocm_npu_linux_wheels.subprocess.run", fake_run):
            yield
