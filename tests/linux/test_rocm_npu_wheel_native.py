# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import io
import subprocess
import tempfile
import unittest
import zipfile
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from unittest import mock

import dev
from build_tools.rocm_npu_linux_wheels import (
    FASTFLOWLM_DISTRIBUTION,
    FASTFLOWLM_PACKAGE,
    LINUX_WHEEL_TAG,
    RUNTIME_DISTRIBUTION,
    RUNTIME_PACKAGE,
    _copy_runtime_native_closure,
    _patch_and_verify_linux_products,
    build_wheel,
    find_rocm_npu_wheels,
    verify_wheels_do_not_have_text_path_references,
    wheel_tag,
)
from build_tools.rocm_npu_wheel_common import verify_wheel_checksums


class RocmNpuLinuxWheelNativeTest(unittest.TestCase):
    def test_linux_workflow_uses_verify_wheel_venv(self):
        workflow = (
            Path(__file__).resolve().parents[2] /
            ".github" / "workflows" / "fastflowlm_hrx_linux.yml"
        ).read_text(encoding="utf-8")
        venv = dev.FASTFLOWLM_WHEEL_VENV.relative_to(dev.ROOT).as_posix()
        expected_python = f"{venv}/bin/python"

        self.assertIn(
            f"run: {expected_python} dev.py npu-probe --runtime-package",
            workflow,
        )
        self.assertIn(
            f"run: {expected_python} dev.py test-fastflowlm --wheel-package "
            "--inference-iterations 5",
            workflow,
        )
        self.assertIn(
            "image: ghcr.io/rocm/therock_build_manylinux_x86_64@sha256:",
            workflow,
        )
        self.assertIn("--device=/dev/accel/accel0", workflow)
        self.assertIn("--ulimit memlock=-1:-1", workflow)
        self.assertNotIn("name: Install Python venv support", workflow)
        self.assertNotIn("python3-venv", workflow)
        self.assertNotIn("python3-pip", workflow)
        self.assertNotIn("rocm-npu-fastflowlm-wheel-venv", workflow)

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

            self.assertEqual(
                wheel.name,
                f"rocm_npu_runtime-0.0.0.dev1-{LINUX_WHEEL_TAG}.whl",
            )
            self.assertTrue(wheel.with_name(wheel.name + ".sha256").is_file())

            with zipfile.ZipFile(wheel) as archive:
                names = set(archive.namelist())
                dist_info = "rocm_npu_runtime-0.0.0.dev1.dist-info"
                self.assertIn("rocm_npu_runtime/__init__.py", names)
                self.assertIn(f"{dist_info}/METADATA", names)
                self.assertIn(f"{dist_info}/WHEEL", names)
                self.assertIn(f"{dist_info}/RECORD", names)
                metadata = archive.read(f"{dist_info}/METADATA").decode("utf-8")
                self.assertIn("Name: rocm-npu-runtime\n", metadata)
                wheel_metadata = archive.read(f"{dist_info}/WHEEL").decode("utf-8")
                self.assertIn(f"Tag: {LINUX_WHEEL_TAG}\n", wheel_metadata)
                record = archive.read(f"{dist_info}/RECORD").decode("utf-8")
                self.assertIn(f"{dist_info}/RECORD,,", record)
                mode = archive.getinfo(
                    "rocm_npu_runtime/__init__.py").external_attr >> 16
                self.assertEqual(mode & 0o777, 0o644)
                mode = archive.getinfo("rocm_npu_runtime/bin/tool").external_attr >> 16
                self.assertEqual(mode & 0o777, 0o755)

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
                self.assertEqual(
                    entry_points,
                    "[console_scripts]\nflm = rocm_npu_fastflowlm:main\n",
                )
                record = archive.read(f"{dist_info}/RECORD").decode("utf-8")
                self.assertIn(f"{dist_info}/entry_points.txt,sha256=", record)

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
            self.assertEqual(wheels[RUNTIME_DISTRIBUTION], runtime)
            self.assertEqual(wheels[FASTFLOWLM_DISTRIBUTION], fastflowlm)
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

            with self.assertRaisesRegex(
                RuntimeError,
                "wheel text payload contains local checkout/build paths",
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
            self.assertIn(
                "wheel path audit: 1 native member(s) contain local path strings",
                output.getvalue(),
            )

    def test_linux_wheel_tag(self):
        self.assertEqual(wheel_tag("Linux"), LINUX_WHEEL_TAG)
        with self.assertRaisesRegex(RuntimeError, "x86-64"):
            wheel_tag("Linux", "arm64")

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

            self.assertTrue((runtime_root / "lib" / "libdep.so").is_file())

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

            self.assertTrue(
                (runtime_root / "lib" / "libfftw3.so.3.6.9").is_file()
            )
            self.assertEqual(
                license_names,
                {"libfftw3.so.3", "libfftw3.so.3.6.9"},
            )

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
                with self.assertRaisesRegex(
                    RuntimeError,
                    "rocm_npu_runtime/lib/libhrx.so -> libdep.so",
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
                with self.assertRaisesRegex(
                    RuntimeError,
                    "rocm_npu_runtime/lib/libhrx.so -> libfftw3.so.3.6.9",
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

            self.assertEqual(
                graph[f"{FASTFLOWLM_PACKAGE}/bin/flm"],
                ["libhrx.so"],
            )

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


if __name__ == "__main__":
    unittest.main()
