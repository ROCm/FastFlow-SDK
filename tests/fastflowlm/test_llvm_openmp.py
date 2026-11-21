# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
from pathlib import Path
from unittest import mock

import pytest

from build_tools import llvm_openmp
from build_tools.fastflow_sdk_archive import (
    _bundle_llvm_openmp,
    _normalize_and_verify_linux_elf,
    _verify_llvm_openmp_imports,
    validate_source_policy,
)

pytestmark = [
    pytest.mark.compiler_stack,
    pytest.mark.fastflowlm,
    pytest.mark.presubmit,
]

PIN = "bdf04ebb32d7a5dce853f9b399dcbd3ed11a5aa7"

# objdump -T --wide excerpts from the built libomp and a FastFlowLM engine.
LIBOMP_SYMBOLS = """
libomp.so:     file format elf64-x86-64

DYNAMIC SYMBOL TABLE:
0000000000000000      DF *UND*\t0000000000000000 (GLIBC_2.2.5) free
00000000000e34f0 g    DF .text\t0000000000000077  VERSION     omp_get_thread_num
00000000000e34f0 g    DF .text\t0000000000000077 (OMP_1.0)    omp_get_thread_num
00000000000d3d20 g    DF .text\t0000000000000224  VERSION     GOMP_parallel
00000000000d3d20 g    DF .text\t0000000000000224 (GOMP_4.0)   GOMP_parallel
"""
ENGINE_SYMBOLS = """
libd3_npu.so:     file format elf64-x86-64

DYNAMIC SYMBOL TABLE:
0000000000000000      DF *UND*\t0000000000000000 (GLIBC_2.2.5) malloc
0000000000000000      DF *UND*\t0000000000000000 (GOMP_4.0)   GOMP_parallel
0000000000000000      DF *UND*\t0000000000000000 (OMP_1.0)    omp_get_thread_num
0000000000000000      DF *UND*\t0000000000000000 (GOMP_4.5)   GOMP_loop_nonmonotonic_dynamic_next
0000000000012340 g    DF .text\t0000000000000010  Base        d3_entry
"""


def _write_versions(root: Path, commit: str = PIN) -> None:
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    (root / "manifests" / "versions.toml").write_text(
        f'[llvm]\nrepo = "github.com/ROCm/llvm-project"\ncommit = "{commit}"\n',
        encoding="utf-8",
    )


def _write_install(install: Path, commit: str = PIN) -> None:
    (install / "lib").mkdir(parents=True)
    (install / "lib" / llvm_openmp.LIBRARY).write_bytes(b"libomp")
    (install / llvm_openmp.LICENSE).write_text("Apache-2.0 WITH LLVM-exception")
    (install / llvm_openmp.MARKER).write_text(
        json.dumps(
            {"repo": "https://github.com/ROCm/llvm-project.git", "commit": commit}
        )
    )


class TestLlvmOpenmp:
    def test_pinned_source_reads_versions_toml(self, tmp_path):
        _write_versions(tmp_path)
        assert llvm_openmp.pinned_source(tmp_path) == {
            "repo": "https://github.com/ROCm/llvm-project.git",
            "commit": PIN,
        }

    def test_load_install_requires_the_pinned_build(self, tmp_path):
        _write_versions(tmp_path)
        install = tmp_path / "build" / "llvm-openmp-install"
        with pytest.raises(RuntimeError, match="dev.py build llvm-openmp"):
            llvm_openmp.load_install(tmp_path, install)
        _write_install(install, commit="0" * 40)
        with pytest.raises(RuntimeError, match="not the pinned llvm-project"):
            llvm_openmp.load_install(tmp_path, install)

    def test_load_install_returns_library_license_and_component(self, tmp_path):
        _write_versions(tmp_path)
        install = tmp_path / "build" / "llvm-openmp-install"
        _write_install(install)
        loaded = llvm_openmp.load_install(tmp_path, install)
        assert loaded["library"] == install / "lib" / "libomp.so"
        assert loaded["license"] == install / "LICENSE.TXT"
        assert llvm_openmp.component(loaded["marker"]) == {
            "name": "llvm-openmp",
            "version": PIN,
            "upstream": "https://github.com/ROCm/llvm-project",
            "license": "Apache-2.0 WITH LLVM-exception",
        }

    def test_parse_dynamic_symbols_splits_defined_and_imported(self):
        defined, undefined = llvm_openmp.parse_dynamic_symbols(LIBOMP_SYMBOLS)
        assert ("GOMP_parallel", "GOMP_4.0") in defined
        assert ("omp_get_thread_num", "OMP_1.0") in defined
        assert ("free", "GLIBC_2.2.5") in undefined
        _, imports = llvm_openmp.parse_dynamic_symbols(ENGINE_SYMBOLS)
        assert ("GOMP_parallel", "GOMP_4.0") in imports
        assert ("malloc", "GLIBC_2.2.5") in imports

    def test_missing_openmp_symbols_ignores_non_openmp_imports(self):
        provided, _ = llvm_openmp.parse_dynamic_symbols(LIBOMP_SYMBOLS)
        _, required = llvm_openmp.parse_dynamic_symbols(ENGINE_SYMBOLS)
        assert llvm_openmp.missing_openmp_symbols(provided, required) == [
            "GOMP_loop_nonmonotonic_dynamic_next@GOMP_4.5"
        ]
        assert llvm_openmp.missing_openmp_symbols(
            provided, {("omp_get_thread_num", None), ("kmp_unknown", None)}
        ) == ["kmp_unknown"]

    def test_verify_openmp_imports_names_the_consumer(self, tmp_path):
        library = tmp_path / "libomp.so"
        engine = tmp_path / "libd3_npu.so"
        outputs = {library: LIBOMP_SYMBOLS, engine: ENGINE_SYMBOLS}

        def objdump(command, **kwargs):
            del kwargs
            return mock.Mock(stdout=outputs[Path(command[-1])])

        with (
            mock.patch("build_tools.llvm_openmp.shutil.which", return_value="objdump"),
            mock.patch("build_tools.llvm_openmp.subprocess.run", side_effect=objdump),
            pytest.raises(
                RuntimeError,
                match="libd3_npu.so: GOMP_loop_nonmonotonic_dynamic_next@GOMP_4.5",
            ),
        ):
            llvm_openmp.verify_openmp_imports(library, [engine])


class TestArchiveOpenmp:
    def test_bundle_relinks_every_libgomp_import(self, tmp_path):
        bundle = tmp_path / "bundle"
        (bundle / "bin").mkdir(parents=True)
        (bundle / "lib" / "flm").mkdir(parents=True)
        (bundle / "bin" / "flm").write_bytes(b"flm")
        (bundle / "lib" / "flm" / "libd3_npu.so").write_bytes(b"d3")
        (bundle / "lib" / "flm" / "libgemm.so").write_bytes(b"gemm")
        library = tmp_path / "libomp.so"
        library.write_bytes(b"libomp")
        needed = {
            "flm": ["libd3_npu.so", "libc.so.6"],
            "libd3_npu.so": ["libgomp.so.1", "libc.so.6"],
            "libgemm.so": ["libgomp.so.1"],
            "libomp.so": ["libc.so.6"],
        }
        commands = []
        with (
            mock.patch(
                "build_tools.fastflow_sdk_archive.shutil.which",
                return_value="patchelf",
            ),
            mock.patch(
                "build_tools.fastflow_sdk_archive.elf_needed",
                side_effect=lambda path, patchelf=None: needed[path.name],
            ),
            mock.patch(
                "build_tools.fastflow_sdk_archive.subprocess.run",
                side_effect=lambda command, **kwargs: commands.append(command),
            ),
        ):
            relinked = _bundle_llvm_openmp(bundle, library)
        assert (bundle / "lib" / "libomp.so").read_bytes() == b"libomp"
        assert relinked == ["lib/flm/libd3_npu.so", "lib/flm/libgemm.so"]
        assert all(
            command[:4] == ["patchelf", "--replace-needed", "libgomp.so.1", "libomp.so"]
            for command in commands
        )
        assert len(commands) == 2

    def test_import_check_covers_every_libomp_consumer(self, tmp_path):
        with mock.patch(
            "build_tools.fastflow_sdk_archive.llvm_openmp.verify_openmp_imports"
        ) as verify:
            _verify_llvm_openmp_imports(
                tmp_path,
                {
                    "bin/flm": ["libc.so.6"],
                    "lib/flm/libd3_npu.so": ["libomp.so", "libc.so.6"],
                    "share/fastflow-sdk/extensions/gemma4-open/lib/ext.so": [
                        "libomp.so"
                    ],
                },
            )
        verify.assert_called_once_with(
            tmp_path / "lib" / "libomp.so",
            [
                tmp_path / "lib/flm/libd3_npu.so",
                tmp_path / "share/fastflow-sdk/extensions/gemma4-open/lib/ext.so",
            ],
        )

    def test_extension_rpath_reaches_the_bundled_runtime(self, tmp_path):
        extension = tmp_path / "share/fastflow-sdk/extensions/gemma4-open/lib/ext.so"
        engine = tmp_path / "lib/flm/libd3_npu.so"
        rpaths = {}
        with (
            mock.patch(
                "build_tools.fastflow_sdk_archive.shutil.which",
                return_value="patchelf",
            ),
            mock.patch(
                "build_tools.fastflow_sdk_archive._linux_elf_files",
                return_value=[engine, extension],
            ),
            mock.patch(
                "build_tools.fastflow_sdk_archive.set_elf_rpath",
                side_effect=lambda path, rpath, patchelf: rpaths.update({path: rpath}),
            ),
            mock.patch("build_tools.fastflow_sdk_archive.elf_needed", return_value=[]),
        ):
            _normalize_and_verify_linux_elf(tmp_path)
        assert rpaths == {
            engine: "$ORIGIN:$ORIGIN/..",
            extension: "$ORIGIN:$ORIGIN/../../../../../lib",
        }

    def test_closure_requires_dependencies_reachable_through_rpath(self, tmp_path):
        extension_dir = tmp_path / "share/fastflow-sdk/extensions/gemma4-open/lib"
        extension = extension_dir / "ext.so"
        engine = tmp_path / "lib/flm/libd3_npu.so"
        for path in (
            extension,
            extension_dir / "libq4_npu_eXpress.so",
            engine,
            tmp_path / "lib/libomp.so",
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
        needed = {
            extension: ["libomp.so", "libq4_npu_eXpress.so", "libc.so.6"],
            engine: ["libomp.so"],
        }
        patches = (
            mock.patch(
                "build_tools.fastflow_sdk_archive.shutil.which",
                return_value="patchelf",
            ),
            mock.patch(
                "build_tools.fastflow_sdk_archive._linux_elf_files",
                return_value=[engine, extension],
            ),
            mock.patch("build_tools.fastflow_sdk_archive.set_elf_rpath"),
            mock.patch(
                "build_tools.fastflow_sdk_archive.elf_needed",
                side_effect=lambda path, patchelf: needed[path],
            ),
        )
        with patches[0], patches[1], patches[2], patches[3]:
            _normalize_and_verify_linux_elf(tmp_path)
            # Bundled, but only beside the extension: lib/flm cannot reach it.
            needed[engine] = ["libq4_npu_eXpress.so"]
            with pytest.raises(
                RuntimeError,
                match="lib/flm/libd3_npu.so -> libq4_npu_eXpress.so",
            ):
                _normalize_and_verify_linux_elf(tmp_path)

    def test_source_policy_rejects_an_unpinned_openmp_runtime(self, tmp_path):
        manifest = {
            "sources": {},
            "components": [{"name": "llvm-openmp", "version": "0" * 40}],
        }
        versions = {
            "llvm": {"commit": PIN},
            "hrx": {"commit": "h"},
            "fastflowlm": {"commit": "f"},
        }
        with (
            mock.patch(
                "build_tools.fastflow_sdk_archive.tomllib.loads",
                return_value=versions,
            ),
            mock.patch(
                "build_tools.fastflow_sdk_archive._git_revision", return_value="s"
            ),
            mock.patch("pathlib.Path.read_text", return_value=""),
        ):
            manifest["sources"] = {
                "fastflow-sdk": {"revision": "s", "dirty": False},
                "hrx": {"revision": "h"},
                "FastFlowLM": {"revision": "f"},
            }
            with pytest.raises(RuntimeError, match="LLVM OpenMP runtime"):
                validate_source_policy(manifest, tmp_path)
            manifest["components"][0]["version"] = PIN
            validate_source_policy(manifest, tmp_path)
