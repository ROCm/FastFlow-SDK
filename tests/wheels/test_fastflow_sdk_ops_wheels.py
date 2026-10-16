# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import email.parser
import json
import zipfile
from pathlib import Path

import pytest

from build_tools import paths
from build_tools.artifact import git_source_state
from build_tools.fastflow_sdk_ops_wheels import (
    _EXE,
    _TORCH_AT_IMPORT,
    _VBIN,
    IRON_OPS_SOURCE,
    OPS_SUMMARY,
    OPS_WHEEL_TAG,
    _installed_ops_compile,
    _installed_ops_probe,
    _ops_compile_specs,
    _ops_refresh_packages,
    _ops_verify_python,
    build_ops_wheel,
    ops_requires,
    probe_operator_modules,
    stage_ops_tree,
)
from build_tools.fastflow_sdk_wheel_common import (
    COMPILER_BINDINGS_DISTRIBUTION,
    COMPILER_CORE_DISTRIBUTION,
    COMPILER_DISTRIBUTION,
    OPS_DISTRIBUTION,
    OPS_PACKAGE,
    verify_wheel_source_revision,
    verify_wheel_text_path_references,
)
from build_tools.iron import OPERATORS, host_compile_suite
from build_tools.release_policy import compiler_requires_python

pytestmark = [pytest.mark.compiler_stack, pytest.mark.presubmit]


def _write_iron_ops_source(root: Path) -> Path:
    source = root / "iron-ops"
    files = {
        "LICENSE": "Apache-2.0\n",
        "iron/__init__.py": "",
        "iron/operators/__init__.py": "from .elementwise_add.op import ElementwiseAdd\n",
        "iron/operators/elementwise_add/op.py": "class ElementwiseAdd:\n    pass\n",
        "iron/operators/elementwise_add/test.py": "def test_should_not_ship():\n    pass\n",
        "iron/common/test_utils.py": "def run_test():\n    pass\n",
        "iron/tests/infrastructure/foo.py": "assert False\n",
        "iron/applications/llama.py": "assert False\n",
        "iron/applications/gemma4_flm/build.py": "def main():\n    pass\n",
        "iron/applications/gemma4_flm/overrides.hpp": "// override\n",
        "iron/applications/gemma4_flm/pli_weights.py": "def pack():\n    pass\n",
        "iron/applications/gemma4_flm/test.py": "def test_should_not_ship():\n    pass\n",
        "iron/applications/gemma4_flm/Makefile": "clone:\n",
        "aie_kernels/generic/add.cc": "// add\n",
        "aie_kernels/README.md": "kernels\n",
    }
    for relative, contents in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    return source


class TestFastflowSdkOpsWheels:
    def test_staged_tree_keeps_import_layout_and_drops_tests(self, tmp_path):
        source = _write_iron_ops_source(tmp_path)
        tree = stage_ops_tree(source, tmp_path / "stage", version="0.1.0.dev1")

        assert (tree / "iron" / "operators" / "__init__.py").is_file()
        assert (tree / "iron" / "common" / "test_utils.py").is_file()
        assert (tree / "aie_kernels" / "generic" / "add.cc").is_file()
        assert not (tree / "iron" / "tests").exists()
        assert (tree / "iron" / "applications" / "gemma4_flm" / "build.py").is_file()
        assert (
            tree / "iron" / "applications" / "gemma4_flm" / "overrides.hpp"
        ).is_file()
        assert not (tree / "iron" / "applications" / "llama.py").exists()
        assert not (tree / "iron" / "applications" / "gemma4_flm" / "test.py").exists()
        assert not (tree / "iron" / "applications" / "gemma4_flm" / "Makefile").exists()
        assert not (
            tree / "iron" / "operators" / "elementwise_add" / "test.py"
        ).exists()
        assert (tree / OPS_PACKAGE / "licenses" / "IRON-LICENSE").is_file()
        assert (tree / OPS_PACKAGE / "licenses" / "fastflow-sdk-LICENSE").is_file()
        init = (tree / OPS_PACKAGE / "__init__.py").read_text(encoding="utf-8")
        assert "def kernels_root" in init
        assert "def gemma4_flm_root" in init
        assert "def manifest" in init

        manifest = json.loads(
            (tree / OPS_PACKAGE / "manifest.json").read_text(encoding="utf-8")
        )
        paths = {record["path"] for record in manifest["files"]}
        assert "iron/operators/__init__.py" in paths
        assert "aie_kernels/generic/add.cc" in paths
        assert "iron/tests/infrastructure/foo.py" not in paths
        assert manifest["requires"] == [f"{COMPILER_DISTRIBUTION} ==0.1.0.dev1"]
        assert manifest["wheel_tag"] == OPS_WHEEL_TAG

    def test_ops_wheel_requires_matching_compiler(self, tmp_path, monkeypatch):
        monkeypatch.setattr("build_tools.fastflow_sdk_ops_wheels.paths.BUILD", tmp_path)
        source = _write_iron_ops_source(tmp_path)
        wheel = build_ops_wheel(
            source, output_directory=tmp_path / "dist", version="0.1.0.dev1"
        )

        assert wheel.name == "fastflow_sdk_ops-0.1.0.dev1-py3-none-any.whl"
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            assert "iron/operators/__init__.py" in names
            assert "aie_kernels/generic/add.cc" in names
            assert "iron/tests/infrastructure/foo.py" not in names
            metadata = email.parser.Parser().parsestr(
                archive.read("fastflow_sdk_ops-0.1.0.dev1.dist-info/METADATA").decode(
                    "utf-8"
                )
            )
            wheel_info = email.parser.Parser().parsestr(
                archive.read("fastflow_sdk_ops-0.1.0.dev1.dist-info/WHEEL").decode(
                    "utf-8"
                )
            )
        assert metadata["Name"] == OPS_DISTRIBUTION
        assert metadata["Summary"] == OPS_SUMMARY
        assert metadata["License-Expression"] == "MIT AND Apache-2.0"
        assert metadata.get_all("License-File") == [
            "IRON-LICENSE",
            "fastflow-sdk-LICENSE",
        ]
        assert metadata["Requires-Python"] == compiler_requires_python()
        assert metadata.get_all("Requires-Dist") == list(ops_requires("0.1.0.dev1"))
        assert wheel_info["Root-Is-Purelib"] == "true"
        assert wheel_info.get_all("Tag") == [OPS_WHEEL_TAG]
        revision = git_source_state(paths.ROOT)["revision"]
        verify_wheel_source_revision([wheel], revision, product="ops wheel")
        with pytest.raises(RuntimeError, match="expected='wrong'"):
            verify_wheel_source_revision([wheel], "wrong", product="ops wheel")
        verify_wheel_text_path_references(
            {OPS_DISTRIBUTION: wheel},
            (paths.ROOT, tmp_path),
            product="ops wheel",
        )

    def test_text_path_scan_covers_kernel_sources(self, tmp_path, monkeypatch):
        monkeypatch.setattr("build_tools.fastflow_sdk_ops_wheels.paths.BUILD", tmp_path)
        source = _write_iron_ops_source(tmp_path)
        (source / "aie_kernels" / "generic" / "add.cc").write_text(
            f"// leaked {tmp_path.resolve()}\n", encoding="utf-8"
        )
        wheel = build_ops_wheel(
            source, output_directory=tmp_path / "dist", version="0.1.0.dev1"
        )
        with pytest.raises(RuntimeError, match="text payload contains local paths"):
            verify_wheel_text_path_references(
                {OPS_DISTRIBUTION: wheel},
                (tmp_path,),
                product="ops wheel",
            )

    def test_metadata_package_resolves_kernel_layout(self, tmp_path, monkeypatch):
        import sys

        source = _write_iron_ops_source(tmp_path)
        tree = stage_ops_tree(source, tmp_path / "stage", version="0.1.0.dev1")
        monkeypatch.syspath_prepend(str(tree))
        sys.modules.pop("fastflow_sdk_ops", None)
        import fastflow_sdk_ops

        try:
            assert fastflow_sdk_ops.install_root() == tree.resolve()
            assert (fastflow_sdk_ops.kernels_root() / "generic" / "add.cc").is_file()
            assert fastflow_sdk_ops.manifest()["product"] == OPS_DISTRIBUTION
        finally:
            sys.modules.pop("fastflow_sdk_ops", None)


def _write_operators_catalog(
    root: Path, packages: dict[str, str], exports: dict[str, str]
) -> Path:
    site = root / "site"
    operators = site / "iron" / "operators"
    operators.mkdir(parents=True)
    (site / "iron" / "__init__.py").write_text("", encoding="utf-8")
    for package, body in packages.items():
        package_dir = operators / package
        package_dir.mkdir()
        (package_dir / "op.py").write_text(body, encoding="utf-8")
    mapping = ", ".join(f"{name!r}: {package!r}" for name, package in exports.items())
    (operators / "__init__.py").write_text(
        (
            "import importlib\n"
            f"_OPERATOR_MODULES = {{{mapping}}}\n"
            "__all__ = sorted(_OPERATOR_MODULES)\n"
            "def __getattr__(name):\n"
            "    module = _OPERATOR_MODULES.get(name)\n"
            "    if module is None:\n"
            "        raise AttributeError(name)\n"
            "    return getattr(importlib.import_module(f'.{module}.op', __name__), name)\n"
        ),
        encoding="utf-8",
    )
    return site


def _load_operators(site: Path, monkeypatch):
    import sys

    monkeypatch.syspath_prepend(str(site))
    for name in list(sys.modules):
        if name == "iron" or name.startswith("iron."):
            del sys.modules[name]
    import iron.operators as operators

    return operators


class TestOpsOperatorProbe:
    @pytest.fixture(autouse=True)
    def _isolate_iron(self):
        import sys

        yield
        for name in list(sys.modules):
            if name == "iron" or name.startswith("iron."):
                del sys.modules[name]

    def test_probe_script_embeds_closed_catalog_helper(self):
        probe = _installed_ops_probe()
        compile(probe, "<ops-probe>", "exec")
        assert "def probe_operator_modules" in probe
        assert "import fastflow_sdk_ops" in probe
        assert "kernels_root" in probe
        assert "ModuleNotFoundError" in probe
        assert 'current.name == "torch"' in probe
        assert '(child / "op.py")' in probe or 'child / "op.py"' in probe
        assert _TORCH_AT_IMPORT == {"swiglu_prefill_stream"}
        for name in _TORCH_AT_IMPORT:
            assert repr(name) in probe

    def test_compile_script_embeds_torch_free_operator_suite(self, tmp_path):
        script = _installed_ops_compile(tmp_path / "ops-compile")
        compile(script, "<ops-compile>", "exec")
        specs = _ops_compile_specs()
        names = [spec["name"] for spec in specs]
        assert names == [
            *(operator.name for operator in host_compile_suite()),
            "add_relu_fused",
        ]
        assert not _TORCH_AT_IMPORT.intersection(names)
        assert "gemm" in names
        assert "relu" in names
        assert "elementwise_add" in names
        assert "mha" in names
        assert "swiglu_decode" not in names
        assert "swiglu_prefill" not in names
        assert "strided_copy" in names
        strided_copy = next(spec for spec in specs if spec["name"] == "strided_copy")
        assert strided_copy["full_elf"] is True
        add_relu_fused = next(
            spec for spec in specs if spec["name"] == "add_relu_fused"
        )
        assert add_relu_fused["kind"] == "add_relu_sequence"
        assert add_relu_fused["full_elf"] is True
        assert all(
            spec["full_elf"] is False
            for spec in specs
            if spec["name"] not in ("strided_copy", "add_relu_fused")
        )
        for spec in specs:
            assert spec["device"] == "npu2"
            assert spec["kwargs"]
        for package in _TORCH_AT_IMPORT:
            assert f"'name': {package!r}" not in script
        assert "def compile_one(" in script
        assert "ThreadPoolExecutor" in script
        assert "subprocess.run" in script
        assert "--operator" in script
        assert "ProcessPoolExecutor" not in script
        assert "compile_job_count" in script
        assert 'if __name__ == "__main__"' in script
        assert "def require_runtime_image" in script
        assert "def classify_compiled_images" in script
        assert 'if spec["full_elf"]:' in script
        assert "OperatorSequence(" in script
        assert '"full-elf"' in script
        assert '"context"' in script
        assert "tile_m" in script
        assert "FullElfArtifact" in script
        assert "activate_toolchain_env" in script
        assert "compiler prefix has no include" in script

    def test_compile_suite_is_every_torch_free_op_with_a_shape(self):
        operators = IRON_OPS_SOURCE / "iron" / "operators"
        if not operators.is_dir():
            pytest.skip("iron-ops source is not checked out for this CI profile")
        packages = set()
        for child in operators.iterdir():
            if not child.is_dir():
                continue
            if (child / "op.py").is_file():
                packages.add(child.name)
                continue
            for nested in child.iterdir():
                if nested.is_dir() and (nested / "op.py").is_file():
                    packages.add(f"{child.name}.{nested.name}")
        catalog = {op.name for op in OPERATORS} | {
            op.module for op in OPERATORS if op.module
        }
        leftover = packages - catalog - set(_TORCH_AT_IMPORT)
        assert leftover == set()

    def test_reuse_venv_keeps_existing_interpreter(self, tmp_path, monkeypatch):
        monkeypatch.setattr("build_tools.fastflow_sdk_ops_wheels.paths.BUILD", tmp_path)
        venv = tmp_path / "compiler-wheel-verify-312"
        python = venv / _VBIN / f"python{_EXE}"
        python.parent.mkdir(parents=True)
        python.write_text("", encoding="utf-8")
        marker = venv / "keep-me"
        marker.write_text("x", encoding="utf-8")
        assert _ops_verify_python(venv, reuse=True) == python
        assert marker.is_file()

    def test_reuse_venv_requires_existing_interpreter(self, tmp_path, monkeypatch):
        monkeypatch.setattr("build_tools.fastflow_sdk_ops_wheels.paths.BUILD", tmp_path)
        with pytest.raises(RuntimeError, match="reuse-venv requires"):
            _ops_verify_python(tmp_path / "missing", reuse=True)

    def test_reuse_venv_refreshes_only_the_ops_wheel(self):
        assert _ops_refresh_packages(reuse=True) == (OPS_DISTRIBUTION,)
        assert _ops_refresh_packages(reuse=False) == (
            OPS_DISTRIBUTION,
            COMPILER_DISTRIBUTION,
            COMPILER_CORE_DISTRIBUTION,
            COMPILER_BINDINGS_DISTRIBUTION,
        )

    def test_catalog_imports_and_skips_torch_packages(self, tmp_path, monkeypatch):
        site = _write_operators_catalog(
            tmp_path,
            {
                "elementwise_add": "class ElementwiseAdd:\n    pass\n",
                "relu": "class ReLU:\n    pass\n",
                "mha": "import torch\n\nclass MHA:\n    pass\n",
            },
            {"ElementwiseAdd": "elementwise_add", "MHA": "mha"},
        )
        operators = _load_operators(site, monkeypatch)
        imported, skipped = probe_operator_modules(
            operators, torch_at_import=frozenset({"mha"})
        )
        assert "iron.operators.elementwise_add.op" in imported
        assert "iron.operators.relu.op" in imported
        assert skipped == ["iron.operators.mha.op"]

    def test_unused_torch_skip_is_a_hard_failure(self, tmp_path, monkeypatch):
        site = _write_operators_catalog(
            tmp_path,
            {
                "elementwise_add": "class ElementwiseAdd:\n    pass\n",
                "mha": "class MHA:\n    pass\n",
            },
            {"ElementwiseAdd": "elementwise_add", "MHA": "mha"},
        )
        operators = _load_operators(site, monkeypatch)
        with pytest.raises(RuntimeError, match="skip list is stale; drop: mha"):
            probe_operator_modules(operators, torch_at_import=frozenset({"mha"}))

    def test_missing_torch_skip_package_fails(self, tmp_path, monkeypatch):
        site = _write_operators_catalog(
            tmp_path,
            {"elementwise_add": "class ElementwiseAdd:\n    pass\n"},
            {"ElementwiseAdd": "elementwise_add"},
        )
        operators = _load_operators(site, monkeypatch)
        with pytest.raises(RuntimeError, match="missing torch-at-import"):
            probe_operator_modules(operators, torch_at_import=frozenset({"mha"}))

    def test_unexpected_torch_import_is_not_skipped(self, tmp_path, monkeypatch):
        site = _write_operators_catalog(
            tmp_path,
            {
                "elementwise_add": "class ElementwiseAdd:\n    pass\n",
                "relu": "import torch\n\nclass ReLU:\n    pass\n",
            },
            {"ElementwiseAdd": "elementwise_add"},
        )
        operators = _load_operators(site, monkeypatch)
        with pytest.raises(ModuleNotFoundError):
            probe_operator_modules(operators, torch_at_import=frozenset())

    def test_all_torch_error_does_not_skip_imported_exports(
        self, tmp_path, monkeypatch
    ):
        site = _write_operators_catalog(
            tmp_path,
            {
                "elementwise_add": "class ElementwiseAdd:\n    pass\n",
                "mha": "import torch\n\nclass MHA:\n    pass\n",
            },
            {"ElementwiseAdd": "elementwise_add", "MHA": "mha"},
        )
        operators = _load_operators(site, monkeypatch)

        def _raise_torch(name):
            raise ModuleNotFoundError("No module named 'torch'", name="torch")

        operators.__getattr__ = _raise_torch
        with pytest.raises(ModuleNotFoundError):
            probe_operator_modules(operators, torch_at_import=frozenset({"mha"}))
