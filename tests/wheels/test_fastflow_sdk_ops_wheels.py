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
    _VBIN,
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
from build_tools.iron import SEQUENCES, host_compile_suite
from build_tools.release_policy import compiler_requires_python

pytestmark = [pytest.mark.compiler_stack, pytest.mark.presubmit]

_ENGINE = "iron/operators/flm/gemma4_engine"


def _write_iron_ops_source(root: Path) -> Path:
    source = root / "iron-ops"
    files = {
        "LICENSE": "Apache-2.0\n",
        "iron/__init__.py": "",
        "iron/operators/__init__.py": "from .elementwise_add import ElementwiseAdd\n",
        "iron/operators/elementwise_add.py": "class ElementwiseAdd:\n    pass\n",
        "iron/operators/flm/__init__.py": "",
        "iron/operators/flm/gemm/__init__.py": "",
        "iron/operators/flm/gemm/op.py": "class ProjectionGEMM:\n    pass\n",
        "iron/operators/flm/gemm/test.py": "def test_should_not_ship():\n    pass\n",
        "iron/common/test_utils.py": "def run_test():\n    pass\n",
        "iron/tests/infrastructure/foo.py": "assert False\n",
        "iron/applications/llama.py": "assert False\n",
        f"{_ENGINE}/build.py": "def main():\n    pass\n",
        f"{_ENGINE}/overrides.hpp": "// override\n",
        f"{_ENGINE}/pli_weights.py": "def pack():\n    pass\n",
        f"{_ENGINE}/Makefile": "clone:\n",
        f"{_ENGINE}/README.md": "engine\n",
        f"{_ENGINE}/build/FastFlowLM/build.py": "assert False\n",
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
        assert (tree / "iron" / "operators" / "elementwise_add.py").is_file()
        assert (tree / "iron" / "operators" / "flm" / "gemm" / "op.py").is_file()
        assert (tree / "iron" / "common" / "test_utils.py").is_file()
        assert not (tree / "aie_kernels").exists()
        assert not (tree / "iron" / "tests").exists()
        assert not (tree / "iron" / "applications").exists()
        assert not (tree / "iron" / "operators" / "flm" / "gemm" / "test.py").exists()
        engine = tree / _ENGINE
        assert sorted(path.name for path in engine.iterdir()) == [
            "build.py",
            "overrides.hpp",
            "pli_weights.py",
        ]
        assert (tree / OPS_PACKAGE / "licenses" / "IRON-LICENSE").is_file()
        assert (tree / OPS_PACKAGE / "licenses" / "fastflow-sdk-LICENSE").is_file()
        init = (tree / OPS_PACKAGE / "__init__.py").read_text(encoding="utf-8")
        assert "def kernels_root" in init
        assert "def gemma4_engine_root" in init
        assert "def manifest" in init

        manifest = json.loads(
            (tree / OPS_PACKAGE / "manifest.json").read_text(encoding="utf-8")
        )
        paths = {record["path"] for record in manifest["files"]}
        assert "iron/operators/__init__.py" in paths
        assert "aie_kernels/generic/add.cc" not in paths
        assert "iron/tests/infrastructure/foo.py" not in paths
        assert f"{_ENGINE}/Makefile" not in paths
        assert manifest["requires"] == [f"{COMPILER_DISTRIBUTION} ==0.1.0.dev1"]
        assert manifest["wheel_tag"] == OPS_WHEEL_TAG

    def test_staging_requires_the_gemma4_engine(self, tmp_path):
        source = _write_iron_ops_source(tmp_path)
        (source / _ENGINE / "overrides.hpp").unlink()
        with pytest.raises(RuntimeError, match="missing gemma4_engine files"):
            stage_ops_tree(source, tmp_path / "stage", version="0.1.0.dev1")

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
            assert not any(name.startswith("aie_kernels/") for name in names)
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

    def test_text_path_scan_covers_engine_sources(self, tmp_path, monkeypatch):
        monkeypatch.setattr("build_tools.fastflow_sdk_ops_wheels.paths.BUILD", tmp_path)
        source = _write_iron_ops_source(tmp_path)
        (source / _ENGINE / "overrides.hpp").write_text(
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
        import types

        source = _write_iron_ops_source(tmp_path)
        tree = stage_ops_tree(source, tmp_path / "stage", version="0.1.0.dev1")
        compiler_kernels = tmp_path / "compiler" / "include" / "aie_kernels"
        config = types.ModuleType("aie.utils.config")
        config.aie_kernels_dir = lambda: str(compiler_kernels)
        for name in ("aie", "aie.utils"):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        monkeypatch.setitem(sys.modules, "aie.utils.config", config)
        monkeypatch.syspath_prepend(str(tree))
        sys.modules.pop("fastflow_sdk_ops", None)
        import fastflow_sdk_ops

        try:
            assert fastflow_sdk_ops.install_root() == tree.resolve()
            assert fastflow_sdk_ops.kernels_root() == compiler_kernels
            assert (fastflow_sdk_ops.gemma4_engine_root() / "overrides.hpp").is_file()
            assert fastflow_sdk_ops.manifest()["product"] == OPS_DISTRIBUTION
        finally:
            sys.modules.pop("fastflow_sdk_ops", None)


def _write_operators_catalog(root: Path, modules: dict[str, str], catalog: str) -> Path:
    """An ``iron.operators`` package of ``modules`` (dotted name to source)."""
    site = root / "site"
    operators = site / "iron" / "operators"
    operators.mkdir(parents=True)
    (site / "iron" / "__init__.py").write_text("", encoding="utf-8")
    for dotted, body in modules.items():
        path = operators.joinpath(*dotted.split(".")).with_suffix(".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    (operators / "__init__.py").write_text(catalog, encoding="utf-8")
    return site


def _load_operators(site: Path, monkeypatch):
    import sys

    monkeypatch.syspath_prepend(str(site))
    for name in list(sys.modules):
        if name == "iron" or name.startswith("iron."):
            del sys.modules[name]
    import iron.operators as operators

    return operators


_CATALOG = (
    "from .relu import ReLU\n"
    "from .flm.gemm.op import ProjectionGEMM\n"
    "CATALOG = {'ReLU': ReLU, 'flm.ProjectionGEMM': ProjectionGEMM}\n"
)


class TestOpsOperatorProbe:
    @pytest.fixture(autouse=True)
    def _isolate_iron(self):
        import sys

        yield
        for name in list(sys.modules):
            if name == "iron" or name.startswith("iron."):
                del sys.modules[name]
        # Only the stand-in the torch test plants; a real torch has a file.
        torch = sys.modules.get("torch")
        if torch is not None and not hasattr(torch, "__file__"):
            del sys.modules["torch"]

    @pytest.fixture
    def no_torch(self):
        import sys

        if "torch" in sys.modules:
            pytest.skip("torch is already imported in this process")

    def test_probe_script_embeds_the_catalog_helper(self):
        probe = _installed_ops_probe()
        compile(probe, "<ops-probe>", "exec")
        assert "def probe_operator_modules" in probe
        assert "import fastflow_sdk_ops" in probe
        assert "kernels_root" in probe
        assert "gemma4_engine_root" in probe
        assert '"torch" in sys.modules' in probe

    def test_compile_script_embeds_the_whole_suite(self, tmp_path):
        script = _installed_ops_compile(tmp_path / "ops-compile")
        compile(script, "<ops-compile>", "exec")
        operators, sequences = _ops_compile_specs()
        assert [spec["name"] for spec in operators] == [
            operator.name for operator in host_compile_suite()
        ]
        assert [spec["name"] for spec in sequences] == [s.name for s in SEQUENCES]
        assert [spec["name"] for spec in operators if spec["full_elf"]] == [
            "strided_copy",
            "gather",
            "gather_device_ids",
        ]
        for spec in operators:
            assert spec["device"] == "npu2"
            assert spec["build_kwargs"]
            assert spec["class_name"]
        for helper in (
            "class Operator(NamedTuple)",
            "class Sequence(NamedTuple)",
            "def require_runtime_image",
            "def require_host_library",
            "def classify_compiled_images",
            "class Compiled(NamedTuple)",
            "def stage_compiled_images",
            "def operator_class",
            "def construct_operator",
            "def add_relu_graph",
            "def offset_copy_graph",
            "def gather_graph",
            "def require_compile_cache",
            "def compile_full_elf",
            "def compile_operator",
            "def compile_sequence",
            "def compile_job_count",
            "def resolve_build_kwargs",
        ):
            assert helper in script, helper
        assert (
            "FULL_ELF_GRAPHS = {'strided_copy': offset_copy_graph, "
            "'gather': gather_graph, 'gather_device_ids': gather_graph}"
        ) in script
        assert "STAGED_NAMES = " in script
        assert "NPU_CACHE_HOME" in script
        assert "--keep-cache" in script
        assert "def compile_one(" in script
        assert "ThreadPoolExecutor" in script
        assert "subprocess.run" in script
        assert "ProcessPoolExecutor" not in script
        assert 'if __name__ == "__main__"' in script
        assert "activate_toolchain_env" in script
        assert "compiler prefix has no include" in script

    def test_compile_script_rebuilds_the_catalog_records(self, tmp_path):
        script = _installed_ops_compile(tmp_path / "ops-compile")
        start = script.index("STAGED_NAMES = ")
        end = script.index("DEFAULT_BUILD_DIR = ")
        namespace = {}
        exec(
            "import importlib, json, os, shutil\n"
            "from pathlib import Path\n"
            "from typing import NamedTuple\n" + script[start:end],
            namespace,
        )
        names = [operator.name for operator in namespace["OPERATORS"]]
        assert names == [operator.name for operator in host_compile_suite()]
        assert namespace["OPERATORS"] == host_compile_suite()
        assert [s.name for s in namespace["SEQUENCES"]] == [s.name for s in SEQUENCES]

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

    def test_probe_imports_every_package_module(self, tmp_path, monkeypatch, no_torch):
        site = _write_operators_catalog(
            tmp_path,
            {
                "relu": "class ReLU:\n    pass\n",
                "flm.__init__": "",
                "flm.gemm.__init__": "",
                "flm.gemm.op": "class ProjectionGEMM:\n    pass\n",
                "flm.gemm.test": "assert False, 'a test module ships no import'\n",
                "flm.gemma4_engine.build": "assert False, 'a script, not a module'\n",
            },
            _CATALOG,
        )
        operators = _load_operators(site, monkeypatch)
        imported = probe_operator_modules(operators)
        assert imported == [
            "iron.operators",
            "iron.operators.flm",
            "iron.operators.flm.gemm",
            "iron.operators.flm.gemm.op",
            "iron.operators.relu",
        ]

    def test_an_import_that_pulls_in_torch_fails(self, tmp_path, monkeypatch, no_torch):
        site = _write_operators_catalog(
            tmp_path,
            {
                "relu": "class ReLU:\n    pass\n",
                "flm.__init__": "",
                "flm.gemm.__init__": "",
                "flm.gemm.op": "class ProjectionGEMM:\n    pass\n",
                "mha": "import sys, types\nsys.modules['torch'] = types.ModuleType('torch')\n",
            },
            _CATALOG,
        )
        operators = _load_operators(site, monkeypatch)
        with pytest.raises(RuntimeError, match="imported torch"):
            probe_operator_modules(operators)

    def test_a_catalog_entry_must_be_an_imported_class(
        self, tmp_path, monkeypatch, no_torch
    ):
        site = _write_operators_catalog(
            tmp_path,
            {"relu": "class ReLU:\n    pass\n"},
            "from .relu import ReLU\nCATALOG = {'ReLU': ReLU, 'Gone': None}\n",
        )
        operators = _load_operators(site, monkeypatch)
        with pytest.raises(RuntimeError, match=r"CATALOG\['Gone'\]"):
            probe_operator_modules(operators)
