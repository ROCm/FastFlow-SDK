# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from build_tools import fastflowlm_qualification_native as qualification
from build_tools import fastflowlm_xadx_assets as xadx_assets
from build_tools.artifact import file_record

pytestmark = [pytest.mark.fastflowlm, pytest.mark.presubmit]


def test_native_qualification_sources_have_single_owner():
    root = Path(__file__).resolve().parents[2]
    legacy_native_sources = [
        *Path(__file__).parent.glob("*.cpp"),
        *Path(__file__).parent.glob("*.hpp"),
    ]
    assert legacy_native_sources == []
    assert not (Path(__file__).parent / "CMakeLists.txt").exists()
    native = root / "tools/rocm-npu-qualification/native"
    assert {
        "kernel_capabilities.cpp",
        "kernel_capabilities.hpp",
        "kernel_native_tests.cpp",
        "kernel_shape_probe.cpp",
        "kernel_validation.hpp",
    } <= {path.name for path in native.iterdir()}


def test_build_helper_import_does_not_require_numpy():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.modules['numpy'] = None; "
                "import build_tools.fastflowlm_qualification_native"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_file_record_is_relative_and_content_bound(tmp_path):
    payload = tmp_path / "nested" / "payload.bin"
    payload.parent.mkdir()
    payload.write_bytes(b"qualification")
    assert file_record(tmp_path, payload) == {
        "path": "nested/payload.bin",
        "size": 13,
        "sha256": hashlib.sha256(b"qualification").hexdigest(),
    }


def test_capability_loader_rejects_unsafe_family_paths(tmp_path):
    document = json.loads(qualification.CAPABILITIES_SOURCE.read_text(encoding="utf-8"))
    document["families"][0]["xclbin_directory"] = "../outside"
    path = tmp_path / "capabilities.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RuntimeError, match="family capability"):
        qualification.load_capabilities(path)


def test_shipping_recipe_generators_are_directly_loadable():
    generators = qualification.recipe_generators()
    assert generators.predicted_size("zero", {"size": 17}) == 17


def test_native_contract_tests_build_and_run_through_ctest(tmp_path):
    cmake = tmp_path / "bin" / "cmake"
    cmake.parent.mkdir()
    cmake.write_text("", encoding="utf-8")
    with (
        mock.patch.object(qualification, "QUALIFICATION_BUILD", tmp_path / "build"),
        mock.patch.object(qualification, "configure_qualification_tools"),
        mock.patch.object(qualification, "qualification_runtime_env", return_value={}),
        mock.patch.object(
            qualification.environment, "cmake_bin", return_value=str(cmake)
        ),
        mock.patch.object(qualification.environment, "cmake_env", return_value={}),
        mock.patch.object(qualification.environment, "sh") as sh,
    ):
        qualification.run_native_qualification_tests()

    build, ctest = [call.args[0] for call in sh.call_args_list]
    assert build[0:4] == [
        str(cmake),
        "--build",
        (tmp_path / "build").as_posix(),
        "--target",
    ]
    assert build[4:6] == list(qualification.NATIVE_TEST_TARGETS)
    assert ctest == [
        cmake.with_name(qualification.paths.executable_name("ctest")),
        "--test-dir",
        (tmp_path / "build").as_posix(),
        "--output-on-failure",
        "-C",
        "Release",
    ]
    ctest_environment = sh.call_args_list[1].kwargs["env"]
    temporary_directory = Path(ctest_environment["TMPDIR"])
    assert {ctest_environment[name] for name in ("TEMP", "TMP", "TMPDIR")} == {
        str(temporary_directory)
    }
    assert temporary_directory.parent == tmp_path / "build"
    assert temporary_directory.name.startswith("ctest-")
    assert not temporary_directory.exists()


@pytest.mark.parametrize(
    ("windows", "variable", "hrx_subdirectory"),
    [(True, "Path", "bin"), (False, "LD_LIBRARY_PATH", "lib")],
)
def test_qualification_runtime_libraries_are_searchable(
    tmp_path, windows, variable, hrx_subdirectory
):
    project_root = tmp_path / "projects"
    fastflowlm_lib = project_root / "FastFlowLM" / "src" / "lib"
    hrx_install = tmp_path / "hrx"
    hrx_lib = hrx_install / hrx_subdirectory
    fastflowlm_lib.mkdir(parents=True)
    hrx_lib.mkdir(parents=True)
    with (
        mock.patch.object(qualification.paths, "PROJ", project_root),
        mock.patch.object(qualification, "HRX_INSTALL", hrx_install),
        mock.patch.object(qualification.paths, "is_windows", return_value=windows),
        mock.patch.object(
            qualification.environment,
            "cmake_env",
            return_value={variable: "existing"},
        ),
    ):
        env = qualification.qualification_runtime_env()
    assert env[variable].split(qualification.os.pathsep) == [
        str(fastflowlm_lib),
        str(hrx_lib),
        "existing",
    ]


def test_xadx_generation_runs_native_contract_tests():
    args = SimpleNamespace(fresh=True, dir=Path("assets"))
    with (
        mock.patch.object(xadx_assets, "build_xadx_generator") as build,
        mock.patch.object(xadx_assets, "run_native_qualification_tests") as native,
        mock.patch.object(xadx_assets, "verify_recipe_generator_parity") as parity,
        mock.patch.object(
            xadx_assets, "generate_xadx_assets", return_value=args.dir
        ) as generate,
    ):
        assert xadx_assets.cmd_generate_xadx_assets(args) == args.dir

    build.assert_called_once_with(True)
    native.assert_called_once_with(configure=False)
    parity.assert_called_once_with(configure=False)
    generate.assert_called_once()
