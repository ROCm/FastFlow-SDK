# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest

from build_tools.fastflow_sdk_wheel_common import (
    COMPILER_BINDINGS_DISTRIBUTION,
    COMPILER_CORE_DISTRIBUTION,
    COMPILER_DISTRIBUTION,
    FASTFLOWLM_DISTRIBUTION,
    OPS_DISTRIBUTION,
    RUNTIME_DISTRIBUTION,
    SDK_DISTRIBUTION,
)
from build_tools.resolve_release_wheel import (
    bindings_runtime_pins,
    expected_distributions,
    pip_install_command,
    python_tag,
    resolve_release_wheel,
    verify_installed,
)

pytestmark = [pytest.mark.compiler_stack, pytest.mark.presubmit]


def test_resolver_does_not_import_compiler_wheel_builder():
    source = Path(resolve_release_wheel.__code__.co_filename).read_text(
        encoding="utf-8"
    )
    assert "fastflow_sdk_compiler_wheels" not in source


def test_expected_compiler_includes_bindings_pins():
    expected = expected_distributions("compiler", "1.2.3")
    assert expected[COMPILER_DISTRIBUTION] == "1.2.3"
    assert expected[COMPILER_CORE_DISTRIBUTION] == "1.2.3"
    assert expected[COMPILER_BINDINGS_DISTRIBUTION] == "1.2.3"
    pins = bindings_runtime_pins()
    assert pins
    for name, version in pins.items():
        assert expected[name] == version


def test_expected_ops_omits_bindings_pins():
    expected = expected_distributions("ops", "1.2.3")
    assert expected == {
        OPS_DISTRIBUTION: "1.2.3",
        COMPILER_DISTRIBUTION: "1.2.3",
        COMPILER_CORE_DISTRIBUTION: "1.2.3",
        COMPILER_BINDINGS_DISTRIBUTION: "1.2.3",
    }
    for name in bindings_runtime_pins():
        assert name not in expected


def test_expected_sdk_closes_every_product():
    expected = expected_distributions("sdk", "1.2.3")
    for name in (
        SDK_DISTRIBUTION,
        FASTFLOWLM_DISTRIBUTION,
        RUNTIME_DISTRIBUTION,
        OPS_DISTRIBUTION,
        COMPILER_DISTRIBUTION,
        COMPILER_CORE_DISTRIBUTION,
        COMPILER_BINDINGS_DISTRIBUTION,
    ):
        assert expected[name] == "1.2.3"
    for name, version in bindings_runtime_pins().items():
        assert expected[name] == version


def test_unknown_product_is_rejected():
    with pytest.raises(RuntimeError, match="unknown release resolve product"):
        expected_distributions("qualification", "1.2.3")


def test_pip_install_command_uses_wheel_file_and_abi3(tmp_path):
    wheel = tmp_path / "fastflow_sdk_ops-1.2.3-py3-none-any.whl"
    command = pip_install_command(
        python="python",
        wheel=wheel,
        find_links=tmp_path / "release",
        python_minor="3.12",
        platforms=("manylinux_2_28_x86_64",),
        target=tmp_path / "target",
    )
    assert "--no-index" not in command
    assert "--disable-pip-version-check" in command
    assert "--no-cache-dir" in command
    assert "--only-binary=:all:" in command
    assert command[command.index("--python-version") + 1] == "3.12"
    assert command[command.index("--implementation") + 1] == "cp"
    abis = [command[index + 1] for index, item in enumerate(command) if item == "--abi"]
    assert abis == [python_tag("3.12"), "abi3"]
    assert command[command.index("--platform") + 1] == "manylinux_2_28_x86_64"
    assert "linux_x86_64" not in command
    assert command[-1] == str(wheel)
    assert all("fastflow-sdk-ops==" not in item for item in command)


def test_verify_installed_allows_extra_packages(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "build_tools.resolve_release_wheel.installed_versions",
        lambda target: {
            COMPILER_DISTRIBUTION: "1.2.3",
            COMPILER_CORE_DISTRIBUTION: "1.2.3",
            COMPILER_BINDINGS_DISTRIBUTION: "1.2.3",
            "numpy": "2.4.6",
            "extra-dep": "0",
        },
    )
    verify_installed(
        tmp_path,
        {
            COMPILER_DISTRIBUTION: "1.2.3",
            COMPILER_CORE_DISTRIBUTION: "1.2.3",
            COMPILER_BINDINGS_DISTRIBUTION: "1.2.3",
        },
    )


def test_verify_installed_rejects_missing_or_wrong_version(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "build_tools.resolve_release_wheel.installed_versions",
        lambda target: {COMPILER_DISTRIBUTION: "1.2.3"},
    )
    with pytest.raises(RuntimeError, match="closed fewer packages"):
        verify_installed(
            tmp_path,
            {
                COMPILER_DISTRIBUTION: "1.2.3",
                COMPILER_CORE_DISTRIBUTION: "1.2.3",
            },
        )


def test_resolve_release_wheel_covers_policy_matrix(tmp_path, monkeypatch):
    wheel = tmp_path / "fastflow_sdk_compiler-1.2.3-py3-none-any.whl"
    wheel.write_bytes(b"pk")
    find_links = tmp_path / "release"
    find_links.mkdir()
    calls = []

    def fake_sh(command, **kwargs):
        calls.append(command)
        return None

    monkeypatch.setattr("build_tools.resolve_release_wheel.sh", fake_sh)
    monkeypatch.setattr(
        "build_tools.resolve_release_wheel.verify_installed",
        lambda target, expected: None,
    )
    monkeypatch.setattr(
        "build_tools.resolve_release_wheel.release_policy.compiler_python_versions",
        lambda: ("3.12", "3.13", "3.14"),
    )
    resolve_release_wheel(
        wheel=wheel,
        find_links=find_links,
        version="1.2.3",
        product="compiler",
        work_dir=tmp_path,
    )
    assert len(calls) == 6
    platforms = {command[command.index("--platform") + 1] for command in calls}
    minors = {command[command.index("--python-version") + 1] for command in calls}
    assert platforms == {"manylinux_2_28_x86_64", "win_amd64"}
    assert minors == {"3.12", "3.13", "3.14"}
    assert all(command[-1] == str(wheel) for command in calls)
    assert all("--no-index" not in command for command in calls)


def test_resolve_sdk_wheel_pairs_platforms_at_the_lowest_python(tmp_path, monkeypatch):
    wheel = tmp_path / "fastflow_sdk-1.2.3-py3-none-any.whl"
    wheel.write_bytes(b"pk")
    calls = []
    monkeypatch.setattr(
        "build_tools.resolve_release_wheel.sh",
        lambda command, **kwargs: calls.append(command),
    )
    monkeypatch.setattr(
        "build_tools.resolve_release_wheel.verify_installed",
        lambda target, expected: None,
    )
    monkeypatch.setattr(
        "build_tools.resolve_release_wheel.release_policy.compiler_python_versions",
        lambda: ("3.12", "3.13", "3.14"),
    )
    resolve_release_wheel(
        wheel=wheel,
        find_links=tmp_path,
        version="1.2.3",
        product="sdk",
        work_dir=tmp_path,
    )
    assert [command[command.index("--python-version") + 1] for command in calls] == [
        "3.12",
        "3.12",
    ]
    platforms = [
        tuple(
            command[index + 1]
            for index, item in enumerate(command)
            if item == "--platform"
        )
        for command in calls
    ]
    assert platforms == [
        ("linux_x86_64", "manylinux_2_28_x86_64"),
        ("win_amd64",),
    ]


def test_resolve_release_wheel_requires_the_wheel_file(tmp_path):
    with pytest.raises(RuntimeError, match="release wheel missing"):
        resolve_release_wheel(
            wheel=tmp_path / "missing.whl",
            find_links=tmp_path,
            version="1.2.3",
            product="ops",
            work_dir=tmp_path,
        )
