# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import csv
import email.parser
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from build_tools import fastflow_sdk_pypi_wheel as pypi
from build_tools.artifact import verify_sha256_sidecar
from build_tools.dev import cli
from build_tools.fastflow_sdk_qualification_wheel import (
    QUALIFICATION_REQUIRES,
    QUALIFICATION_REQUIRES_PYTHON,
)
from build_tools.fastflow_sdk_wheel_common import (
    COMPILER_BINDINGS_DISTRIBUTION,
    COMPILER_BINDINGS_RUNTIME_REQUIREMENTS,
    COMPILER_CORE_DISTRIBUTION,
    COMPILER_DISTRIBUTION,
    FASTFLOWLM_DISTRIBUTION,
    OPS_DISTRIBUTION,
    QUALIFICATION_DISTRIBUTION,
    RUNTIME_DISTRIBUTION,
    SDK_DISTRIBUTION,
    _wheel_hash,
    build_wheel,
)

pytestmark = [
    pytest.mark.compiler_stack,
    pytest.mark.fastflowlm,
    pytest.mark.presubmit,
]

VERSION = "0.1.0"
PLATFORM_TAGS = {"linux": "manylinux_2_28_x86_64", "windows": "win_amd64"}


def _component_specs(version: str = VERSION) -> dict[str, dict]:
    def pin(name):
        return f"{name} =={version}"

    return {
        SDK_DISTRIBUTION: dict(
            package="fastflow_sdk_meta",
            platform=False,
            requires=(
                pin(FASTFLOWLM_DISTRIBUTION),
                pin(COMPILER_DISTRIBUTION),
                pin(OPS_DISTRIBUTION),
            ),
            requires_python=">=3.12,<3.15",
        ),
        RUNTIME_DISTRIBUTION: dict(
            package="fastflow_sdk_runtime",
            platform=True,
            executable="bin/fastflow",
            console_scripts={"fastflow": "fastflow_sdk_runtime:main"},
        ),
        FASTFLOWLM_DISTRIBUTION: dict(
            package="fastflow_sdk_fastflowlm",
            platform=True,
            executable="bin/flm",
            requires=(pin(RUNTIME_DISTRIBUTION),),
            console_scripts={"flm": "fastflow_sdk_fastflowlm:main"},
            licenses={"vcpkg/curl/copyright": "curl"},
        ),
        QUALIFICATION_DISTRIBUTION: dict(
            package="fastflow_sdk_qualification",
            platform=False,
            requires=(pin(FASTFLOWLM_DISTRIBUTION), *QUALIFICATION_REQUIRES),
            requires_python=QUALIFICATION_REQUIRES_PYTHON,
            console_scripts={"fastflow-qualify": "fastflow_sdk_qualification.cli:main"},
        ),
        COMPILER_DISTRIBUTION: dict(
            package="fastflow_sdk_compiler_meta",
            platform=False,
            requires=(
                pin(COMPILER_CORE_DISTRIBUTION),
                pin(COMPILER_BINDINGS_DISTRIBUTION),
            ),
            requires_python=">=3.12,<3.15",
        ),
        COMPILER_CORE_DISTRIBUTION: dict(
            package="fastflow_sdk_compiler",
            platform=True,
            executable="bin/aie-opt",
            extra={"lib/aie/libAIEAggregateCAPI.so": b"\x7fELF core"},
            requires_python=">=3.12,<3.15",
            console_scripts={"aie-opt": "fastflow_sdk_compiler:aie_opt"},
        ),
        COMPILER_BINDINGS_DISTRIBUTION: dict(
            package="fastflow_sdk_compiler_bindings",
            platform=True,
            abi3=True,
            extra={"../aie/_mlir_libs/_aie.abi3.so": b"\x7fELF binding"},
            requires=(
                pin(COMPILER_CORE_DISTRIBUTION),
                *COMPILER_BINDINGS_RUNTIME_REQUIREMENTS,
            ),
            requires_python=">=3.12,<3.15",
        ),
        OPS_DISTRIBUTION: dict(
            package="fastflow_sdk_ops",
            platform=False,
            requires=(pin(COMPILER_DISTRIBUTION),),
            requires_python=">=3.12,<3.15",
        ),
    }


def _build_component(tmp_path, distribution, spec, platform, version):
    tree = tmp_path / "trees" / f"{distribution}-{platform}"
    package = tree / spec["package"]
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(f"# {distribution}\n", encoding="utf-8")
    licenses = {"fastflow-sdk-LICENSE": "MIT", **spec.get("licenses", {})}
    for name, text in licenses.items():
        target = package / "licenses" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    if "executable" in spec:
        target = package / spec["executable"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f"#!{distribution}\n".encode())
        target.chmod(0o755)
    for name, data in spec.get("extra", {}).items():
        target = (package / name).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    if spec["platform"]:
        python_tag = "cp312-abi3" if spec.get("abi3") else "py3-none"
        tag = f"{python_tag}-{PLATFORM_TAGS[platform]}"
    else:
        tag = "py3-none-any"
    return build_wheel(
        tree,
        distribution,
        version,
        tmp_path / "release",
        summary=distribution,
        tag=tag,
        requires=spec.get("requires", ()),
        console_scripts=spec.get("console_scripts"),
        requires_python=spec.get("requires_python", ">=3.11"),
        root_is_purelib=not spec["platform"],
    )


def _write_sha256sums(release: Path) -> None:
    lines = [
        f"{hashlib.sha256(wheel.read_bytes()).hexdigest()}  {wheel.name}"
        for wheel in sorted(release.glob("*.whl"))
    ]
    (release / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def release(tmp_path):
    """A fake release with one wheel per component and platform."""
    specs = _component_specs()
    for platform in pypi.PYPI_PLATFORMS:
        for distribution, spec in specs.items():
            if spec["platform"] or platform == "linux":
                _build_component(tmp_path, distribution, spec, platform, VERSION)
    release = tmp_path / "release"
    for sidecar in release.glob("*.sha256"):
        sidecar.unlink()
    _write_sha256sums(release)
    return release


def _replace_component(release, tmp_path, distribution, platform, **changes):
    spec = {**_component_specs()[distribution], **changes}
    prefix = distribution.replace("-", "_") + "-"
    for wheel in release.glob(f"{prefix}*.whl"):
        if not spec["platform"] or PLATFORM_TAGS[platform] in wheel.name:
            wheel.unlink()
    wheel = _build_component(
        tmp_path / "replaced", distribution, spec, platform, VERSION
    )
    target = release / wheel.name
    wheel.replace(target)
    _write_sha256sums(release)
    return target


def _retag(wheel: Path, platform_tag: str) -> Path:
    name, version, python_tag, abi_tag, old = wheel.name[:-4].split("-")
    target = wheel.with_name(
        f"{name}-{version}-{python_tag}-{abi_tag}-{platform_tag}.whl"
    )
    dist_info = f"{name}-{version}.dist-info"
    rows = []
    with zipfile.ZipFile(wheel) as source, zipfile.ZipFile(target, "w") as output:
        for info in source.infolist():
            if info.filename == f"{dist_info}/RECORD":
                continue
            data = source.read(info)
            if info.filename == f"{dist_info}/WHEEL":
                data = data.replace(old.encode(), platform_tag.encode())
            output.writestr(info, data)
            rows.append((info.filename, _wheel_hash(data), str(len(data))))
        stream = io.StringIO()
        csv.writer(stream, lineterminator="\n").writerows(
            [*rows, (f"{dist_info}/RECORD", "", "")]
        )
        output.writestr(f"{dist_info}/RECORD", stream.getvalue())
    wheel.unlink()
    _write_sha256sums(wheel.parent)
    return target


def _members(wheel: Path) -> dict[str, tuple[bytes, int]]:
    with zipfile.ZipFile(wheel) as archive:
        return {
            info.filename: (archive.read(info), info.external_attr >> 16)
            for info in archive.infolist()
        }


@pytest.mark.parametrize("platform", pypi.PYPI_PLATFORMS)
def test_merges_every_component_file_unchanged(release, tmp_path, platform):
    wheel = pypi.build_pypi_wheel(release, platform, tmp_path / "out")

    python_tag = "cp312-abi3"
    assert wheel.name == (
        f"fastflow_sdk-{VERSION}-{python_tag}-{PLATFORM_TAGS[platform]}.whl"
    )
    verify_sha256_sidecar(wheel, kind="wheel")
    merged = _members(wheel)
    components = pypi.select_component_wheels(release, platform)
    assert set(components) == set(pypi.PYPI_COMPONENTS)
    payload = 0
    for distribution, component in components.items():
        dist_info = f"{distribution.replace('-', '_')}-{VERSION}.dist-info/"
        for name, (data, mode) in _members(component).items():
            if name.startswith(dist_info):
                continue
            assert merged[name] == (data, mode), name
            payload += 1
    dist_info = f"fastflow_sdk-{VERSION}.dist-info/"
    assert payload == len([name for name in merged if not name.startswith(dist_info)])
    assert merged["fastflow_sdk_fastflowlm/bin/flm"][1] & 0o111
    with zipfile.ZipFile(wheel) as archive:
        assert {info.create_system for info in archive.infolist()} == {3}


def test_writes_merged_sdk_metadata(release, tmp_path):
    wheel = pypi.build_pypi_wheel(release, "linux", tmp_path / "out")

    dist_info = f"fastflow_sdk-{VERSION}.dist-info"
    members = _members(wheel)
    parser = email.parser.Parser()
    metadata = parser.parsestr(members[f"{dist_info}/METADATA"][0].decode())
    tags = parser.parsestr(members[f"{dist_info}/WHEEL"][0].decode())
    assert metadata["Name"] == SDK_DISTRIBUTION
    assert metadata["Version"] == VERSION
    assert metadata["Requires-Python"] == ">=3.12,<3.15"
    assert metadata["Keywords"].split(",") == list(pypi.PYPI_KEYWORDS)
    classifiers = metadata.get_all("Classifier")
    assert classifiers == sorted(classifiers)
    assert set(classifiers) == {
        *pypi.PYPI_CLASSIFIERS,
        *(f"Programming Language :: Python :: 3.{minor}" for minor in (12, 13, 14)),
    }
    assert not [name for name in classifiers if name.startswith("License ::")]
    windows = pypi.build_pypi_wheel(release, "windows", tmp_path / "windows")
    windows_metadata = parser.parsestr(
        _members(windows)[f"{dist_info}/METADATA"][0].decode()
    )
    assert windows_metadata.get_all("Classifier") == classifiers
    assert sorted(metadata.get_all("Requires-Dist")) == sorted(
        {*COMPILER_BINDINGS_RUNTIME_REQUIREMENTS, *QUALIFICATION_REQUIRES}
    )
    assert metadata["License-Expression"] == (
        "MIT AND Apache-2.0 AND LicenseRef-FastFlowLM-Proprietary-Kernel-Terms "
        "AND (Apache-2.0 WITH LLVM-exception)"
    )
    license_files = metadata.get_all("License-File")
    assert f"{FASTFLOWLM_DISTRIBUTION}/vcpkg/curl/copyright" in license_files
    for name in license_files:
        assert f"{dist_info}/licenses/{name}" in members
    assert "pip install fastflow-sdk" in metadata.get_payload()
    assert tags["Tag"] == "cp312-abi3-manylinux_2_28_x86_64"
    assert tags["Root-Is-Purelib"] == "false"
    entry_points = members[f"{dist_info}/entry_points.txt"][0].decode()
    for script in ("fastflow", "flm", "fastflow-qualify", "aie-opt"):
        assert f"\n{script} = " in f"\n{entry_points}"
    assert sorted(
        name.removeprefix(f"{dist_info}/sboms/")
        for name in members
        if name.startswith(f"{dist_info}/sboms/")
    ) == sorted(f"{name}.cyclonedx.json" for name in pypi.PYPI_COMPONENTS)

    record = json.loads(members[f"{dist_info}/{pypi.SDK_COMPONENTS_FILE}"][0])
    components = pypi.select_component_wheels(release, "linux")
    assert record["distribution"] == SDK_DISTRIBUTION
    assert record["tag"] == tags["Tag"]
    assert {
        component["distribution"]: (component["wheel"], component["sha256"])
        for component in record["components"]
    } == {
        distribution: (path.name, hashlib.sha256(path.read_bytes()).hexdigest())
        for distribution, path in components.items()
    }


def test_build_is_byte_for_byte_reproducible(release, tmp_path):
    first = pypi.build_pypi_wheel(release, "windows", tmp_path / "first")
    second = pypi.build_pypi_wheel(release, "windows", tmp_path / "second")

    assert first.read_bytes() == second.read_bytes()


def test_rejects_a_wheel_that_differs_from_the_release_digests(release, tmp_path):
    sums = release / "SHA256SUMS"
    lines = sums.read_text(encoding="utf-8").splitlines()
    sums.write_text(
        "\n".join(
            ("0" * 64 + line[64:]) if "runtime" in line else line for line in lines
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="digest mismatch"):
        pypi.build_pypi_wheel(release, "linux", tmp_path / "out")
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").iterdir())


def test_rejects_a_wheel_missing_from_the_release_digests(release, tmp_path):
    sums = release / "SHA256SUMS"
    lines = sums.read_text(encoding="utf-8").splitlines()
    sums.write_text(
        "\n".join(line for line in lines if "_ops-" not in line) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="does not list"):
        pypi.build_pypi_wheel(release, "linux", tmp_path / "out")


def test_rejects_a_component_whose_record_does_not_match(release, tmp_path):
    wheel = next(release.glob("fastflow_sdk_ops-*.whl"))
    tampered = wheel.with_suffix(".tmp")
    with zipfile.ZipFile(wheel) as source, zipfile.ZipFile(tampered, "w") as output:
        for info in source.infolist():
            data = source.read(info)
            if info.filename.endswith("__init__.py"):
                data += b"# tampered\n"
            output.writestr(info, data)
    tampered.replace(wheel)
    _write_sha256sums(release)

    with pytest.raises(RuntimeError, match="RECORD does not match"):
        pypi.build_pypi_wheel(release, "linux", tmp_path / "out")
    assert not list((tmp_path / "out").glob("*.whl"))


def test_rejects_dist_info_files_that_do_not_match_their_record(release, tmp_path):
    wheel = next(release.glob("fastflow_sdk_ops-*.whl"))
    tampered = wheel.with_suffix(".tmp")
    with zipfile.ZipFile(wheel) as source, zipfile.ZipFile(tampered, "w") as output:
        for info in source.infolist():
            data = source.read(info)
            if ".dist-info/licenses/" in info.filename:
                data += b"tampered\n"
            output.writestr(info, data)
    tampered.replace(wheel)
    _write_sha256sums(release)

    with pytest.raises(RuntimeError, match=r"RECORD does not match .*\.dist-info/"):
        pypi.build_pypi_wheel(release, "linux", tmp_path / "out")


def test_requires_exactly_one_wheel_per_component(release, tmp_path):
    next(release.glob("fastflow_sdk_compiler_core-*win_amd64.whl")).unlink()

    with pytest.raises(RuntimeError, match="fastflow-sdk-compiler-core=0"):
        pypi.build_pypi_wheel(release, "windows", tmp_path / "out")


def test_rejects_linux_wheels_pypi_does_not_accept(release, tmp_path):
    _retag(
        next(release.glob("fastflow_sdk_fastflowlm-*manylinux*.whl")), "linux_x86_64"
    )

    with pytest.raises(RuntimeError, match="PyPI accepts only manylinux"):
        pypi.build_pypi_wheel(release, "linux", tmp_path / "out")


def test_uses_the_most_restrictive_manylinux_tag(release, tmp_path):
    _retag(
        next(release.glob("fastflow_sdk_runtime-*manylinux*.whl")),
        "manylinux_2_34_x86_64",
    )

    wheel = pypi.build_pypi_wheel(release, "linux", tmp_path / "out")

    assert wheel.name.endswith("-cp312-abi3-manylinux_2_34_x86_64.whl")


def test_rejects_conflicting_external_requirements(release, tmp_path):
    _replace_component(
        release,
        tmp_path,
        QUALIFICATION_DISTRIBUTION,
        "linux",
        requires=(f"{FASTFLOWLM_DISTRIBUTION} =={VERSION}", "numpy ==2.5.0"),
    )

    with pytest.raises(RuntimeError, match="conflicting requirements for numpy"):
        pypi.build_pypi_wheel(release, "linux", tmp_path / "out")


def test_rejects_a_dependency_on_an_unbundled_component(release, tmp_path):
    _replace_component(
        release,
        tmp_path,
        OPS_DISTRIBUTION,
        "linux",
        requires=(f"fastflow-sdk-extras =={VERSION}",),
    )

    with pytest.raises(RuntimeError, match="does not bundle"):
        pypi.build_pypi_wheel(release, "linux", tmp_path / "out")


def test_rejects_files_shipped_by_two_components(release, tmp_path):
    _replace_component(
        release,
        tmp_path,
        OPS_DISTRIBUTION,
        "linux",
        extra={"../aie/_mlir_libs/_aie.abi3.so": b"\x7fELF shadow"},
    )

    with pytest.raises(RuntimeError, match="shipped by both"):
        pypi.build_pypi_wheel(release, "linux", tmp_path / "out")


def test_rejects_components_with_different_versions(release, tmp_path):
    spec = _component_specs()[OPS_DISTRIBUTION]
    next(release.glob("fastflow_sdk_ops-*.whl")).unlink()
    _build_component(tmp_path / "other", OPS_DISTRIBUTION, spec, "linux", "0.1.1")
    for wheel in (tmp_path / "other" / "release").glob("*.whl"):
        wheel.replace(release / wheel.name)
    _write_sha256sums(release)

    with pytest.raises(RuntimeError, match="different versions"):
        pypi.build_pypi_wheel(release, "linux", tmp_path / "out")


@pytest.mark.parametrize(
    ("specifiers", "expected"),
    [
        ((">=3.12,<3.15", ">=3.11"), ">=3.12,<3.15"),
        ((">=3.11,<3.15", ">=3.12,<3.14"), ">=3.12,<3.14"),
        ((">=3.12",), ">=3.12"),
    ],
)
def test_requires_python_is_the_intersection(specifiers, expected):
    components = [
        pypi.ComponentWheel(
            distribution=f"c{index}",
            path=Path(f"c{index}.whl"),
            sha256="",
            version=VERSION,
            tag=("py3", "none", "any"),
            dist_info="",
            metadata=email.parser.Parser().parsestr(f"Requires-Python: {value}\n"),
            entry_points={},
            records={},
        )
        for index, value in enumerate(specifiers)
    ]

    assert pypi.merge_requires_python(components) == expected


@pytest.mark.parametrize(
    ("requires_python", "expected"),
    [
        (">=3.12,<3.15", ["3.12", "3.13", "3.14"]),
        (">=3.12.1,<3.14.2", ["3.12", "3.13", "3.14"]),
        (">=3.12,<4", []),
        (">=3.12", []),
    ],
)
def test_python_version_classifiers_follow_requires_python(requires_python, expected):
    assert pypi.python_version_classifiers(requires_python) == [
        f"Programming Language :: Python :: {version}" for version in expected
    ]


def test_license_terms_split_only_at_the_top_level():
    assert pypi._and_terms("MIT AND (Apache-2.0 WITH LLVM-exception)") == [
        "MIT",
        "(Apache-2.0 WITH LLVM-exception)",
    ]
    assert pypi._and_terms("(MIT AND Apache-2.0)") == ["(MIT AND Apache-2.0)"]
    with pytest.raises(RuntimeError, match="unsupported license"):
        pypi._and_terms("MIT OR Apache-2.0")


def test_audit_treats_bundled_libraries_as_provided(release, tmp_path):
    wheel = pypi.build_pypi_wheel(release, "linux", tmp_path / "out")

    assert pypi.bundled_shared_libraries(wheel) == {
        "libAIEAggregateCAPI.so",
        "_aie.abi3.so",
    }


def test_cli_registers_package_and_audit_commands(tmp_path):
    parser = cli.create_parser()
    package = parser.parse_args(
        ["package-pypi-wheel", "--release-dir", str(tmp_path), "--platform", "linux"]
    )
    audit = parser.parse_args(["audit-pypi-wheel", str(tmp_path / "x.whl")])

    assert package._handler is pypi.cmd_package_pypi_wheel
    assert package.dir == str(pypi.PYPI_WHEEL_ARTIFACTS)
    assert audit._handler is pypi.cmd_audit_pypi_wheel
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["package-pypi-wheel", "--release-dir", "x", "--platform", "macos"]
        )
