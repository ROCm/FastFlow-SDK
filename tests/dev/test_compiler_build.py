# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest

from build_tools.dev import compiler_build

pytestmark = pytest.mark.presubmit


def test_build_aiebu_configures_target_and_stages_only_assembler(tmp_path, monkeypatch):
    project = tmp_path / "projects"
    source = project / "aiebu"
    source.mkdir(parents=True)
    (source / "CMakeLists.txt").write_text("project(aiebu)", encoding="utf-8")
    build = tmp_path / "build" / "aiebu"
    install = tmp_path / "build" / "aiebu-install"
    assembler = build / "src" / "cpp" / "utils" / "asm" / "aiebu-asm"
    assembler.parent.mkdir(parents=True)
    assembler.write_bytes(b"assembler")

    commands = []
    monkeypatch.setattr(compiler_build.paths, "PROJ", project)
    monkeypatch.setattr(compiler_build.paths, "AIEBU_BUILD", build)
    monkeypatch.setattr(compiler_build.paths, "AIEBU_INSTALL", install)
    monkeypatch.setattr(compiler_build.paths, "is_windows", lambda: False)
    monkeypatch.setattr(
        compiler_build.environment, "compilers", lambda: ("clang", "clang++")
    )
    monkeypatch.setattr(compiler_build.environment, "cmake_env", lambda: {})
    monkeypatch.setattr(compiler_build.environment, "cmake_bin", lambda: "cmake")
    monkeypatch.setattr(
        compiler_build.environment, "cmake_path", lambda path: str(path)
    )
    monkeypatch.setattr(
        compiler_build.environment,
        "cmake_compiler_flags",
        lambda cc, cxx: [f"-DCMAKE_C_COMPILER={cc}", f"-DCMAKE_CXX_COMPILER={cxx}"],
    )
    monkeypatch.setattr(compiler_build.environment, "cmake_ccache_flags", lambda: [])
    monkeypatch.setattr(
        compiler_build.environment,
        "sh",
        lambda command, **kwargs: commands.append((command, kwargs)),
    )

    compiler_build.cmd_build_aiebu(None)

    assert commands[0][0][:4] == ["cmake", "-G", "Ninja", "-S"]
    assert "-DAIEBU_BUILD_TESTS=OFF" in commands[0][0]
    assert commands[1][0][-2:] == ["--target", "aiebu-asm"]
    assert (install / "bin" / "aiebu-asm").read_bytes() == b"assembler"
