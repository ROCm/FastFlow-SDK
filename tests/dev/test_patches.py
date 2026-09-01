# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import contextlib
import hashlib
import io
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from build_tools import paths  # noqa: E402
from build_tools.dev import patches  # noqa: E402

pytestmark = pytest.mark.presubmit


class TestPatches:
    def test_paths_are_derived_from_repository_root(self):
        assert paths.ROOT == ROOT
        assert paths.PROJ == ROOT / "projects"
        assert paths.PATCHES == ROOT / "patches"
        assert paths.FETCH_CACHE == ROOT / ".cache" / "find-links"

    def test_eudsl_cache_uses_shared_sha256_file(self, tmp_path):
        pin = {"sha256": "expected"}
        archive = tmp_path / "eudsl.tar.gz"
        archive.write_bytes(b"archive")
        with (
            mock.patch.object(patches, "eudsl_sdist", return_value=(archive, pin)),
            mock.patch.object(
                patches, "sha256_file", return_value="expected"
            ) as digest,
        ):
            assert patches.eudsl_cache_ok()
        digest.assert_called_once_with(archive)

    def test_patch_state_retries_with_whitespace_tolerance(self):
        results = [
            mock.Mock(returncode=1),
            mock.Mock(returncode=1),
            mock.Mock(returncode=1),
            mock.Mock(returncode=0),
        ]
        with mock.patch.object(patches.subprocess, "run", side_effect=results) as run:
            state = patches._patch_state(Path("change.patch"), Path("source"))

        assert state == ("appliable", ("--ignore-space-change",))
        assert len(run.call_args_list) == 4
        assert "--ignore-space-change" in run.call_args_list[-1].args[0]

    def test_patch_validation_index_uses_short_runner_temp(self, tmp_path, monkeypatch):
        runner_temp = tmp_path / "runner-temp"
        monkeypatch.setenv("RUNNER_TEMP", str(runner_temp))
        with (
            mock.patch.object(
                patches.tempfile,
                "mkdtemp",
                side_effect=RuntimeError("stop after path selection"),
            ) as mkdtemp,
            pytest.raises(RuntimeError, match="stop after path selection"),
        ):
            patches.declared_patch_expected_state("llvm-project", Path("llvm"))
        mkdtemp.assert_called_once_with(
            prefix="rns-patch-",
            dir=runner_temp.resolve(),
        )

    def test_quick_doctor_skips_replay_for_unambiguous_patches(self):
        patch_files = [Path("0001.patch"), Path("0002.patch")]
        with (
            mock.patch.object(
                patches,
                "_patch_state",
                side_effect=[("applied", ()), ("appliable", ())],
            ),
            mock.patch.object(patches, "declared_patch_expected_state") as replay,
        ):
            states = patches.doctor_patch_states(
                "component", Path("source"), patch_files, deep=False
            )

        assert states == ["applied", "appliable"]
        replay.assert_not_called()

    def test_deep_doctor_replays_even_unambiguous_patches(self):
        patch_files = [Path("0001.patch")]
        expected = {"revision": "abc"}
        with (
            mock.patch.object(patches, "_patch_state", return_value=("applied", ())),
            mock.patch.object(patches, "git_source_state", return_value=expected),
            mock.patch.object(
                patches,
                "declared_patch_expected_state",
                return_value=expected,
            ) as replay,
        ):
            states = patches.doctor_patch_states(
                "component", Path("source"), patch_files, deep=True
            )

        assert states == ["applied"]
        replay.assert_called_once_with("component", Path("source"))

    def test_valid_replay_marks_ambiguous_patch_pending(self):
        expected = {"revision": "abc"}
        with (
            mock.patch.object(patches, "_patch_state", return_value=("broken", ())),
            mock.patch.object(
                patches, "git_source_state", return_value={"revision": "other"}
            ),
            mock.patch.object(
                patches,
                "declared_patch_expected_state",
                return_value=expected,
            ),
        ):
            states = patches.doctor_patch_states(
                "component", Path("source"), [Path("0001.patch")]
            )

        assert states == ["pending"]

    def test_validate_declared_state_reports_changed_paths(self):
        expected = {
            "revision": "abc",
            "changed_files": [{"path": "a.txt", "sha256": "one"}],
        }
        actual = {
            "revision": "abc",
            "changed_files": [{"path": "b.txt", "sha256": "two"}],
        }
        with (
            mock.patch.object(
                patches, "declared_patch_expected_state", return_value=expected
            ),
            mock.patch.object(patches, "git_source_state", return_value=actual),
            pytest.raises(
                RuntimeError,
                match=(
                    r"component source differs from its pin plus declared patches: "
                    r"a\.txt, b\.txt"
                ),
            ),
        ):
            patches.validate_declared_patch_state("component", Path("source"))

    def test_fetch_for_non_compiler_sources_skips_eudsl_prefetch(self):
        with (
            mock.patch.object(patches, "apply_patches") as apply,
            mock.patch.object(patches, "eudsl_sdist") as sdist,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            patches.cmd_fetch(object(), {"hrx"})

        apply.assert_called_once_with({"hrx"})
        sdist.assert_not_called()

    def test_fetch_removes_archive_when_download_hash_mismatches(self, tmp_path):
        cache = tmp_path / "cache"
        archive = cache / "eudsl.tar.gz"
        pin = {
            "package": "eudsl",
            "version": "1",
            "find_links": "https://example.invalid",
            "sha256": hashlib.sha256(b"expected").hexdigest(),
        }

        def download(_command):
            archive.write_bytes(b"wrong")
            return subprocess.CompletedProcess([], 0)

        with (
            mock.patch.object(paths, "FETCH_CACHE", cache),
            mock.patch.object(patches, "eudsl_sdist", return_value=(archive, pin)),
            mock.patch.object(patches, "eudsl_cache_ok", return_value=False),
            mock.patch.object(patches, "apply_patches"),
            mock.patch.object(patches, "venv_py", return_value=Path("python")),
            mock.patch.object(patches, "sh", side_effect=download),
            pytest.raises(SystemExit, match="fetch: sha256 mismatch"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            patches.cmd_fetch(object())

        assert not archive.exists()
