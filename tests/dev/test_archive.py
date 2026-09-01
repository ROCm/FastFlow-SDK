# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import io
import stat
import tarfile
import warnings
import zipfile

import pytest

from build_tools.archive import (
    create_deterministic_tar_gz,
    create_deterministic_tar_zst,
    create_deterministic_zip,
    extract_archive,
    extract_archive_atomic,
    read_member,
)

pytestmark = pytest.mark.presubmit


def _tar_gz(path, members):
    with tarfile.open(path, "w:gz") as archive:
        for item in members:
            member, data = item if isinstance(item, tuple) else (item, b"x")
            archive.addfile(member, io.BytesIO(data) if member.isfile() else None)


def _regular_tar_member(name, data=b"x"):
    member = tarfile.TarInfo(name)
    member.size = len(data)
    return member, data


def test_safe_zip_extraction_rejects_traversal(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../outside", b"x")
    with pytest.raises(RuntimeError, match="escapes extraction root"):
        extract_archive(archive, tmp_path / "extract")


def test_safe_tar_extraction_rejects_traversal(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    _tar_gz(archive, [_regular_tar_member("../outside")])
    with pytest.raises(RuntimeError, match="escapes extraction root"):
        extract_archive(archive, tmp_path / "extract")


def test_safe_extraction_rejects_duplicate_normalized_paths(tmp_path):
    archive = tmp_path / "duplicate.zip"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("bundle/./payload", b"first")
            output.writestr("bundle/payload", b"second")
    with pytest.raises(RuntimeError, match="duplicate normalized path"):
        extract_archive(archive, tmp_path / "extract")


def test_safe_tar_extraction_rejects_duplicate_normalized_paths(tmp_path):
    archive = tmp_path / "duplicate.tar.gz"
    _tar_gz(
        archive,
        [
            _regular_tar_member("bundle/./payload"),
            _regular_tar_member("bundle/payload"),
        ],
    )
    with pytest.raises(RuntimeError, match="duplicate normalized path"):
        extract_archive(archive, tmp_path / "extract")


@pytest.mark.parametrize("kind", ["zip", "tar"])
def test_safe_extraction_rejects_file_directory_collisions(tmp_path, kind):
    if kind == "zip":
        archive = tmp_path / "collision.zip"
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("bundle", b"file")
            output.writestr("bundle/payload", b"nested")
    else:
        archive = tmp_path / "collision.tar.gz"
        _tar_gz(
            archive,
            [
                _regular_tar_member("bundle"),
                _regular_tar_member("bundle/payload"),
            ],
        )
    with pytest.raises(RuntimeError, match="file/directory normalized collision"):
        extract_archive(archive, tmp_path / "extract")


def test_safe_extraction_rejects_zip_links(tmp_path):
    archive = tmp_path / "link.zip"
    with zipfile.ZipFile(archive, "w") as output:
        member = zipfile.ZipInfo("bundle/link")
        member.external_attr = (stat.S_IFLNK | 0o777) << 16
        output.writestr(member, "../target")
    with pytest.raises(RuntimeError, match="links and special files"):
        extract_archive(archive, tmp_path / "extract")


def test_safe_extraction_rejects_zip_special_files(tmp_path):
    archive = tmp_path / "special.zip"
    with zipfile.ZipFile(archive, "w") as output:
        member = zipfile.ZipInfo("bundle/fifo")
        member.external_attr = (stat.S_IFIFO | 0o644) << 16
        output.writestr(member, b"")
    with pytest.raises(RuntimeError, match="links and special files"):
        extract_archive(archive, tmp_path / "extract")


def test_safe_extraction_rejects_tar_links(tmp_path):
    archive = tmp_path / "link.tar.gz"
    member = tarfile.TarInfo("bundle/link")
    member.type = tarfile.SYMTYPE
    member.linkname = "../target"
    _tar_gz(archive, [member])
    with pytest.raises(RuntimeError, match="links and special files"):
        extract_archive(archive, tmp_path / "extract")


def test_safe_extraction_rejects_tar_special_files(tmp_path):
    archive = tmp_path / "special.tar.gz"
    member = tarfile.TarInfo("bundle/device")
    member.type = tarfile.CHRTYPE
    _tar_gz(archive, [member])
    with pytest.raises(RuntimeError, match="links and special files"):
        extract_archive(archive, tmp_path / "extract")


@pytest.mark.parametrize("kind", ["zip", "tar"])
def test_preflight_leaves_destination_unchanged_for_late_invalid_member(tmp_path, kind):
    destination = tmp_path / "extract"
    destination.mkdir()
    existing = destination / "existing"
    existing.write_bytes(b"unchanged")
    if kind == "zip":
        archive = tmp_path / "late-invalid.zip"
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("new", b"would-be-written")
            output.writestr("../outside", b"invalid")
    else:
        archive = tmp_path / "late-invalid.tar.gz"
        _tar_gz(
            archive,
            [
                _regular_tar_member("new", b"would-be-written"),
                _regular_tar_member("../outside", b"invalid"),
            ],
        )

    with pytest.raises(RuntimeError, match="escapes extraction root"):
        extract_archive(archive, destination)
    assert existing.read_bytes() == b"unchanged"
    assert sorted(path.name for path in destination.iterdir()) == ["existing"]


@pytest.mark.parametrize("kind", ["zip", "tar"])
def test_read_member_validates_unrelated_members_before_reading(tmp_path, kind):
    if kind == "zip":
        archive = tmp_path / "invalid.zip"
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("wanted", b"payload")
            member = zipfile.ZipInfo("unrelated-link")
            member.external_attr = (stat.S_IFLNK | 0o777) << 16
            output.writestr(member, "target")
    else:
        archive = tmp_path / "invalid.tar.gz"
        link = tarfile.TarInfo("unrelated-link")
        link.type = tarfile.SYMTYPE
        link.linkname = "target"
        _tar_gz(archive, [_regular_tar_member("wanted", b"payload"), link])

    with pytest.raises(RuntimeError, match="links and special files"):
        read_member(archive, "wanted")


@pytest.mark.parametrize("kind", ["zip", "tar"])
def test_read_member_enforces_size_limit_after_archive_validation(tmp_path, kind):
    if kind == "zip":
        archive = tmp_path / "payload.zip"
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("wanted", b"payload")
    else:
        archive = tmp_path / "payload.tar.gz"
        _tar_gz(archive, [_regular_tar_member("wanted", b"payload")])

    assert read_member(archive, "wanted", size_limit=7) == b"payload"
    with pytest.raises(RuntimeError, match="expected exactly one regular member"):
        read_member(archive, "wanted", size_limit=6)


def test_safe_extraction_rejects_existing_destination_links(tmp_path):
    destination = tmp_path / "extract"
    destination.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    try:
        (destination / "linked").symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"host cannot create directory symlinks: {error}")
    archive = tmp_path / "payload.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("linked/payload", b"x")
    with pytest.raises(RuntimeError, match="existing link"):
        extract_archive(archive, destination)
    assert not (target / "payload").exists()


def test_atomic_extraction_rejects_files_beside_the_single_root(tmp_path):
    archive = tmp_path / "payload.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("bundle/payload", b"x")
        output.writestr("unexpected", b"x")
    with pytest.raises(RuntimeError, match="exactly one root directory"):
        extract_archive_atomic(
            archive,
            tmp_path / "destination",
            allowed_root=tmp_path,
        )


def test_zip_and_gzip_tar_are_deterministic_and_roundtrip(tmp_path):
    tree = tmp_path / "tree"
    (tree / "bin").mkdir(parents=True)
    payload = tree / "bin" / "tool"
    payload.write_bytes(b"payload")
    payload.chmod(0o755)

    first_zip = tmp_path / "first.zip"
    second_zip = tmp_path / "second.zip"
    create_deterministic_zip(tree, first_zip, arcname="tree", compresslevel=6)
    create_deterministic_zip(tree, second_zip, arcname="tree", compresslevel=6)
    assert first_zip.read_bytes() == second_zip.read_bytes()

    first_tar = tmp_path / "first.tar.gz"
    second_tar = tmp_path / "second.tar.gz"
    create_deterministic_tar_gz(tree, first_tar, arcname="tree", compresslevel=6)
    create_deterministic_tar_gz(tree, second_tar, arcname="tree", compresslevel=6)
    assert first_tar.read_bytes() == second_tar.read_bytes()
    extracted = tmp_path / "gzip-extracted"
    extract_archive(first_tar, extracted)
    assert (extracted / "tree" / "bin" / "tool").read_bytes() == b"payload"


def test_zstd_tar_is_deterministic_and_roundtrips(tmp_path):
    pytest.importorskip("zstandard")
    tree = tmp_path / "tree"
    (tree / "bin").mkdir(parents=True)
    (tree / "bin" / "tool").write_bytes(b"payload")
    first = tmp_path / "first.tar.zst"
    second = tmp_path / "second.tar.zst"
    create_deterministic_tar_zst(tree, first, arcname="tree", level=6)
    create_deterministic_tar_zst(tree, second, arcname="tree", level=6)
    assert first.read_bytes() == second.read_bytes()
    extracted = tmp_path / "zstd-extracted"
    extract_archive(first, extracted)
    assert (extracted / "tree" / "bin" / "tool").read_bytes() == b"payload"
