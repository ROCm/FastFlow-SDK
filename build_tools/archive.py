# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Safe, deterministic cross-product archive mechanics."""

from __future__ import annotations

import gzip
import os
import posixpath
import shutil
import stat
import subprocess
import tarfile
import tempfile
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

FIXED_MTIME = 946684800  # 2000-01-01T00:00:00Z
ZIP_DATE_TIME = (2000, 1, 1, 0, 0, 0)


def zstandard_module():
    """Loads the optional zstandard dependency only when a zstd archive needs it."""
    try:
        import zstandard
    except ImportError as error:
        raise RuntimeError(
            "the zstandard Python package is required for .tar.zst archives; "
            "run: python dev.py setup"
        ) from error
    return zstandard


def normalize_member_name(name: str) -> str:
    """Returns a safe POSIX member path used for collision detection."""
    if not name or "\0" in name:
        raise RuntimeError(f"invalid archive member path: {name!r}")
    portable = name.replace("\\", "/")
    if PurePosixPath(portable).is_absolute() or PureWindowsPath(name).is_absolute():
        raise RuntimeError(f"archive path escapes extraction root: {name}")
    normalized = posixpath.normpath(portable)
    if normalized == ".":
        return "."
    if normalized == ".." or normalized.startswith("../"):
        raise RuntimeError(f"archive path escapes extraction root: {name}")
    if PureWindowsPath(normalized).drive:
        raise RuntimeError(f"archive path escapes extraction root: {name}")
    return PurePosixPath(normalized).as_posix()


def safe_member_path(destination: Path, name: str) -> Path:
    normalized = normalize_member_name(name)
    destination = destination.resolve()
    candidate = (
        destination
        if normalized == "."
        else destination.joinpath(*PurePosixPath(normalized).parts)
    )
    current = destination
    for part in PurePosixPath(normalized).parts:
        if part == ".":
            continue
        current /= part
        if current.is_symlink():
            raise RuntimeError(f"archive path traverses an existing link: {name}")
    target = candidate.resolve()
    if target != destination and destination not in target.parents:
        raise RuntimeError(f"archive path escapes extraction root: {name}")
    return target


@dataclass(frozen=True)
class _ArchiveMember:
    name: str
    normalized: str
    is_directory: bool
    size: int


def _member_key(normalized: str) -> str:
    return os.path.normcase(normalized)


def _validate_member_structure(
    members: Iterator[_ArchiveMember],
    *,
    destination: Path | None = None,
) -> list[_ArchiveMember]:
    validated: list[_ArchiveMember] = []
    entries: dict[str, _ArchiveMember] = {}
    for member in members:
        key = _member_key(member.normalized)
        if key in entries:
            raise RuntimeError(f"duplicate normalized path in archive: {member.name}")
        parts = PurePosixPath(member.normalized).parts
        for index in range(1, len(parts)):
            ancestor = entries.get(
                _member_key(PurePosixPath(*parts[:index]).as_posix())
            )
            if ancestor is not None and not ancestor.is_directory:
                raise RuntimeError(
                    "file/directory normalized collision in archive: "
                    f"{ancestor.name} and {member.name}"
                )
        if not member.is_directory:
            prefix = key + os.sep
            descendant = next(
                (
                    existing
                    for existing_key, existing in entries.items()
                    if existing_key.startswith(prefix)
                ),
                None,
            )
            if descendant is not None:
                raise RuntimeError(
                    "file/directory normalized collision in archive: "
                    f"{member.name} and {descendant.name}"
                )
        if member.normalized == "." and not member.is_directory:
            raise RuntimeError(
                f"file/directory normalized collision in archive: {member.name}"
            )
        if destination is not None:
            safe_member_path(destination, member.name)
        entries[key] = member
        validated.append(member)
    return validated


def _zip_members(source: zipfile.ZipFile) -> Iterator[_ArchiveMember]:
    for member in source.infolist():
        mode = (member.external_attr >> 16) & 0xFFFF
        file_type = stat.S_IFMT(mode)
        is_directory = member.is_dir()
        allowed_type = stat.S_IFDIR if is_directory else stat.S_IFREG
        if file_type not in (0, allowed_type):
            raise RuntimeError(
                f"links and special files are not allowed in archives: "
                f"{member.filename}"
            )
        yield _ArchiveMember(
            name=member.filename,
            normalized=normalize_member_name(member.filename),
            is_directory=is_directory,
            size=member.file_size,
        )


def _tar_members(source: tarfile.TarFile) -> Iterator[_ArchiveMember]:
    for member in source:
        if not (member.isfile() or member.isdir()):
            raise RuntimeError(
                f"links and special files are not allowed in archives: {member.name}"
            )
        yield _ArchiveMember(
            name=member.name,
            normalized=normalize_member_name(member.name),
            is_directory=member.isdir(),
            size=member.size,
        )


@contextmanager
def open_tar(archive: Path) -> Iterator[tarfile.TarFile]:
    """Opens gzip or zstd tar archives without importing zstandard at bootstrap."""
    archive = Path(archive)
    if archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as source:
            yield source
        return
    if not archive.name.endswith(".tar.zst"):
        raise RuntimeError(f"unsupported tar archive format: {archive}")

    raw_stream = archive.open("rb")
    decoded_stream = None
    process = None
    try:
        try:
            decoded_stream = (
                zstandard_module().ZstdDecompressor().stream_reader(raw_stream)
            )
        except RuntimeError:
            zstd = shutil.which("zstd")
            if not zstd:
                raise
            process = subprocess.Popen(
                [zstd, "-q", "-d", "-c", str(archive)],
                stdout=subprocess.PIPE,
            )
            decoded_stream = process.stdout
        if decoded_stream is None:
            raise RuntimeError(f"cannot decode zstd archive: {archive}")
        with tarfile.open(fileobj=decoded_stream, mode="r|") as source:
            yield source
    finally:
        if decoded_stream is not None:
            decoded_stream.close()
        raw_stream.close()
        if process is not None and process.wait() != 0:
            raise RuntimeError(
                f"zstd failed to read {archive} (exit {process.returncode})"
            )


def extract_archive(archive: Path, destination: Path) -> None:
    """Extracts regular files/directories after traversal, collision, and link checks."""
    archive = Path(archive)
    destination = Path(destination)
    if destination.is_symlink():
        raise RuntimeError(f"extraction destination must not be a link: {destination}")

    if archive.suffix in (".zip", ".whl"):
        with zipfile.ZipFile(archive) as source:
            _validate_member_structure(_zip_members(source), destination=destination)
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                mode = (member.external_attr >> 16) & 0xFFFF
                target = safe_member_path(destination, member.filename)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with (
                    source.open(member) as input_stream,
                    target.open("wb") as output_stream,
                ):
                    shutil.copyfileobj(input_stream, output_stream)
                if mode:
                    target.chmod(mode & 0o777)
        return

    if not archive.name.endswith((".tar.gz", ".tar.zst")):
        raise RuntimeError(f"unsupported archive format: {archive}")
    with open_tar(archive) as source:
        _validate_member_structure(_tar_members(source), destination=destination)
    destination.mkdir(parents=True, exist_ok=True)
    with open_tar(archive) as source:
        for member in source:
            target = safe_member_path(destination, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            input_stream = source.extractfile(member)
            if input_stream is None:
                raise RuntimeError(f"cannot read archive member: {member.name}")
            with input_stream, target.open("wb") as output_stream:
                shutil.copyfileobj(input_stream, output_stream)
            target.chmod(member.mode & 0o777)


def extract_archive_atomic(
    archive: Path,
    destination: Path,
    *,
    allowed_root: Path,
    verify_root: Callable[[Path], object] | None = None,
) -> Path:
    """Extracts to a temporary sibling and promotes one verified root directory."""
    destination = Path(destination)
    allowed_root = Path(allowed_root).resolve()
    target = destination.resolve()
    if target == allowed_root or allowed_root not in target.parents:
        raise RuntimeError(
            f"refusing to remove unsafe extraction target: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-extract-", dir=destination.parent)
    )
    try:
        extract_archive(archive, temporary)
        entries = list(temporary.iterdir())
        if len(entries) != 1 or not entries[0].is_dir():
            raise RuntimeError(
                f"archive must contain exactly one root directory: {archive}"
            )
        root = entries[0]
        if verify_root is not None:
            verify_root(root)
        root_name = root.name
        if destination.is_symlink():
            destination.unlink()
        elif destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
        return destination / root_name
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def read_member(
    archive: Path, member_name: str, *, size_limit: int | None = None
) -> bytes:
    """Validates the complete archive, then reads exactly one regular member."""
    archive = Path(archive)
    wanted = normalize_member_name(member_name)
    if archive.suffix in (".zip", ".whl"):
        with zipfile.ZipFile(archive) as source:
            members = _validate_member_structure(_zip_members(source))
            matches = [
                member
                for member in members
                if member.normalized == wanted
                and not member.is_directory
                and (size_limit is None or member.size <= size_limit)
            ]
            if len(matches) == 1:
                return source.read(matches[0].name)
    elif archive.name.endswith((".tar.gz", ".tar.zst")):
        with open_tar(archive) as source:
            members = _validate_member_structure(_tar_members(source))
        matches = [
            member
            for member in members
            if member.normalized == wanted
            and not member.is_directory
            and (size_limit is None or member.size <= size_limit)
        ]
        if len(matches) == 1:
            with open_tar(archive) as source:
                for member in source:
                    if normalize_member_name(member.name) != wanted:
                        continue
                    stream = source.extractfile(member)
                    if stream is None:
                        break
                    with stream:
                        return stream.read()
    else:
        raise RuntimeError(f"unsupported archive format: {archive}")
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one regular member {member_name!r} in {archive}"
        )
    raise AssertionError("unreachable")


def _tree_entries(root: Path) -> Iterator[tuple[Path, str]]:
    root = Path(root)
    for path in sorted([root, *root.rglob("*")], key=lambda item: item.as_posix()):
        relative = "." if path == root else path.relative_to(root).as_posix()
        yield path, relative


def write_deterministic_tar(
    root: Path,
    output: tarfile.TarFile,
    *,
    arcname: str,
    fixed_mtime: int = FIXED_MTIME,
) -> None:
    """Writes a dereferenced, link-free tree to an already-open tar stream."""
    prefix = normalize_member_name(arcname)
    for path, relative in _tree_entries(Path(root)):
        member_name = prefix if relative == "." else posixpath.join(prefix, relative)
        info = tarfile.TarInfo(member_name)
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mtime = fixed_mtime
        if path.is_file():
            if path.is_symlink():
                resolved = path.resolve()
                root_resolved = Path(root).resolve()
                if root_resolved not in resolved.parents or not resolved.is_file():
                    raise RuntimeError(
                        f"archive symlink escapes source tree or is not a file: {path}"
                    )
            info.type = tarfile.REGTYPE
            info.size = path.stat().st_size
            info.mode = 0o755 if stat.S_IMODE(path.stat().st_mode) & 0o111 else 0o644
            with path.open("rb") as stream:
                output.addfile(info, stream)
        elif path.is_dir() and not path.is_symlink():
            info.type = tarfile.DIRTYPE
            info.mode = 0o755
            output.addfile(info)
        else:
            raise RuntimeError(f"unsupported archive filesystem entry: {path}")


def create_deterministic_tar_gz(
    root: Path,
    archive: Path,
    *,
    arcname: str,
    compresslevel: int,
) -> None:
    with (
        Path(archive).open("wb") as raw_stream,
        gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_stream,
            compresslevel=compresslevel,
            mtime=FIXED_MTIME,
        ) as compressed,
        tarfile.open(
            fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT
        ) as output,
    ):
        write_deterministic_tar(root, output, arcname=arcname)


def create_deterministic_tar_zst(
    root: Path,
    archive: Path,
    *,
    arcname: str,
    level: int,
    compression_parameters: dict[str, object] | None = None,
) -> None:
    zstandard = zstandard_module()
    if compression_parameters:
        parameters = zstandard.ZstdCompressionParameters.from_level(
            level, **compression_parameters
        )
        compressor = zstandard.ZstdCompressor(compression_params=parameters)
    else:
        compressor = zstandard.ZstdCompressor(level=level, threads=0)
    with (
        Path(archive).open("wb") as raw_stream,
        compressor.stream_writer(raw_stream, closefd=False) as compressed,
        tarfile.open(
            fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT
        ) as output,
    ):
        write_deterministic_tar(root, output, arcname=arcname)


def create_deterministic_tar_zstd_cli(
    root: Path,
    archive: Path,
    *,
    arcname: str,
    level: int,
) -> None:
    zstd = shutil.which("zstd")
    if not zstd:
        raise RuntimeError("zstd is required to create this archive")
    process = subprocess.Popen(
        [zstd, f"-{level}", "-T0", "-q", "-f", "-o", str(archive)],
        stdin=subprocess.PIPE,
    )
    if process.stdin is None:
        raise RuntimeError("zstd did not provide an input stream")
    with (
        process.stdin,
        tarfile.open(
            fileobj=process.stdin, mode="w|", format=tarfile.PAX_FORMAT
        ) as output,
    ):
        write_deterministic_tar(root, output, arcname=arcname)
    if process.wait() != 0:
        raise RuntimeError(f"zstd failed (exit {process.returncode})")


def create_deterministic_zip(
    root: Path,
    archive: Path,
    *,
    arcname: str,
    compresslevel: int,
    executable: Callable[[Path], bool] | None = None,
) -> None:
    prefix = normalize_member_name(arcname)
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=compresslevel
    ) as output:
        for path, relative in _tree_entries(Path(root)):
            if relative == "." or path.is_dir():
                continue
            if not path.is_file():
                raise RuntimeError(f"unsupported archive filesystem entry: {path}")
            if path.is_symlink():
                resolved = path.resolve()
                root_resolved = Path(root).resolve()
                if root_resolved not in resolved.parents or not resolved.is_file():
                    raise RuntimeError(
                        f"archive symlink escapes source tree or is not a file: {path}"
                    )
            member_name = posixpath.join(prefix, relative)
            info = zipfile.ZipInfo(member_name, date_time=ZIP_DATE_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            is_executable = (
                executable(path)
                if executable is not None
                else bool(stat.S_IMODE(path.stat().st_mode) & 0o111)
            )
            mode = 0o755 if is_executable else 0o644
            info.external_attr = (stat.S_IFREG | mode) << 16
            output.writestr(info, path.read_bytes(), compresslevel=compresslevel)
