# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Dependency policy shared by the FastFlowLM developer commands and tests."""

from pathlib import Path


VCPKG_COMMIT = "f87344cac03158cbf1467264565f1fd36b382a24"
VCPKG_TRIPLET = "x64-windows-rel"
MIN_RUSTC = (1, 80, 0)

REQUIRED_BOOST_HEADERS = {
    "Boost.ProgramOptions": Path("boost/program_options.hpp"),
    "Boost.Beast": Path("boost/beast/core.hpp"),
}
REQUIRED_VCPKG_LIBRARIES = {
    "curl": (("libcurl.lib", "curl.lib"),),
    "FFmpeg": (
        ("avformat.lib",),
        ("avcodec.lib",),
        ("avutil.lib",),
        ("swscale.lib",),
        ("swresample.lib",),
    ),
    "FFTW": (("fftw3.lib",), ("fftw3f.lib",), ("fftw3l.lib",)),
}


def resolve_windows_roots(repo_root: Path, environ) -> dict[str, Path | None]:
    """Resolves one coherent vcpkg tree, with explicit configuration first."""
    explicit_vcpkg = environ.get("FASTFLOWLM_VCPKG_INSTALLED_DIR") or environ.get(
        "VCPKG_INSTALLED_DIR"
    )
    if explicit_vcpkg:
        vcpkg = Path(explicit_vcpkg)
    elif environ.get("VCPKG_ROOT"):
        installed = Path(environ["VCPKG_ROOT"]) / "installed"
        candidates = [installed / VCPKG_TRIPLET, installed / "x64-windows"]
        vcpkg = next(
            (path for path in candidates if path.is_dir()), candidates[0]
        )
    else:
        candidates = [
            repo_root / ".cache" / "vcpkg-installed" / VCPKG_TRIPLET,
            Path("C:/dev/vcpkg/installed/x64-windows"),
            Path("C:/vcpkg/installed/x64-windows"),
        ]
        vcpkg = next((path for path in candidates if path.is_dir()), candidates[0])

    explicit_boost = (
        environ.get("FASTFLOWLM_BOOST_ROOT")
        or environ.get("BOOST_ROOT")
        or environ.get("BOOSTROOT")
    )
    vcpkg_lib = vcpkg / "lib" if vcpkg else None
    if explicit_boost:
        root = Path(explicit_boost)
        boost_providers = [(root, root / "stage" / "lib")]
    else:
        boost_providers = [(vcpkg / "include", vcpkg_lib)]
        for root in (
            Path("C:/dev/boost_1_88_0"),
            Path("C:/Program Files/boost/boost_1_88_0"),
        ):
            boost_providers.append((root, root / "stage" / "lib"))

    boost = None
    boost_lib = None
    for include_dir, lib_dir in boost_providers:
        if (
            (include_dir / "boost").is_dir()
            and lib_dir
            and lib_dir.is_dir()
            and any(lib_dir.glob("*boost_program_options*.lib"))
        ):
            boost, boost_lib = include_dir, lib_dir
            break
    if boost is None:
        boost = next(
            (include_dir for include_dir, _ in boost_providers if (include_dir / "boost").is_dir()),
            boost_providers[0][0] if boost_providers else None,
        )
    return {
        "boost_root": boost,
        "boost_lib": boost_lib,
        "vcpkg_root": vcpkg,
        "vcpkg_bin": vcpkg / "bin" if vcpkg else None,
    }


def windows_dependency_errors(
    roots, *, cargo_available: bool, rustc_version: tuple[int, ...] | None
) -> list[str]:
    """Checks capabilities provided by the selected dependency installation."""
    missing = []
    if not cargo_available:
        missing.append("Rust/Cargo (install with rustup; rustc >= 1.80 required)")
    if rustc_version is None or rustc_version < MIN_RUSTC:
        got = ".".join(str(n) for n in rustc_version) if rustc_version else "missing/unknown"
        need = ".".join(str(n) for n in MIN_RUSTC)
        missing.append(f"rustc >= {need} (found {got})")

    vcpkg_root = roots["vcpkg_root"]
    if (
        not vcpkg_root
        or not (vcpkg_root / "include").is_dir()
        or not (vcpkg_root / "lib").is_dir()
    ):
        missing.append(
            "FASTFLOWLM_VCPKG_INSTALLED_DIR / managed vcpkg tree with include+lib"
        )

    boost_root = roots["boost_root"]
    for label, relative_path in REQUIRED_BOOST_HEADERS.items():
        if not boost_root or not (boost_root / relative_path).is_file():
            missing.append(
                f"{label} headers (run `python dev.py setup --profile fastflowlm` "
                "or set FASTFLOWLM_BOOST_ROOT)"
            )
    if not roots["boost_lib"]:
        missing.append("Boost program_options library")

    vcpkg_lib = vcpkg_root / "lib" if vcpkg_root else None
    if vcpkg_lib and vcpkg_lib.is_dir():
        for label, alternatives in REQUIRED_VCPKG_LIBRARIES.items():
            absent = [
                "/".join(names)
                for names in alternatives
                if not any((vcpkg_lib / name).is_file() for name in names)
            ]
            if absent:
                missing.append(f"{label} libraries ({', '.join(absent)})")
    elif vcpkg_root:
        missing.append(f"vcpkg library directory ({vcpkg_lib})")
    return missing
