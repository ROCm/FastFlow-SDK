# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Installed-product qualification for FastFlow SDK releases."""

from importlib.metadata import PackageNotFoundError, version


def package_version() -> str:
    try:
        return version("fastflow-sdk-qualification")
    except PackageNotFoundError:
        return "0.0.0+source"


__version__ = package_version()
