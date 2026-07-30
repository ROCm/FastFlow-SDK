# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Linux amdxdna sysfs and accelerator-devnode diagnostics."""

import errno
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

ACCEL_DIR = Path("/dev/accel")
SYS_ACCEL_DIR = Path("/sys/class/accel")
AMDXDNA_MODULE_DIR = Path("/sys/module/amdxdna")
KNOWN_AMDXDNA_SVA_REGRESSION_SRCVERSIONS = {
    "2DBDA75956CAFA9D029EA89",
}


def read_sysfs_text(path: Path):
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    return text if text else None


def _print_probe_file(label: str, path: Path, indent="  "):
    value = read_sysfs_text(path)
    if value is None:
        print(f"{indent}[info] {label}: unavailable ({path})")
    else:
        print(f"{indent}[info] {label}: {value}")


def _modinfo_field(field: str):
    if not shutil.which("modinfo"):
        return None
    result = subprocess.run(
        ["modinfo", "-F", field, "amdxdna"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value if value else None


def sys_accel_nodes():
    return sorted(SYS_ACCEL_DIR.glob("accel*")) if SYS_ACCEL_DIR.is_dir() else []


def known_amdxdna_sva_regression(
    kernel_release: str, srcversion: str | None
) -> bool:
    return (
        kernel_release.startswith("6.17.0-22")
        and srcversion in KNOWN_AMDXDNA_SVA_REGRESSION_SRCVERSIONS
    )


def _accel_open_failure_notes(err: OSError, sys_node: Path):
    notes = []
    if err.errno in (errno.EACCES, errno.EPERM):
        notes.append("runner host must grant read/write access before this job runs")
        return notes
    if err.errno == errno.EOPNOTSUPP:
        notes.append(
            "amdxdna refused userspace open with EOPNOTSUPP; check kernel log for "
            "'amdxdna_drm_open: SVA bind device failed, ret -95'"
        )
        group_type = read_sysfs_text(
            sys_node / "device" / "iommu_group" / "type"
        )
        if group_type:
            notes.append(f"IOMMU group type: {group_type}")
        try:
            kernel_release = os.uname().release
        except AttributeError:
            kernel_release = ""
        srcversion = read_sysfs_text(AMDXDNA_MODULE_DIR / "srcversion")
        if known_amdxdna_sva_regression(kernel_release, srcversion):
            notes.append(
                "known-bad Ubuntu 6.17.0-22 amdxdna/SVA stack; use 6.17.0-20 "
                "or a fixed newer kernel for NPU tests"
            )
        else:
            notes.append(
                "this is an amdxdna/IOMMU SVA setup failure before HRX dispatch starts"
            )
    return notes


def probe_accel_open(target: Path, sys_node: Path) -> bool:
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(target, flags)
    except OSError as error:
        print(f"  [MISS] {target} exists but open(O_RDWR) failed: {error}")
        for note in _accel_open_failure_notes(error, sys_node):
            print(f"         {note}")
        return False
    os.close(fd)
    return True


def probe_npu_devnodes():
    rc = 0
    usable = []
    sys_nodes = sys_accel_nodes()
    if not sys_nodes:
        print(f"  [MISS] no {SYS_ACCEL_DIR}/accel* sysfs node")
        return 1, usable
    for sys_node in sys_nodes:
        dev_text = read_sysfs_text(sys_node / "dev")
        target = ACCEL_DIR / sys_node.name
        if not dev_text or ":" not in dev_text:
            print(f"  [MISS] {sys_node}/dev unavailable; cannot validate {target}")
            rc = 1
            continue
        major_s, minor_s = dev_text.split(":", 1)
        try:
            major, minor = int(major_s), int(minor_s)
        except ValueError:
            print(f"  [MISS] {sys_node}/dev has unexpected value {dev_text!r}")
            rc = 1
            continue
        if target.exists():
            try:
                device_stat = target.stat()
            except OSError as error:
                print(f"  [MISS] cannot stat {target}: {error}")
                rc = 1
                continue
            if not stat.S_ISCHR(device_stat.st_mode):
                print(f"  [MISS] {target} exists but is not a character device")
                rc = 1
                continue
            if (
                os.major(device_stat.st_rdev) != major
                or os.minor(device_stat.st_rdev) != minor
            ):
                have = (
                    f"{os.major(device_stat.st_rdev)}:"
                    f"{os.minor(device_stat.st_rdev)}"
                )
                print(f"  [MISS] {target} has dev {have}, expected {dev_text}")
                rc = 1
                continue
            if not (
                os.access(target, os.R_OK) and os.access(target, os.W_OK)
            ):
                print(f"  [MISS] {target} exists but is not read+write for this user")
                print(
                    "         runner host must grant read/write access before this job runs"
                )
                rc = 1
                continue
            if not probe_accel_open(target, sys_node):
                rc = 1
                continue
            print(f"  [ok ] {target} ({dev_text})")
            usable.append(target)
            continue
        print(f"  [MISS] missing {target} for {sys_node} ({dev_text})")
        print(
            "         runner host must provide this devtmpfs/udev node or pass it through"
        )
        print(f"         root repair example: install -d -m 755 {ACCEL_DIR}")
        print(f"                              mknod -m 666 {target} c {major} {minor}")
        rc = 1
    return rc, usable


def print_npu_driver_info():
    try:
        kernel = " ".join(os.uname())
    except AttributeError:
        kernel = sys.platform
    print("== amdxdna driver / firmware ==")
    print(f"  [info] kernel: {kernel}")
    if AMDXDNA_MODULE_DIR.is_dir():
        _print_probe_file(
            "amdxdna module version", AMDXDNA_MODULE_DIR / "version"
        )
        _print_probe_file(
            "amdxdna module srcversion", AMDXDNA_MODULE_DIR / "srcversion"
        )
        _print_probe_file(
            "amdxdna module initstate", AMDXDNA_MODULE_DIR / "initstate"
        )
    else:
        print(
            f"  [MISS] no {AMDXDNA_MODULE_DIR} - is the amdxdna driver loaded?"
        )
    for field in ["filename", "version", "srcversion"]:
        value = _modinfo_field(field)
        if value:
            print(f"  [info] amdxdna modinfo {field}: {value}")
    firmware_files = _modinfo_field("firmware")
    if firmware_files:
        print("  [info] amdxdna modinfo firmware files:")
        for line in firmware_files.splitlines():
            print(f"    {line}")
    sys_nodes = sys_accel_nodes()
    if not sys_nodes:
        print(f"  [MISS] no {SYS_ACCEL_DIR}/accel* sysfs node")
        return
    for node in sys_nodes:
        print(f"  [info] {node}:")
        try:
            print(f"    device path: {(node / 'device').resolve(strict=False)}")
        except OSError:
            pass
        _print_probe_file(
            "firmware version", node / "device" / "fw_version", indent="    "
        )
        _print_probe_file("vbnv", node / "device" / "vbnv", indent="    ")
        _print_probe_file("vendor", node / "device" / "vendor", indent="    ")
        _print_probe_file("device", node / "device" / "device", indent="    ")
        _print_probe_file(
            "revision", node / "device" / "revision", indent="    "
        )
