# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest

from build_tools.dev.fastflowlm_build import _host_xrt_pkgconfig_dir

pytestmark = pytest.mark.presubmit


def test_host_xrt_pkgconfig_dir_uses_real_host_pc(tmp_path):
    host = tmp_path / "xrt"
    pc = host / "share" / "pkgconfig" / "xrt.pc"
    pc.parent.mkdir(parents=True)
    pc.write_text("Name: XRT\nVersion: 2.23.0\n", encoding="utf-8")
    assert _host_xrt_pkgconfig_dir(host) == pc.parent


def test_host_xrt_pkgconfig_dir_does_not_invent_a_stub(tmp_path):
    host = tmp_path / "xrt"
    host.mkdir()
    assert _host_xrt_pkgconfig_dir(host) is None
    assert not list(host.rglob("xrt.pc"))
