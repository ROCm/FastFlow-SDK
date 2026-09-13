# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FastFlowLM transcript classifiers used by qualification."""

from __future__ import annotations

import pytest

from rocm_npu_qualification.fastflowlm import has_runtime_error

pytestmark = [pytest.mark.fastflowlm, pytest.mark.presubmit]


def test_has_runtime_error_classifies_windows_context_pool_remap():
    assert has_runtime_error(
        "D3DKMTCreateContextVirtual failed: 0xc01e0009 STATUS_GRAPHICS_DRIVER_MISMATCH"
    )
    assert has_runtime_error("CreateHwContext NTSTATUS 0xC01E0009")
    assert not has_runtime_error("Start generating...\nThe scarecrow won an award.")
