# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Repository-local build, packaging, and product tooling.

The ``dev`` subpackage owns shared developer-command infrastructure such as
repository setup, build orchestration, and test execution.  Sibling modules
own product-specific artifact, wheel, and package operations.  Callers import
those owner modules explicitly; this package intentionally provides no eager
re-exports.
"""
