# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Developer-command implementation package.

Ownership is intentionally split by concern:

* ``paths`` owns repository layout and host path conventions.
* ``environment`` owns tool discovery and subprocess environments.
* ``profiles`` owns declarative dependency-closure policy.
* ``patches`` owns source pin, patch, and prefetch state.
* ``doctor`` reports readiness without mutating the repository.
* ``setup`` resolves and executes dependency setup plans.
* ``build`` owns build output paths and CMake build orchestration.
* ``sources`` owns source checkout and submodule operations.
* ``iron`` owns shared IRON targets and toolchain policy.
* ``testing`` owns host and on-device test command orchestration.

The root ``dev.py`` file owns only public CLI parsing and command dispatch.
Artifact commands are delegated to product-specific owner modules alongside
this package: ``build_tools.fastflowlm_package``,
``build_tools.compiler_product``, ``build_tools.compiler_wheels``,
``build_tools.npu_prereqs``, and ``build_tools.npu_test_package``.
"""
