# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Developer-command implementation package.

Ownership is intentionally split by concern:

* ``profiles`` owns declarative dependency-closure policy.
* ``patches`` owns source pin, patch, and prefetch state.
* ``doctor`` reports readiness without mutating the repository.
* ``clang_format`` owns the first-party C/C++ format command.
* ``setup`` resolves and executes dependency setup plans.
* ``cli`` owns parser construction, command registration, and dispatch.
* ``build`` coordinates the build graph; ``compiler_build``, ``runtime_build``,
  and ``fastflowlm_build`` own their build domains.
* ``sources`` owns source checkout and submodule operations.
* ``host_testing`` and ``device_testing`` own their respective test commands.

The root ``dev.py`` file is only an executable shim for ``dev.cli.main``.
Shared repository paths and process/tool environments are foundations owned by
``build_tools.paths`` and ``build_tools.environment``. Product-specific checks
remain with their product owners.
Shared IRON targets and toolchain policy live in ``build_tools.iron``.
Artifact commands are delegated to product-specific owner modules alongside
this package: ``build_tools.fastflowlm_package``,
``build_tools.compiler_product``, ``build_tools.compiler_wheels``,
``build_tools.npu_prereqs``, and ``build_tools.npu_test_package``.
"""
