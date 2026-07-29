"""Developer-command implementation package.

Ownership is intentionally split by concern:

* ``paths`` owns repository layout and host path conventions.
* ``environment`` owns tool discovery and subprocess environments.
* ``profiles`` owns declarative dependency-closure policy.
* ``patches`` owns source pin, patch, and prefetch state.
* ``doctor`` reports readiness without mutating the repository.
* ``setup`` resolves and executes dependency setup plans.

The root ``dev.py`` file owns only public CLI parsing and command dispatch.
"""
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Shared implementation helpers for the developer command router."""
