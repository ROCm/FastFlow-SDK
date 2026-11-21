# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Reference pages of the documentation site, rendered from their sources.

Each page is generated from the file that owns its facts: the CLI's help text
and check IDs, ``tools/fastflow/checks.toml``, ``manifests/versions.toml``,
FastFlowLM's model list, IRON's operator sources, and the GitHub releases.
``pages.generate`` renders all of them, and ``pages.write`` writes them, with
the example scripts the pages include, into the gitignored paths of ``docs/``
before ``python -m build_tools.docs build`` runs Zensical.
"""
