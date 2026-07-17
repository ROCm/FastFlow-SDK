# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Release-only dynamic dependencies for the relocatable Linux runtime bundle.
set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE dynamic)
set(VCPKG_BUILD_TYPE release)

# Custom non-Windows triplets must declare their target system. This matches
# vcpkg's stock x64-linux triplet and keeps this overlay native to Linux.
set(VCPKG_CMAKE_SYSTEM_NAME Linux)
