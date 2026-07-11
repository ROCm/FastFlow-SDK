# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Release-only variant of vcpkg's stock x64-windows triplet, used by the Windows CI to
# install OpenSSL (bootgen's find_package(OpenSSL) -- the only system library this stack
# needs on Windows).
#
# Same dynamic linkage as x64-windows; the ONLY difference is VCPKG_BUILD_TYPE=release,
# which makes vcpkg build OpenSSL once instead of debug+release and ~halves a cold build.
#
# This lives in the repo rather than being printf'd into place by the workflow so that
# actions/cache can key the vcpkg binary-package store on hashFiles() of it: change the
# triplet, get a new cache scope.
set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE dynamic)
set(VCPKG_BUILD_TYPE release)
