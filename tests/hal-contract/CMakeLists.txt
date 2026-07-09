# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Builds hal_pdi_loader against the amdxdna HAL + iree base libs that `dev.py build hrx`
# produced. Driven by tests/e2e/run_e2e.py; not part of the main build.
#
# This exists so the e2e does not hand-roll a link line: library naming (.a vs .lib),
# the archive-group flags needed for the circular iree base deps, and the system
# libraries all differ per platform, and CMake already knows how to spell them.
cmake_minimum_required(VERSION 3.20)
project(hal_contract C)

set(HRX_SRC "" CACHE PATH "projects/hrx source root")
set(HRX_BUILD "" CACHE PATH "projects/hrx CMake build dir")
if(NOT HRX_SRC OR NOT HRX_BUILD)
  message(FATAL_ERROR "set -DHRX_SRC=<projects/hrx> -DHRX_BUILD=<.../build/cmake>")
endif()

set(_hal_dir "${HRX_BUILD}/runtime/src/iree/hal/drivers/amdxdna")
find_library(HAL_LIB
  NAMES iree_hal_drivers_amdxdna_amdxdna
  PATHS "${_hal_dir}" NO_DEFAULT_PATH REQUIRED)

# The iree base targets are many small archives with circular references between them.
# Glob rather than name them: the set changes with IREE's target graph, and the e2e only
# cares that the HAL's xclbin parser links.
#
# GLOB_RECURSE, not GLOB: the archives are nested (base/, base/internal/, base/threading/,
# base/tooling/). A flat glob finds only libiree_base_base and still links today, because
# the xclbin parser happens to need nothing else -- i.e. it fails silently, weakening the
# check instead of breaking it. CONFIGURE_DEPENDS so a cached configure re-globs when the
# hrx build grows a target.
file(GLOB_RECURSE BASE_LIBS CONFIGURE_DEPENDS
  "${HRX_BUILD}/runtime/src/iree/base/*${CMAKE_STATIC_LIBRARY_SUFFIX}")
list(LENGTH BASE_LIBS _n_base)
if(_n_base LESS 2)
  message(FATAL_ERROR
    "found only ${_n_base} iree base lib(s) under ${HRX_BUILD}/runtime/src/iree/base "
    "-- expected the nested internal/threading/tooling archives too; is hrx built?")
endif()
message(STATUS "hal-contract: linking ${_n_base} iree base archives")

add_executable(hal_pdi_loader hal_pdi_loader.c)
target_include_directories(hal_pdi_loader PRIVATE
  "${HRX_SRC}/runtime/src" "${HRX_BUILD}/runtime/src")

if(MSVC)
  # MSVC (and clang-cl) resolve inter-archive references without a group; link.exe also
  # pulls in the system libraries it needs. MSVC is true for clang-cl too, which is what
  # the Windows CI uses.
  target_link_libraries(hal_pdi_loader PRIVATE ${HAL_LIB} ${BASE_LIBS})
else()
  find_package(Threads REQUIRED)
  target_link_libraries(hal_pdi_loader PRIVATE
    -Wl,--start-group ${HAL_LIB} ${BASE_LIBS} -Wl,--end-group
    Threads::Threads ${CMAKE_DL_LIBS} m)
endif()
