# Plain CMake build against spack-loaded LArSoft packages (no cetmodules,
# no sbndcode/icaruscode). By default only the TITUS product reader
# (EventDisplay_Reader) is built; TITUS_LEGACY adds the old
# gallery-framework libraries (core/, 3DViewer), which need larcorealg and
# lardataalg. Set up the environment with env.sh first.

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(gallery_framework LANGUAGES CXX VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)  # the spack stack is built with cxxstd=17
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

option(TITUS_LEGACY "Also build core/ (Base, Analysis, LArUtil) and the 3D viewer" OFF)

include(GNUInstallDirs)
# Libraries, dictionaries (.pcm) and rootmaps side by side in build/lib, so
# the build can be used directly with build/lib on LD_LIBRARY_PATH
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
list(APPEND CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_FULL_LIBDIR})

find_package(ROOT CONFIG REQUIRED COMPONENTS Core RIO Tree)
find_package(gallery REQUIRED)
find_package(canvas REQUIRED)
find_package(cetlib_except REQUIRED)
find_package(larcoreobj REQUIRED)
find_package(lardataobj REQUIRED)
find_package(nusimdata REQUIRED)
find_package(sbnobj QUIET)

if (TITUS_LEGACY)
  find_package(larcorealg REQUIRED)
  find_package(lardataalg REQUIRED)
  find_package(fhiclcpp REQUIRED)
  find_package(messagefacility REQUIRED)
  find_package(CLHEP REQUIRED)
endif()

# ROOT dictionary for a library: lib<target>_rdict.pcm and lib<target>.rootmap
# next to it in build/lib, installed with it. rootcling finds the headers
# through the target's include directories, so the calling directory is added
# (cetmodules used to do that for every directory).
function(titus_root_dictionary target)
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  ROOT_GENERATE_DICTIONARY(${target}_Dict ${ARGN} MODULE ${target} LINKDEF LinkDef.h)
  install(TARGETS ${target} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(FILES
    ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/lib${target}_rdict.pcm
    ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/lib${target}.rootmap
    DESTINATION ${CMAKE_INSTALL_LIBDIR})
endfunction()

if (TITUS_LEGACY)
  add_subdirectory(core)
endif()
add_subdirectory(UserDev)

message(STATUS "TITUS_LEGACY (core/, 3DViewer): ${TITUS_LEGACY}")
