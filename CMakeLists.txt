cmake_minimum_required(VERSION 3.15.0 FATAL_ERROR)
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/external/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")


project(moderncpp_project 
    VERSION 1.1.2
    LANGUAGES CXX
    DESCRIPTION "HC-MC is a testbed for homotopy continuation methods."
)

# Set required C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

# Configure the visibility of symbols in targets
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

#####################################
# Define Options
#####################################
option(BUILD_TESTING "Build tests" FALSE)
option(BUILD_SHARED_LIBS "Build shared libraries" FALSE)
option(BUILD_WITH_MT "Build libraries as MultiThreaded DLL (Windows Only)" FALSE)

#####################################
# Find Dependencies - vcpkg handles the heavy lifting
#####################################
find_package(Eigen3 CONFIG REQUIRED)

# Use pkg-config to locate cppad
find_package(PkgConfig REQUIRED)
pkg_check_modules(CPPAD REQUIRED cppad)

# Add the cppad include directories and libraries
include_directories(${CPPAD_INCLUDE_DIRS})
link_directories(${CPPAD_LIBRARY_DIRS})

# Add testing dependencies if enabled
if(BUILD_TESTING)
    find_package(doctest CONFIG REQUIRED)
    enable_testing()
endif()

#####################################
# Define CMake Module Imports
#####################################
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(GenerateExportHeader)
include(${CMAKE_SOURCE_DIR}/cmake/compiler_options.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/cpack.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/cppcheck.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/clang_format.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/clang_tidy.cmake)

#####################################
# Define Targets
#####################################
add_subdirectory(project/HC-MC)

#####################################
# Define Install Targets
#####################################
install(TARGETS
    HC-MC
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)

install(FILES  
    ${CMAKE_SOURCE_DIR}/LICENSE
    DESTINATION .
)