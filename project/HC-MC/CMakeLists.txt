cmake_minimum_required(VERSION 3.15)
project(HC-MC)

#####################################
# Define Target Information
#####################################
set(TARGET_NAME HC-MC)
set(TARGET_INCLUDE_FOLDER ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Find dependencies
find_package(PkgConfig REQUIRED)
find_package(doctest CONFIG REQUIRED)

# Locate CppAD using pkg-config
pkg_check_modules(CPPAD REQUIRED cppad)

# Header-only library target
add_library(${TARGET_NAME} INTERFACE)
target_include_directories(${TARGET_NAME} 
    INTERFACE 
        ${TARGET_INCLUDE_FOLDER}
        ${CPPAD_INCLUDE_DIRS}
)

# Link against required dependencies
target_link_libraries(${TARGET_NAME} 
    INTERFACE 
        Eigen3::Eigen
        ${CPPAD_LIBRARIES}
)

#####################################
# Add Main Executable
#####################################
add_executable(${TARGET_NAME}_main
    main.cpp
)

# Enable warnings for the executable
target_enable_warnings(${TARGET_NAME}_main)

# Link main executable with the library and required dependencies
target_link_libraries(${TARGET_NAME}_main
    PRIVATE
        ${TARGET_NAME}
        Eigen3::Eigen
        ${CPPAD_LIBRARIES}
)

#####################################
# Tests
#####################################
if(BUILD_TESTING)
    # Create test executable
    add_executable(${TARGET_NAME}_tests 
        tests/pt_tests.cpp
    )
    
    # Enable warnings for the test executable
    target_enable_warnings(${TARGET_NAME}_tests)
    
    # Link against our library and doctest
    target_link_libraries(${TARGET_NAME}_tests 
        PRIVATE 
            ${TARGET_NAME}
            doctest::doctest
    )
    
    # Add to CTest
    add_test(
        NAME ${TARGET_NAME}_tests 
        COMMAND ${TARGET_NAME}_tests
    )
endif()