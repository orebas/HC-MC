# Find clang-tidy executable
find_program(CLANG_TIDY_BIN NAMES clang-tidy)

if(CLANG_TIDY_BIN)
    message(STATUS "Found: clang-tidy")
    
    # Don't set CMAKE_CXX_CLANG_TIDY - we don't want it running on every build
    
    # Get all source files
    file(GLOB_RECURSE ALL_SOURCE_FILES 
        ${PROJECT_SOURCE_DIR}/project/*.cpp
        ${PROJECT_SOURCE_DIR}/project/*.h
        ${PROJECT_SOURCE_DIR}/project/*.hpp
    )

    # Create a custom target for running clang-tidy
    add_custom_target(
        tidy
        COMMAND ${CLANG_TIDY_BIN}
            -p=${CMAKE_BINARY_DIR}
            ${ALL_SOURCE_FILES}
            > ${CMAKE_BINARY_DIR}/clang-tidy-report.txt
        COMMENT "Running clang-tidy and generating report..."
        VERBATIM
    )

    # Add a target to show the report
    add_custom_target(
        tidy-report
        COMMAND ${CMAKE_COMMAND} -E cat ${CMAKE_BINARY_DIR}/clang-tidy-report.txt
        DEPENDS tidy
        COMMENT "Showing clang-tidy report..."
    )
else()
    message(STATUS "clang-tidy not found. Tidy targets will not be available.")
endif()