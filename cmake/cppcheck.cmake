# Find clang-tidy executable
find_program(CPPCHECK_BIN NAMES cppcheck)

if(CPPCHECK_BIN)
    message(STATUS "Found: cppcheck")
    
    # Existing configuration for build-time checks
    list(
        APPEND CMAKE_CXX_CPPCHECK 
            "${CPPCHECK_BIN}"
            "--enable=all"
            "--enable=warning,performance,portability,information,missingInclude"
            "--inconclusive"
            "--check-config"
            "--force" 
            "--inline-suppr"
            "--suppressions-list=${CMAKE_SOURCE_DIR}/cppcheck_suppressions.txt"
            "--xml"
            "--output-file=${CMAKE_BINARY_DIR}/cppcheck.xml"
    )

    # Add custom target for explicit cppcheck runs
    add_custom_target(
        cppcheck-analysis
        COMMAND ${CPPCHECK_BIN}
            --enable=all
            --enable=missingInclude
            --std=c++17
            --error-exitcode=1
            --suppress=missingInclude
            -i${CMAKE_SOURCE_DIR}/build
            ${PROJECT_SOURCE_DIR}/project
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Running cppcheck analysis"
    )

else()
    message(STATUS "cppcheck not found. Analysis targets will not be available.")
endif()  