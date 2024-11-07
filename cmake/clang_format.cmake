# MIT License 
# Copyright (c) 2018-Today Michele Adduci <adduci@tutanota.com>
#
# Clang-Format instructions

find_program(CLANG_FORMAT_BIN NAMES clang-format)

if(CLANG_FORMAT_BIN)
  message(STATUS "Found: clang-format")
  
  file(GLOB_RECURSE ALL_SOURCE_FILES 
    ${PROJECT_SOURCE_DIR}/project/*.cpp
    ${PROJECT_SOURCE_DIR}/project/*.h
    ${PROJECT_SOURCE_DIR}/project/*.hpp
  )

  add_custom_target(
    format
    COMMAND ${CLANG_FORMAT_BIN}
    -i
    --style=file
    ${ALL_SOURCE_FILES}
  )

  add_custom_target(
    format-check
    COMMAND ${CLANG_FORMAT_BIN}
    --style=file
    --dry-run
    --Werror
    ${ALL_SOURCE_FILE}
  )

  add_custom_target(
    format-all
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target format
    COMMENT "Running clang-format on all source files"
  )
else()
  message(STATUS "clang-format not found. Formatting targets will not be available.")
endif()
