# cmake/StaticAnalysis.cmake
# Configures formatting and static analysis targets using clang-format and clang-tidy.

find_program(CLANG_FORMAT_EXE NAMES clang-format clang-format-21 clang-format-20 clang-format-19)
find_program(CLANG_TIDY_EXE NAMES clang-tidy clang-tidy-21 clang-tidy-20 clang-tidy-19)

file(GLOB_RECURSE PROJECT_CXX_SOURCES
  CONFIGURE_DEPENDS
  "${CMAKE_SOURCE_DIR}/src/*.cpp"
  "${CMAKE_SOURCE_DIR}/src/*.hpp"
  "${CMAKE_SOURCE_DIR}/src/*.h"
  "${CMAKE_SOURCE_DIR}/models/*.cpp"
  "${CMAKE_SOURCE_DIR}/models/*.hpp"
  "${CMAKE_SOURCE_DIR}/models/*.h"
  "${CMAKE_SOURCE_DIR}/models/*.hip"
)

if (CLANG_FORMAT_EXE)
  # Format check target (dry-run mode, exits with non-zero on formatting differences)
  add_custom_target(format-check
    COMMAND ${CLANG_FORMAT_EXE} --dry-run --Werror ${PROJECT_CXX_SOURCES}
    COMMENT "Running clang-format dry-run check on project sources"
    VERBATIM
  )

  # In-place format target
  add_custom_target(format
    COMMAND ${CLANG_FORMAT_EXE} -i ${PROJECT_CXX_SOURCES}
    COMMENT "Formatting project sources in-place with clang-format"
    VERBATIM
  )

  if (BUILD_TESTING)
    add_test(
      NAME check-formatting
      COMMAND ${CLANG_FORMAT_EXE} --dry-run --Werror ${PROJECT_CXX_SOURCES}
    )
    set_tests_properties(check-formatting PROPERTIES LABELS "cpu;static;formatting")
  endif()
endif()

if (CLANG_TIDY_EXE)
  # Static analysis target (runs clang-tidy with compile_commands.json)
  add_custom_target(static-analysis
    COMMAND ${CLANG_TIDY_EXE} -p "${CMAKE_BINARY_DIR}" ${PROJECT_CXX_SOURCES}
    COMMENT "Running clang-tidy static analysis on project sources"
    VERBATIM
  )

  if (BUILD_TESTING AND CMAKE_EXPORT_COMPILE_COMMANDS)
    add_test(
      NAME check-static-analysis
      COMMAND ${CLANG_TIDY_EXE} -p "${CMAKE_BINARY_DIR}" ${PROJECT_CXX_SOURCES}
    )
    set_tests_properties(check-static-analysis PROPERTIES LABELS "cpu;static;clang-tidy")
  endif()
endif()
