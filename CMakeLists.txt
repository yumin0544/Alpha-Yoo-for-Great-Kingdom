cmake_minimum_required(VERSION 3.20)

project(BoardGameEngine VERSION 0.1.0 LANGUAGES CXX)

add_library(engine
    engine/src/Board.cpp
    engine/src/Move.cpp
    engine/src/State.cpp
)
target_compile_features(engine PUBLIC cxx_std_20)
target_include_directories(engine PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/engine/include>"
)
set_target_properties(engine PROPERTIES CXX_EXTENSIONS OFF)

function(enable_project_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive- /utf-8)
    else()
        target_compile_options(${target} PRIVATE
            -Wall -Wextra -Wpedantic -Wconversion -Wshadow
        )
    endif()
endfunction()

enable_project_warnings(engine)

add_executable(demo examples/main.cpp)
target_link_libraries(demo PRIVATE engine)
enable_project_warnings(demo)

add_executable(great_kingdom examples/play.cpp)
set_target_properties(great_kingdom PROPERTIES OUTPUT_NAME GreatKingdom)
target_link_libraries(great_kingdom PRIVATE engine)
enable_project_warnings(great_kingdom)
if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # Keep the playable Windows executable independent of the compiler's DLLs.
    target_link_options(great_kingdom PRIVATE -static -static-libgcc -static-libstdc++)
endif()

include(CTest)
if(BUILD_TESTING)
    foreach(test_name IN ITEMS board_test capture_test territory_test)
        add_executable(${test_name} tests/${test_name}.cpp)
        target_link_libraries(${test_name} PRIVATE engine)
        enable_project_warnings(${test_name})
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
    add_test(NAME console_test COMMAND "${CMAKE_COMMAND}"
        "-DGAME_EXECUTABLE=$<TARGET_FILE:great_kingdom>"
        "-DBINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/console_test.cmake"
    )
endif()
