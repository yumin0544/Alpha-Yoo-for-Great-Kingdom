cmake_minimum_required(VERSION 3.20)

project(BoardGameEngine VERSION 0.1.0 LANGUAGES CXX)

option(BUILD_PYTHON_BINDINGS "Build the my_board_engine Python extension" OFF)
option(BUILD_NEURAL_TESTS "Run optional PyTorch policy/value integration tests" OFF)
if(BUILD_NEURAL_TESTS AND NOT BUILD_PYTHON_BINDINGS)
    message(FATAL_ERROR "BUILD_NEURAL_TESTS requires BUILD_PYTHON_BINDINGS=ON")
endif()

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

add_library(mcts
    mcts/src/Node.cpp
    mcts/src/MCTS.cpp
    mcts/src/PUCT.cpp
)
target_link_libraries(mcts PUBLIC engine)
target_include_directories(mcts PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/mcts/include>"
)
set_target_properties(mcts PROPERTIES CXX_EXTENSIONS OFF)
enable_project_warnings(mcts)

if(BUILD_PYTHON_BINDINGS)
    # These targets are linked into a shared Python module on Unix as well.
    set_target_properties(engine mcts PROPERTIES POSITION_INDEPENDENT_CODE ON)
    find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
    set(PYBIND11_FINDPYTHON ON)
    find_package(pybind11 3.0 CONFIG REQUIRED)
    pybind11_add_module(my_board_engine bindings/module.cpp)
    target_link_libraries(my_board_engine PRIVATE mcts)
    set_target_properties(my_board_engine PROPERTIES CXX_EXTENSIONS OFF)
    enable_project_warnings(my_board_engine)
    # pip installs this extension at the top level: import my_board_engine.
    install(TARGETS my_board_engine LIBRARY DESTINATION . RUNTIME DESTINATION .)
endif()

add_executable(mcts_demo examples/mcts_demo.cpp)
set_target_properties(mcts_demo PROPERTIES OUTPUT_NAME MCTSDemo)
target_link_libraries(mcts_demo PRIVATE mcts)
enable_project_warnings(mcts_demo)
if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_options(mcts_demo PRIVATE -static -static-libgcc -static-libstdc++)
endif()

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
    add_executable(mcts_test tests/mcts_test.cpp)
    target_link_libraries(mcts_test PRIVATE mcts)
    enable_project_warnings(mcts_test)
    add_test(NAME mcts_test COMMAND mcts_test)
    add_test(NAME console_test COMMAND "${CMAKE_COMMAND}"
        "-DGAME_EXECUTABLE=$<TARGET_FILE:great_kingdom>"
        "-DBINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/console_test.cmake"
    )
    if(BUILD_PYTHON_BINDINGS)
        add_executable(binding_reference tests/binding_reference.cpp)
        target_link_libraries(binding_reference PRIVATE mcts)
        enable_project_warnings(binding_reference)
        add_test(NAME python_binding_test COMMAND "${CMAKE_COMMAND}" -E env
            "PYTHONPATH=$<TARGET_FILE_DIR:my_board_engine>"
            "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/python_binding_test.py"
            --reference "$<TARGET_FILE:binding_reference>"
        )
        set_tests_properties(python_binding_test PROPERTIES TIMEOUT 60)
        if(BUILD_NEURAL_TESTS)
            execute_process(COMMAND "${Python_EXECUTABLE}" -c "import torch, numpy"
                RESULT_VARIABLE neural_dependencies
                ERROR_VARIABLE neural_import_error
            )
            if(NOT neural_dependencies EQUAL 0)
                message(FATAL_ERROR "PyTorch/NumPy are required for neural tests: ${neural_import_error}")
            endif()
            if(WIN32)
                set(neural_path_separator ";")
            else()
                set(neural_path_separator ":")
            endif()
            add_test(NAME neural_test COMMAND "${CMAKE_COMMAND}" -E env
                "PYTHONPATH=$<TARGET_FILE_DIR:my_board_engine>${neural_path_separator}${CMAKE_CURRENT_SOURCE_DIR}/python"
                "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/neural_test.py"
            )
            set_tests_properties(neural_test PROPERTIES TIMEOUT 60)
            add_test(NAME puct_test COMMAND "${CMAKE_COMMAND}" -E env
                "PYTHONPATH=$<TARGET_FILE_DIR:my_board_engine>${neural_path_separator}${CMAKE_CURRENT_SOURCE_DIR}/python"
                "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/puct_test.py"
            )
            set_tests_properties(puct_test PROPERTIES TIMEOUT 120)
        endif()
    endif()
endif()
