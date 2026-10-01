cmake_minimum_required(VERSION 3.20)

project(BoardGameEngine)

set(CMAKE_CXX_STANDARD 20)


include_directories(
    engine/include
)


add_library(
    engine
    engine/src/Board.cpp
)


add_executable(
    demo
    examples/main.cpp
)


target_link_libraries(
    demo
    engine
)