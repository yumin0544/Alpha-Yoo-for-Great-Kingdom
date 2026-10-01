#pragma once

#include "board/Board.h"

#include <exception>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>

namespace kingdom::test {

inline void require(bool condition, const char* expression, const char* file, int line) {
    if (!condition) {
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) +
                                 ": failed: " + expression);
    }
}

inline Board make_board(std::initializer_list<Position> black,
                        std::initializer_list<Position> white = {},
                        std::optional<Position> neutral = std::nullopt) {
    Board board(neutral);
    for (Position pos : black) {
        require(board.place(pos, Cell::Black), "black fixture placement", __FILE__, __LINE__);
    }
    for (Position pos : white) {
        require(board.place(pos, Cell::White), "white fixture placement", __FILE__, __LINE__);
    }
    return board;
}

inline Cell owner_at(const Board::Territory& territory, Position pos) {
    return territory.owners[static_cast<std::size_t>(Board::index(pos))];
}

template <class Function>
int run(const char* suite, Function tests) {
    try {
        tests();
        std::cout << suite << ": all checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << suite << ": " << error.what() << '\n';
        return 1;
    }
}

} // namespace kingdom::test

// Unlike assert, these checks remain enabled in Release builds.
#define CHECK(...) ::kingdom::test::require((__VA_ARGS__), #__VA_ARGS__, __FILE__, __LINE__)
