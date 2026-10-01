#pragma once

#include "board/Move.h"

#include <cstddef>
#include <limits>
#include <vector>

namespace kingdom {

// Tree links are arena indices; no pointers can be invalidated by arena growth.
// Each edge's wins belong to the player who made its incoming move.
struct Node {
    static constexpr std::size_t kNoParent = std::numeric_limits<std::size_t>::max();

    std::size_t parent;
    Move move;
    Cell reward_player;
    bool terminal;
    std::size_t visits = 0;
    double wins = 0.0;
    std::vector<std::size_t> children;
    std::vector<Move> untried_moves;

    Node(std::size_t parent_index, Move incoming_move, Cell player,
         std::vector<Move> candidates, bool is_terminal);

    [[nodiscard]] double win_rate() const noexcept;
    void record_result(Cell winner);
};

} // namespace kingdom
