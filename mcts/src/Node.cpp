#include "Node.h"

#include <stdexcept>
#include <utility>

namespace kingdom {

Node::Node(std::size_t parent_index, Move incoming_move, Cell player,
           std::vector<Move> candidates, bool is_terminal)
    : parent(parent_index), move(incoming_move), reward_player(player),
      terminal(is_terminal), untried_moves(std::move(candidates)) {
    if (!is_player(player)) {
        throw std::invalid_argument("MCTS reward player must be Black or White");
    }
}

double Node::win_rate() const noexcept {
    return visits == 0 ? 0.0 : wins / static_cast<double>(visits);
}

void Node::record_result(Cell winner) {
    if (!is_player(winner)) {
        throw std::invalid_argument("A completed rollout must have a winner");
    }
    ++visits;
    if (winner == reward_player) {
        wins += 1.0;
    }
}

} // namespace kingdom
