#pragma once

#include "Node.h"
#include "board/State.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

namespace kingdom {

struct MCTSOptions {
    std::size_t simulations = 1000;
    double exploration = 1.4142135623730951;
    std::uint64_t seed = 42;
    // Zero disables the time limit. Completed simulations are never truncated.
    std::uint64_t time_limit_ms = 0;
};

struct MoveStatistics {
    Move move;
    std::size_t visits = 0;
    double win_rate = 0.0;
};

struct SearchResult {
    std::optional<Move> best_move;
    std::size_t simulations = 0;
    std::size_t nodes = 0;
    std::size_t total_rollout_plies = 0;
    // Estimated root-player win rate for best_move, not an aggregate tree rate.
    double win_rate = 0.0;
    double elapsed_seconds = 0.0;
    // Expanded accepted root moves only, ordered by board index then pass.
    std::vector<MoveStatistics> moves;
};

// Pure UCT with uniform random terminal playouts. No neural net or heuristic score.
// A search builds a fresh tree and does not mutate the caller's State.
class MCTS {
public:
    explicit MCTS(MCTSOptions options = {});

    [[nodiscard]] SearchResult search(const State& state);
    [[nodiscard]] const MCTSOptions& options() const noexcept { return options_; }

private:
    MCTSOptions options_;
    std::mt19937_64 random_;

    [[nodiscard]] static std::vector<Move> candidates(const State& state);
    [[nodiscard]] Move take_random(std::vector<Move>& moves);
    [[nodiscard]] std::size_t select_child(const std::vector<Node>& tree,
                                         std::size_t parent);
    [[nodiscard]] Cell rollout(State& state, std::size_t& total_plies);
};

} // namespace kingdom
