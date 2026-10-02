#pragma once

#include "board/State.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <random>
#include <vector>

namespace kingdom {

struct PUCTOptions {
    std::size_t simulations = 128;
    double c_puct = 1.5;
    std::uint64_t seed = 42;
    // Zero disables the soft deadline. At least one simulation is completed.
    std::uint64_t time_limit_ms = 0;
    double dirichlet_alpha = 0.3;
    // Noise is applied once to root priors; zero disables it for evaluation.
    double dirichlet_epsilon = 0.0;
};

struct PUCTEvaluation {
    // Non-negative policy weights: row * 9 + col, followed by pass at 81.
    // The search masks illegal actions and renormalizes the remaining weights.
    std::array<double, Board::kCellCount + 1> policy{};
    // Expected outcome in [-1, 1], from this state's current player's view.
    double value = 0.0;
};

using PUCTEvaluationFunction = std::function<PUCTEvaluation(const State&)>;

struct PUCTMoveStatistics {
    Move move;
    double prior = 0.0;
    std::size_t visits = 0;
    // Mean outcome for the parent player who chooses this move, in [-1, 1].
    double value = 0.0;
};

struct PUCTSearchResult {
    std::optional<Move> best_move;
    std::size_t simulations = 0;
    std::size_t nodes = 0;
    std::size_t network_evaluations = 0;
    // Mean backed-up outcome over completed simulations for the root player.
    double root_value = 0.0;
    double best_value = 0.0;
    double elapsed_seconds = 0.0;
    // All legal root actions, including unvisited ones; board order, then pass.
    std::vector<PUCTMoveStatistics> moves;
};

// Policy/value PUCT. Tree operations run in C++; the evaluator is called only
// when expanding a non-terminal leaf. No random terminal playouts are performed.
// Search creates a fresh tree, preserving the caller's complete State.
class PUCT {
public:
    static constexpr int kActionCount = Board::kCellCount + 1;

    explicit PUCT(PUCTOptions options = {});

    [[nodiscard]] PUCTSearchResult search(const State& state,
                                         const PUCTEvaluationFunction& evaluate);
    [[nodiscard]] const PUCTOptions& options() const noexcept { return options_; }

private:
    PUCTOptions options_;
    std::mt19937_64 random_;
};

} // namespace kingdom
