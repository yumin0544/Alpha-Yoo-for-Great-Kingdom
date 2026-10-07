#pragma once

#include "board/State.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace kingdom {

// A certificate of the final game outcome, not a neural estimate. Unknown is
// never interchangeable with a safe move, a draw, or a disproven attack.
enum class TacticalOutcome { Unknown, Win, Loss };

struct TacticalSolverOptions {
    int max_depth = 12;
    std::size_t max_nodes = 100000;
    // Zero disables the wall-clock bound. Node/depth bounds still apply.
    std::uint64_t time_limit_ms = 1000;
};

struct TacticalSolverResult {
    // All outcomes are from the supplied state's current player's viewpoint.
    TacticalOutcome outcome = TacticalOutcome::Unknown;
    std::vector<Move> winning_moves;
    std::vector<Move> losing_moves;
    std::vector<Move> unknown_moves;
    // One representative line through a proved outcome, not all defenses.
    std::vector<Move> principal_variation;
    int proof_depth = 0;
    // Largest fully returned depth-limited root analysis (may still be Unknown).
    int completed_depth = 0;
    std::size_t nodes = 0;
    double elapsed_ms = 0.0;
    bool budget_exhausted = false;
};

// Sound bounded minimax using the existing engine's exact State transitions.
// Go liberty/connection features order moves only; no legal defenses are pruned
// when proving a loss. A root win needs one certified action, so other actions
// may remain unclassified even when the overall outcome is Win.
[[nodiscard]] TacticalSolverResult solve_tactics(
    const State& state, TacticalSolverOptions options = {});

} // namespace kingdom
