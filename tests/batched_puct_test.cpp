#include "BatchedPUCT.h"
#include "test_support.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>

using namespace kingdom;
using namespace kingdom::test;

namespace {

PUCTOptions options(std::size_t simulations) {
    PUCTOptions result;
    result.simulations = simulations;
    return result;
}

PUCTEvaluation evaluation_for(const EncodedPUCTPosition& position) {
    PUCTEvaluation result;
    for (std::size_t i = 0; i < Board::kCellCount; ++i) {
        result.policy[i] = static_cast<double>(i + 1) / 81.0;
    }
    result.value = position.features[405] != 0.0F ? 0.25 : -0.5;
    return result;
}

std::vector<PUCTEvaluation> evaluate(const std::vector<EncodedPUCTPosition>& positions) {
    std::vector<PUCTEvaluation> result;
    for (const auto& position : positions) result.push_back(evaluation_for(position));
    return result;
}

std::size_t visit_count(const PUCTSearchResult& result) {
    return std::accumulate(result.moves.begin(), result.moves.end(), std::size_t{0},
        [](std::size_t total, const PUCTMoveStatistics& move) { return total + move.visits; });
}

void compare_results(const PUCTSearchResult& lhs, const PUCTSearchResult& rhs) {
    CHECK(lhs.best_move == rhs.best_move);
    CHECK(lhs.simulations == rhs.simulations);
    CHECK(lhs.nodes == rhs.nodes);
    CHECK(lhs.network_evaluations == rhs.network_evaluations);
    CHECK(lhs.root_value == rhs.root_value);
    CHECK(lhs.best_value == rhs.best_value);
    CHECK(lhs.moves.size() == rhs.moves.size());
    for (std::size_t i = 0; i < lhs.moves.size(); ++i) {
        CHECK(lhs.moves[i].move == rhs.moves[i].move);
        CHECK(lhs.moves[i].visits == rhs.moves[i].visits);
        CHECK(lhs.moves[i].value == rhs.moves[i].value);
        CHECK(lhs.moves[i].prior == rhs.moves[i].prior);
    }
}

void sequential_parity_and_encoding() {
    for (std::size_t count : {std::size_t{1}, std::size_t{31}, std::size_t{300}}) {
        PUCT legacy(options(count));
        BatchedPUCT batched(options(count), 1, false);
        State state;
        CHECK(state.play(Move::place(0, 0)).accepted());
        CHECK(state.play(Move::pass()).accepted());
        const State original = state;
        compare_results(legacy.search(state, [](const State& leaf) {
            return evaluation_for(encode_puct_position(leaf));
        }), batched.search(state, evaluate));
        CHECK(state.board().cells() == original.board().cells());
        CHECK(state.ownership() == original.ownership());
        CHECK(state.to_play() == original.to_play());
        CHECK(state.consecutive_passes() == original.consecutive_passes());
        CHECK(state.result() == original.result());
        const auto encoded = encode_puct_position(state);
        CHECK(encoded.features[0] == 1.0F);
        CHECK(encoded.features[162 + 40] == 1.0F);
        CHECK(encoded.features[405] == 1.0F);
        CHECK(encoded.features[486] == static_cast<float>(40 / 41.0));
        CHECK(encoded.features[567] == 1.0F);
        CHECK(encoded.features[648] == 0.5F);
        CHECK(!encoded.legal_mask[0]);
        CHECK(!encoded.legal_mask[40]);
        CHECK(encoded.legal_mask[81]);
        for (std::size_t i = 0; i < Board::kCellCount; ++i) {
            CHECK(encoded.features[729 + i] == (encoded.legal_mask[i] ? 1.0F : 0.0F));
        }
    }
}

void batching_budget_and_duplicate_reservations() {
    for (const std::size_t count : {std::size_t{1}, std::size_t{7}, std::size_t{19}, std::size_t{301}}) {
        BatchedPUCT search(options(count), 8, false);
        std::size_t calls = 0;
        std::size_t positions_seen = 0;
        std::size_t maximum = 0;
        const auto result = search.search(State{}, [&](const auto& positions) {
            ++calls;
            positions_seen += positions.size();
            maximum = std::max(maximum, positions.size());
            CHECK(!positions.empty());
            CHECK(positions.size() <= 8);
            std::set<std::array<float, 810>> distinct;
            for (const auto& position : positions) CHECK(distinct.insert(position.features).second);
            return evaluate(positions);
        });
        CHECK(result.simulations == count);
        CHECK(visit_count(result) == count);
        CHECK(result.network_evaluations == positions_seen);
        CHECK(search.stats().network_evaluations == positions_seen);
        CHECK(search.stats().inference_batches == calls);
        CHECK(search.stats().max_observed_batch_size == maximum);
        CHECK(result.network_evaluations <= count + 1);
        CHECK(result.nodes <= count + 1);
        if (count >= 8) CHECK(maximum == 8);
    }
}

void subtree_reuse_and_delta_visits() {
    BatchedPUCT search(options(64), 8, true);
    State state;
    auto result = search.search(state, evaluate);
    CHECK(search.stats().reused_nodes == 0);
    CHECK(search.stats().inherited_visits == 0);
    const auto repeated = search.search(state, evaluate);
    CHECK(search.stats().reused_nodes == result.nodes);
    CHECK(search.stats().inherited_visits == 64);
    CHECK(repeated.simulations == 64);
    CHECK(visit_count(repeated) == 64);
    CHECK(repeated.network_evaluations <= 64);
    CHECK(search.advance(*repeated.best_move));
    CHECK(state.play(*repeated.best_move).accepted());
    result = search.search(state, evaluate);
    CHECK(search.stats().reused_nodes > 0);
    CHECK(search.stats().inherited_visits > 0);
    CHECK(search.stats().advance_calls == 1);
    CHECK(search.stats().reuse_hits == 1);
    CHECK(result.simulations == 64);
    CHECK(visit_count(result) == 64);
    CHECK(result.nodes < repeated.nodes + 65);
    CHECK(!search.advance(Move::place(4, 4)));
    CHECK(!search.advance(Move::place(-1, 0)));
    const auto after_illegal = search.search(state, evaluate);
    CHECK(search.stats().reused_nodes == result.nodes);
    CHECK(search.stats().inherited_visits > 0);
    CHECK(visit_count(after_illegal) == 64);
    CHECK(search.stats().advance_calls == 3);
    CHECK(search.stats().reuse_hits == 1);
    search.clear();
    CHECK(search.search(state, evaluate).network_evaluations > 0);
    CHECK(search.stats().reused_nodes == 0);
}

void complete_state_mismatch_resets() {
    BatchedPUCT search(options(16), 4, true);
    State state;
    static_cast<void>(search.search(state, evaluate));
    State passed = state;
    CHECK(passed.play(Move::pass()).accepted());
    CHECK(passed.board().cells() == state.board().cells());
    CHECK(search.search(passed, evaluate).simulations == 16);
    CHECK(search.stats().reused_nodes == 0);
    // Same board and actor, but no pass history: setup constructor differs.
    State setup(passed.board(), passed.to_play());
    CHECK(search.search(setup, evaluate).simulations == 16);
    CHECK(search.stats().reused_nodes == 0);
    CHECK(search.stats().inherited_visits == 0);
    State no_neutral({}, std::nullopt);
    CHECK(search.search(no_neutral, evaluate).simulations == 16);
    CHECK(search.stats().reused_nodes == 0);
}

void terminal_and_capture_values() {
    Board::Cells cells{};
    for (int row = 0; row < 9; ++row) {
        for (int col = 0; col < 9; ++col) {
            if (col < 4 || (col == 4 && row < 4)) cells[row * 9 + col] = Cell::Black;
            else if (col > 4 || (col == 4 && row > 4)) cells[row * 9 + col] = Cell::White;
        }
    }
    for (const Cell actor : {Cell::Black, Cell::White}) {
        State state(Board(cells), actor);
        BatchedPUCT search(options(32), 8, true);
        std::size_t calls = 0;
        const auto result = search.search(state, [&](const auto& positions) {
            ++calls;
            return evaluate(positions);
        });
        CHECK(result.best_move == Move::place(4, 4));
        CHECK(result.best_value == 1.0);
        CHECK(visit_count(result) == 32);
        CHECK(result.network_evaluations == 1);
        CHECK(calls == 1);
        CHECK(search.advance(*result.best_move));
        CHECK(state.play(*result.best_move).accepted());
        CHECK(state.result().reason == EndReason::Capture);
        CHECK(search.search(state, {}).root_value == -1.0);
        CHECK(search.stats().network_evaluations == 0);
        const auto encoded = encode_puct_position(state);
        CHECK(std::none_of(encoded.legal_mask.begin(), encoded.legal_mask.end(), [](bool x) { return x; }));
    }
    State two_passes;
    CHECK(two_passes.play(Move::pass()).accepted());
    CHECK(two_passes.play(Move::pass()).accepted());
    BatchedPUCT search(options(8));
    const auto terminal = search.search(two_passes, {});
    CHECK(!terminal.best_move);
    CHECK(terminal.simulations == 0);
    CHECK(terminal.moves.empty());
    CHECK(terminal.root_value == -1.0);
}

void validation_failure_recovery_and_deadline() {
    auto invalid_constructor = [](PUCTOptions config, std::size_t batch) {
        bool rejected = false;
        try { BatchedPUCT search(config, batch); }
        catch (const std::invalid_argument&) { rejected = true; }
        CHECK(rejected);
    };
    invalid_constructor(options(0), 8);
    invalid_constructor(options(8), 0);
    auto noisy = options(8);
    noisy.dirichlet_epsilon = 0.1;
    invalid_constructor(noisy, 8);
    BatchedPUCT search(options(16), 8, true);
    for (int failure = 0; failure < 5; ++failure) {
        bool rejected = false;
        std::size_t call = 0;
        try {
            static_cast<void>(search.search(State{}, [&](const auto& positions) {
                auto result = evaluate(positions);
                if (++call == 2) {
                    if (failure == 0) throw std::runtime_error("inference failed");
                    if (failure == 1) result.pop_back();
                    if (failure == 2) result[0].value = std::numeric_limits<double>::quiet_NaN();
                    if (failure == 3) result[0].policy[0] = -1.0;
                    if (failure == 4) result[0].policy.fill(0.0);
                }
                return result;
            }));
        } catch (const std::exception&) { rejected = true; }
        CHECK(rejected);
        const auto recovered = search.search(State{}, evaluate);
        CHECK(search.stats().reused_nodes == 0);
        CHECK(search.stats().inherited_visits == 0);
        CHECK(visit_count(recovered) == 16);
        search.clear();
    }
    GameRules changed;
    changed.stones_per_player = 1;
    bool rejected = false;
    try { static_cast<void>(search.search(State(changed), evaluate)); }
    catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
    auto config = options(100000);
    config.time_limit_ms = 1;
    BatchedPUCT timed(config, 8);
    const auto limited = timed.search(State{}, evaluate);
    CHECK(limited.simulations >= 1);
    CHECK(limited.simulations < config.simulations);
    CHECK(visit_count(limited) == limited.simulations);
}

void full_game_and_large_finite_policy() {
    State state;
    BatchedPUCT black(options(16), 4);
    BatchedPUCT white(options(16), 4);
    int plies = 0;
    while (!state.result().finished() && plies < 164) {
        auto& actor = state.to_play() == Cell::Black ? black : white;
        const auto result = actor.search(state, evaluate);
        CHECK(visit_count(result) == 16);
        CHECK(state.play(*result.best_move).accepted());
        static_cast<void>(black.advance(*result.best_move));
        static_cast<void>(white.advance(*result.best_move));
        ++plies;
    }
    CHECK(state.result().finished());
    auto config = options(16);
    config.c_puct = std::numeric_limits<double>::max();
    BatchedPUCT large(config, 4, false);
    const auto result = large.search(State{}, [](const auto& positions) {
        auto results = evaluate(positions);
        for (auto& entry : results) entry.policy.fill(std::numeric_limits<double>::max());
        return results;
    });
    CHECK(visit_count(result) == 16);
    CHECK(std::isfinite(result.root_value));
    for (const auto& move : result.moves) CHECK(std::isfinite(move.prior));
}

} // namespace

int main() {
    return run("batched_puct", [] {
        sequential_parity_and_encoding();
        batching_budget_and_duplicate_reservations();
        subtree_reuse_and_delta_visits();
        complete_state_mismatch_resets();
        terminal_and_capture_values();
        validation_failure_recovery_and_deadline();
        full_game_and_large_finite_policy();
    });
}
