#include "TacticalSolver.h"
#include "test_support.h"

#include <algorithm>
#include <array>
#include <random>

using namespace kingdom;
using namespace kingdom::test;

namespace {

TacticalSolverOptions unlimited(int depth) {
    return {depth, 1000000, 0};
}

bool contains(const std::vector<Move>& moves, Move move) {
    return std::find(moves.begin(), moves.end(), move) != moves.end();
}

TacticalOutcome opposite(TacticalOutcome value) {
    return value == TacticalOutcome::Win ? TacticalOutcome::Loss
         : value == TacticalOutcome::Loss ? TacticalOutcome::Win : TacticalOutcome::Unknown;
}

// Deliberately independent, unordered, complete depth-limited reference.
TacticalOutcome reference(const State& state, int depth) {
    if (state.result().finished()) {
        return state.result().winner == state.to_play() ? TacticalOutcome::Win : TacticalOutcome::Loss;
    }
    if (depth == 0) return TacticalOutcome::Unknown;
    bool unknown = false;
    for (Move move : state.legal_moves()) {
        State child = state;
        CHECK(child.play(move).accepted());
        auto value = opposite(reference(child, depth - 1));
        if (value == TacticalOutcome::Win) return value;
        if (value == TacticalOutcome::Unknown) unknown = true;
    }
    return unknown ? TacticalOutcome::Unknown : TacticalOutcome::Loss;
}

void check_partition_and_certificates(const State& state, const TacticalSolverResult& result, int depth) {
    const auto legal = state.legal_moves();
    CHECK(result.winning_moves.size() + result.losing_moves.size() + result.unknown_moves.size() == legal.size());
    for (Move move : legal) {
        CHECK(static_cast<int>(contains(result.winning_moves, move)) +
              static_cast<int>(contains(result.losing_moves, move)) +
              static_cast<int>(contains(result.unknown_moves, move)) == 1);
        if (contains(result.unknown_moves, move)) continue;
        State child = state;
        CHECK(child.play(move).accepted());
        const auto value = opposite(reference(child, depth - 1));
        CHECK(value == (contains(result.winning_moves, move) ? TacticalOutcome::Win : TacticalOutcome::Loss));
    }
    if (result.outcome == TacticalOutcome::Win) CHECK(!result.winning_moves.empty() || state.result().finished());
    if (result.outcome == TacticalOutcome::Loss && !state.result().finished()) {
        CHECK(result.losing_moves.size() == legal.size());
        CHECK(result.unknown_moves.empty());
    }
    if (result.outcome != TacticalOutcome::Unknown) {
        State line = state;
        for (Move move : result.principal_variation) CHECK(line.play(move).accepted());
        CHECK(line.result().finished());
        const Cell expected = result.outcome == TacticalOutcome::Win ? state.to_play() : opponent(state.to_play());
        CHECK(line.result().winner == expected);
        CHECK(static_cast<int>(result.principal_variation.size()) == result.proof_depth);
        CHECK(result.proof_depth <= depth);
    }
}

void capture_priority_and_preservation() {
    State state(make_board({{0, 6}, {1, 6}, {1, 8}, {2, 7}},
                           {{0, 7}, {1, 7}, {2, 8}}), Cell::Black);
    const State original = state;
    const auto result = solve_tactics(state, unlimited(1));
    CHECK(result.outcome == TacticalOutcome::Win);
    CHECK(contains(result.winning_moves, Move::place(0, 8)));
    CHECK(result.proof_depth == 1);
    State terminal = state;
    CHECK(terminal.play(result.principal_variation.front()).accepted());
    CHECK(terminal.result().reason == EndReason::Capture);
    CHECK(terminal.result().captured_stones == 2);
    CHECK(state.board().cells() == original.board().cells());
    CHECK(state.ownership() == original.ownership());
    CHECK(state.to_play() == original.to_play());
    CHECK(state.result() == original.result());
    CHECK(state.remaining_stones(Cell::Black) == original.remaining_stones(Cell::Black));
    CHECK(state.remaining_stones(Cell::White) == original.remaining_stones(Cell::White));
    check_partition_and_certificates(state, result, 1);
    auto terminal_result = solve_tactics(terminal, unlimited(0));
    CHECK(terminal_result.outcome == TacticalOutcome::Loss);
    CHECK(terminal_result.proof_depth == 0);
    CHECK(terminal_result.winning_moves.empty());
    CHECK(terminal_result.losing_moves.empty());
    CHECK(terminal_result.unknown_moves.empty());
}

void pass_and_stock_are_real_defenses() {
    GameRules rules;
    rules.stones_per_player = 1;
    State exhausted(make_board({{0, 0}}, {{8, 8}}), Cell::Black, rules);
    CHECK(exhausted.legal_moves().size() == 1);
    auto horizon = solve_tactics(exhausted, unlimited(1));
    CHECK(horizon.outcome == TacticalOutcome::Unknown);
    CHECK(horizon.unknown_moves.size() == 1);
    auto loss = solve_tactics(exhausted, unlimited(2));
    CHECK(loss.outcome == TacticalOutcome::Loss);
    CHECK(loss.losing_moves.size() == 1);
    CHECK(loss.proof_depth == 2);
    check_partition_and_certificates(exhausted, loss, 2);
    CHECK(exhausted.play(Move::pass()).accepted());
    auto white_wins = solve_tactics(exhausted, unlimited(1));
    CHECK(white_wins.outcome == TacticalOutcome::Win);
    CHECK(white_wins.winning_moves.front().is_pass());
    check_partition_and_certificates(exhausted, white_wins, 1);
    State white_start(make_board({{0, 0}}, {{8, 8}}), Cell::White, rules);
    CHECK(white_start.play(Move::pass()).accepted());
    auto black_loses = solve_tactics(white_start, unlimited(1));
    CHECK(black_loses.outcome == TacticalOutcome::Loss);
    CHECK(black_loses.losing_moves.front().is_pass());
    CHECK(black_loses.proof_depth == 1);
}

void suicide_and_variant() {
    Board board = make_board({{0, 1}}, {{0, 0}, {0, 2}, {1, 0}, {1, 2}, {2, 1}});
    State state(board, Cell::Black);
    const auto loses = solve_tactics(state, unlimited(1));
    CHECK(contains(loses.losing_moves, Move::place(1, 1)));
    CHECK(loses.outcome == TacticalOutcome::Unknown);
    check_partition_and_certificates(state, loses, 1);
    GameRules rules;
    rules.suicide_rule = SuicideRule::Forbidden;
    State forbidden(board, Cell::Black, rules);
    const auto result = solve_tactics(forbidden, unlimited(1));
    CHECK(!contains(result.losing_moves, Move::place(1, 1)));
    CHECK(!contains(result.unknown_moves, Move::place(1, 1)));
    CHECK(!contains(result.winning_moves, Move::place(1, 1)));
    check_partition_and_certificates(forbidden, result, 1);
}

void horizon_and_budget_never_mean_loss() {
    State state;
    auto horizon = solve_tactics(state, unlimited(0));
    CHECK(horizon.outcome == TacticalOutcome::Unknown);
    CHECK(horizon.unknown_moves.size() == state.legal_moves().size());
    auto limited = solve_tactics(state, {12, 1, 0});
    CHECK(limited.outcome == TacticalOutcome::Unknown);
    CHECK(limited.nodes == 1);
    CHECK(limited.budget_exhausted);
    CHECK(limited.winning_moves.empty());
    CHECK(limited.losing_moves.empty());
    CHECK(limited.unknown_moves.size() == state.legal_moves().size());
    for (std::size_t cap : {2U, 8U, 50U}) {
        auto partial = solve_tactics(state, {12, cap, 0});
        CHECK(partial.nodes <= cap);
        CHECK(partial.outcome == TacticalOutcome::Unknown);
        CHECK(partial.budget_exhausted);
        CHECK(partial.losing_moves.empty());
    }
    auto timed = solve_tactics(state, {12, 1000000, 1});
    CHECK(timed.budget_exhausted);
    CHECK(timed.outcome == TacticalOutcome::Unknown);
    bool failed = false;
    try { static_cast<void>(solve_tactics(state, {-1, 1, 0})); }
    catch (const std::invalid_argument&) { failed = true; }
    CHECK(failed);
    failed = false;
    try { static_cast<void>(solve_tactics(state, {1, 0, 0})); }
    catch (const std::invalid_argument&) { failed = true; }
    CHECK(failed);
}

void complete_reference_cross_check() {
    std::vector<State> positions;
    positions.emplace_back();
    positions.emplace_back(make_board({{1, 2}, {2, 1}}, {{2, 2}}), Cell::Black);
    positions.emplace_back(make_board({{1, 2}, {2, 1}, {3, 2}}, {{2, 2}}), Cell::White);
    positions.emplace_back(make_board({{3, 3}, {4, 2}}, {{4, 3}}, Position{4, 4}), Cell::Black);
    for (const State& state : positions) {
        for (int depth : {1, 2}) {
            auto proof = solve_tactics(state, unlimited(depth));
            CHECK(!proof.budget_exhausted);
            CHECK(proof.outcome == reference(state, depth));
            check_partition_and_certificates(state, proof, depth);
        }
    }
}

} // namespace

int main() {
    return run("tactical_solver", [] {
        capture_priority_and_preservation();
        pass_and_stock_are_real_defenses();
        suicide_and_variant();
        horizon_and_budget_never_mean_loss();
        complete_reference_cross_check();
    });
}
