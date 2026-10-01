#include "MCTS.h"
#include "test_support.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

using namespace kingdom;
using namespace kingdom::test;

namespace {

MCTSOptions options(std::size_t simulations, std::uint64_t seed = 42) {
    MCTSOptions value;
    value.simulations = simulations;
    value.seed = seed;
    return value;
}

const MoveStatistics& statistics_for(const SearchResult& result, Move move) {
    const auto found = std::find_if(result.moves.begin(), result.moves.end(),
                                  [move](const auto& entry) { return entry.move == move; });
    CHECK(found != result.moves.end());
    return *found;
}

void check_statistics(const State& state, const SearchResult& result, std::size_t simulations) {
    CHECK(result.best_move.has_value());
    CHECK(state.is_legal(*result.best_move));
    CHECK(result.simulations == simulations);
    CHECK(result.nodes >= 2);
    CHECK(result.nodes <= result.simulations + 1);
    CHECK(std::isfinite(result.elapsed_seconds));
    CHECK(result.elapsed_seconds >= 0.0);
    CHECK(result.win_rate >= 0.0 && result.win_rate <= 1.0);

    const auto legal = state.legal_moves();
    CHECK(result.moves.size() <= legal.size());
    if (simulations >= static_cast<std::size_t>(Board::kCellCount + 1)) {
        CHECK(result.moves.size() == legal.size());
    }
    std::size_t visits = 0;
    for (const auto& entry : result.moves) {
        CHECK(std::find(legal.begin(), legal.end(), entry.move) != legal.end());
        CHECK(std::count_if(result.moves.begin(), result.moves.end(),
                            [&entry](const auto& other) { return other.move == entry.move; }) == 1);
        CHECK(entry.win_rate >= 0.0 && entry.win_rate <= 1.0);
        visits += entry.visits;
    }
    CHECK(visits == simulations);
    const auto& best = statistics_for(result, *result.best_move);
    CHECK(best.visits > 0);
    CHECK(result.win_rate == best.win_rate);
    for (const auto& entry : result.moves) {
        CHECK(best.visits >= entry.visits);
        if (best.visits == entry.visits) {
            CHECK(best.win_rate >= entry.win_rate);
        }
    }
}

void initial_search_and_input_preservation() {
    State initial;
    MCTS search(options(256));
    const auto result = search.search(initial);
    check_statistics(initial, result, 256);
    CHECK(result.moves.size() == 81); // 80 empty cells and pass.
    CHECK(result.total_rollout_plies > 0);

    State position;
    CHECK(position.play(Move::place(0, 0)).accepted());
    CHECK(position.play(Move::pass()).accepted());
    const State before = position;
    const auto later_result = search.search(position);
    check_statistics(position, later_result, 256);
    CHECK(position.board().cells() == before.board().cells());
    CHECK(position.ownership() == before.ownership());
    CHECK(position.to_play() == before.to_play());
    CHECK(position.consecutive_passes() == before.consecutive_passes());
    CHECK(position.remaining_stones(Cell::Black) == before.remaining_stones(Cell::Black));
    CHECK(position.remaining_stones(Cell::White) == before.remaining_stones(Cell::White));
    CHECK(position.score() == before.score());
    CHECK(position.result() == before.result());
}

void seeded_search_is_reproducible() {
    State state;
    MCTS first(options(192, 918273));
    MCTS second(options(192, 918273));
    const auto a = first.search(state);
    const auto b = second.search(state);
    CHECK(a.best_move == b.best_move);
    CHECK(a.simulations == b.simulations);
    CHECK(a.nodes == b.nodes);
    CHECK(a.total_rollout_plies == b.total_rollout_plies);
    CHECK(a.win_rate == b.win_rate);
    CHECK(a.moves.size() == b.moves.size());
    for (const auto& entry : a.moves) {
        const auto& other = statistics_for(b, entry.move);
        CHECK(entry.visits == other.visits);
        CHECK(entry.win_rate == other.win_rate);
    }
}

void arena_growth_preserves_tree_links() {
    State state;
    MCTS search(options(5000, 1729));
    const auto result = search.search(state);
    check_statistics(state, result, 5000);
    // Exceed the initial arena reserve to exercise indexed links after a move.
    CHECK(result.nodes > 4096);
}

void forced_pass_and_terminal_search() {
    GameRules rules;
    rules.stones_per_player = 1;
    State state(rules);
    CHECK(state.play(Move::place(0, 0)).accepted());
    CHECK(state.play(Move::place(8, 8)).accepted());
    MCTS search(options(64));
    const auto black = search.search(state);
    check_statistics(state, black, 64);
    CHECK(black.best_move->is_pass());
    CHECK(black.moves.size() == 1);
    CHECK(black.win_rate == 0.0);
    CHECK(state.play(*black.best_move).accepted());
    CHECK(!state.result().finished());
    const auto white = search.search(state);
    check_statistics(state, white, 64);
    CHECK(white.best_move->is_pass());
    CHECK(white.win_rate == 1.0);
    CHECK(state.play(*white.best_move).accepted());
    CHECK(state.result().reason == EndReason::TwoPasses);

    const auto terminal = search.search(state);
    CHECK(!terminal.best_move.has_value());
    CHECK(terminal.moves.empty());
    CHECK(terminal.simulations == 0);
    CHECK(terminal.nodes == 0);
    CHECK(terminal.total_rollout_plies == 0);
}

Board one_liberty_endgame() {
    // Both connected groups have (4,4) as their only liberty; no neutral variant.
    Board::Cells cells{};
    for (int row = 0; row < Board::kSize; ++row) {
        for (int col = 0; col < Board::kSize; ++col) {
            if (col < 4 || (col == 4 && row < 4)) {
                cells[static_cast<std::size_t>(row * Board::kSize + col)] = Cell::Black;
            } else if (col > 4 || (col == 4 && row > 4)) {
                cells[static_cast<std::size_t>(row * Board::kSize + col)] = Cell::White;
            }
        }
    }
    return Board(cells);
}

void both_players_capture_and_opponent_chooses_own_win() {
    const Board board = one_liberty_endgame();
    CHECK(board.count(Cell::Black) == 40);
    CHECK(board.count(Cell::White) == 40);
    CHECK(board.liberties({0, 0}).size() == 1);
    CHECK(board.liberties({8, 8}).size() == 1);
    const Move capture = Move::place(4, 4);
    for (const Cell actor : {Cell::Black, Cell::White}) {
        State state(board, actor);
        CHECK(state.legal_moves().size() == 2);
        MCTS search(options(512, 555));
        const auto result = search.search(state);
        check_statistics(state, result, 512);
        CHECK(result.best_move == capture);
        CHECK(result.win_rate == 1.0);
        CHECK(statistics_for(result, Move::pass()).visits > 0);
        // If White passes, Black can capture or pass. Black must favor its own
        // capture rather than the pass that would give White the score victory.
        CHECK(statistics_for(result, Move::pass()).win_rate < 0.5);
        CHECK(state.play(*result.best_move).accepted());
        CHECK(state.result().reason == EndReason::Capture);
        CHECK(state.result().winner == actor);
        CHECK(state.result().captured_stones == 40);
        CHECK(search.search(state).simulations == 0);
    }
}

void suicide_legality_and_territory_candidates() {
    const Board board = make_board({{0, 1}}, {{0, 0}, {0, 2}, {1, 0}, {1, 2}, {2, 1}});
    const Move suicide = Move::place(1, 1);
    State loses(board, Cell::Black);
    CHECK(loses.is_legal(suicide));
    MCTS search(options(768, 2026));
    const auto result = search.search(loses);
    check_statistics(loses, result, 768);
    const auto& losing = statistics_for(result, suicide);
    CHECK(losing.visits > 0);
    CHECK(losing.win_rate == 0.0);
    CHECK(result.best_move != suicide);

    GameRules forbidden_rules;
    forbidden_rules.suicide_rule = SuicideRule::Forbidden;
    State forbidden(board, Cell::Black, forbidden_rules);
    const auto forbidden_result = search.search(forbidden);
    check_statistics(forbidden, forbidden_result, 768);
    CHECK(std::none_of(forbidden_result.moves.begin(), forbidden_result.moves.end(),
                       [suicide](const auto& entry) { return entry.move == suicide; }));

    const Board house = make_board({{0, 2}, {1, 1}, {2, 0}});
    State own_house(house, Cell::Black);
    CHECK(own_house.territory_owner({0, 0}) == Cell::Black);
    const auto own_result = search.search(own_house);
    check_statistics(own_house, own_result, 768);
    CHECK(std::none_of(own_result.moves.begin(), own_result.moves.end(),
                       [](const auto& entry) { return entry.move == Move::place(0, 0); }));

    GameRules allow_own;
    allow_own.allow_own_territory_moves = true;
    State own_allowed(house, Cell::Black, allow_own);
    const auto allowed_result = search.search(own_allowed);
    check_statistics(own_allowed, allowed_result, 768);
    CHECK(statistics_for(allowed_result, Move::place(0, 0)).visits > 0);

    State opponent_house(house, Cell::White);
    const auto opponent_result = search.search(opponent_house);
    check_statistics(opponent_house, opponent_result, 768);
    CHECK(std::none_of(opponent_result.moves.begin(), opponent_result.moves.end(),
                       [](const auto& entry) { return entry.move == Move::place(0, 0); }));
}

void invalid_options_and_time_budget() {
    auto expect_invalid = [](MCTSOptions config) {
        bool rejected = false;
        try {
            MCTS invalid(config);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        CHECK(rejected);
    };
    expect_invalid(options(0));
    auto invalid_exploration = options(1);
    for (const double exploration : {-1.0, std::numeric_limits<double>::infinity(),
                                     std::numeric_limits<double>::quiet_NaN()}) {
        invalid_exploration.exploration = exploration;
        expect_invalid(invalid_exploration);
    }

    auto config = options(100000);
    config.time_limit_ms = 1;
    MCTS timed(config);
    State state;
    const auto result = timed.search(state);
    CHECK(result.simulations <= config.simulations);
    CHECK(result.simulations < config.simulations);
    CHECK(std::isfinite(result.elapsed_seconds));
    CHECK(result.elapsed_seconds >= 0.0);
    CHECK(result.best_move.has_value());
    CHECK(state.is_legal(*result.best_move));

    // A generous deadline must still honor the simulation cap.
    config = options(8);
    config.time_limit_ms = 1000;
    MCTS capped(config);
    check_statistics(state, capped.search(state), 8);
}

void complete_self_play_games() {
    for (std::uint64_t seed = 0; seed < 5; ++seed) {
        MCTS black(options(64, seed * 2));
        MCTS white(options(64, seed * 2 + 1));
        State state;
        int turns = 0;
        // At most 81 placements, with at most one intervening pass and two
        // final passes. Capture and suicide normally terminate much earlier.
        while (!state.result().finished() && turns < 164) {
            const Cell actor = state.to_play();
            const auto result = (actor == Cell::Black ? black : white).search(state);
            CHECK(result.best_move.has_value());
            CHECK(state.is_legal(*result.best_move));
            CHECK(state.play(*result.best_move).accepted());
            CHECK(state.board().at({4, 4}) == Cell::Neutral);
            CHECK(state.remaining_stones(Cell::Black) >= 0);
            CHECK(state.remaining_stones(Cell::White) >= 0);
            ++turns;
        }
        CHECK(state.result().finished());
        CHECK(is_player(state.result().winner));
        CHECK(turns <= 164);
    }
}

} // namespace

int main() {
    return run("mcts", [] {
        initial_search_and_input_preservation();
        seeded_search_is_reproducible();
        arena_growth_preserves_tree_links();
        forced_pass_and_terminal_search();
        both_players_capture_and_opponent_chooses_own_win();
        suicide_legality_and_territory_candidates();
        invalid_options_and_time_budget();
        complete_self_play_games();
    });
}
