#include "board/State.h"
#include "test_support.h"

#include <algorithm>

using namespace kingdom;
using namespace kingdom::test;

namespace {

void single_stone_capture() {
    State state(make_board({{1, 2}, {2, 1}, {3, 2}}, {{2, 2}}), Cell::Black);
    CHECK(state.board().liberties({2, 2}).size() == 1);
    const auto outcome = state.play(Move::place(2, 3));
    CHECK(outcome.accepted());
    CHECK(outcome.result.finished());
    CHECK(state.result().reason == EndReason::Capture);
    CHECK(state.result().winner == Cell::Black);
    CHECK(state.result().captured_stones == 1);
    CHECK(state.board().at({2, 2}) == Cell::Empty);
    CHECK(state.board().at({2, 3}) == Cell::Black);
    CHECK(state.play(Move::pass()).error == MoveError::GameOver);
}

void connected_group_capture_and_remaining_gap() {
    State state(make_board({{1, 2}, {1, 3}, {2, 1}, {3, 2}, {3, 3}},
                           {{2, 2}, {2, 3}}), Cell::Black);
    CHECK(state.play(Move::place(2, 4)).accepted());
    CHECK(state.result().winner == Cell::Black);
    CHECK(state.result().captured_stones == 2);
    CHECK(state.board().count(Cell::White) == 0);

    State gap(make_board({{1, 2}, {2, 1}, {3, 2}, {3, 3}}, {{2, 2}, {2, 3}}),
              Cell::Black);
    CHECK(gap.play(Move::place(1, 3)).accepted());
    CHECK(!gap.result().finished());
    CHECK(gap.board().count(Cell::White) == 2);
    CHECK(gap.board().liberties({2, 2}).size() == 1);
}

void simultaneous_surround_prefers_mover() {
    // User's simultaneous-surround example, placed against the actual top/right edges.
    State state(make_board({{0, 6}, {1, 6}, {1, 8}, {2, 7}},
                           {{0, 7}, {1, 7}, {2, 8}}), Cell::Black);
    CHECK(state.play(Move::place(0, 8)).accepted());
    CHECK(state.result().reason == EndReason::Capture);
    CHECK(state.result().winner == Cell::Black);
    CHECK(state.result().captured_stones == 2);
    CHECK(state.board().at({0, 8}) == Cell::Black);
    CHECK(state.board().at({1, 8}) == Cell::Black);
    CHECK(state.board().at({0, 7}) == Cell::Empty);
    CHECK(state.board().at({1, 7}) == Cell::Empty);
    CHECK(state.board().at({2, 8}) == Cell::White);
}

void neutral_wall_and_white_capture() {
    State neutral(make_board({{3, 3}, {4, 2}}, {{4, 3}}, Position{4, 4}), Cell::Black);
    CHECK(neutral.play(Move::place(5, 3)).accepted());
    CHECK(neutral.result().reason == EndReason::Capture);
    CHECK(neutral.result().captured_stones == 1);
    CHECK(neutral.board().at({4, 4}) == Cell::Neutral);

    State white(make_board({{2, 2}}, {{1, 2}, {2, 1}, {3, 2}}), Cell::White);
    CHECK(white.play(Move::place(2, 3)).accepted());
    CHECK(white.result().winner == Cell::White);
    CHECK(white.board().count(Cell::Black) == 0);
}

void configurable_suicide() {
    // Connecting to an existing black stone avoids a pre-existing white territory cell.
    const Board board = make_board({{0, 1}}, {{0, 0}, {0, 2}, {1, 0}, {1, 2}, {2, 1}});
    CHECK(board.liberties({0, 1}).size() == 1);
    CHECK(board.liberties({0, 1}).front() == Position{1, 1});
    Board illustrated_move = board;
    CHECK(illustrated_move.place({1, 1}, Cell::Black));
    CHECK(illustrated_move.group_at({0, 1}).size() == 2);
    CHECK(illustrated_move.liberties({0, 1}).empty());
    CHECK(!illustrated_move.liberties({0, 0}).empty());
    CHECK(!illustrated_move.liberties({0, 2}).empty());
    CHECK(!illustrated_move.liberties({2, 1}).empty());
    GameRules variant;
    variant.suicide_rule = SuicideRule::Forbidden;
    State forbidden(board, Cell::Black, variant);
    CHECK(forbidden.territory_owner({1, 1}) == Cell::Empty);
    const auto before = forbidden.board().cells();
    CHECK(!forbidden.is_legal(Move::place(1, 1)));
    CHECK(forbidden.play(Move::place(1, 1)).error == MoveError::Suicide);
    CHECK(forbidden.board().cells() == before);
    CHECK(forbidden.to_play() == Cell::Black);
    CHECK(!forbidden.result().finished());

    State loses(board, Cell::Black);
    CHECK(loses.is_legal(Move::place(1, 1)));
    const auto legal = loses.legal_moves();
    CHECK(std::find(legal.begin(), legal.end(), Move::place(1, 1)) != legal.end());
    const int remaining = loses.remaining_stones(Cell::Black);
    const auto outcome = loses.play(Move::place(1, 1));
    CHECK(outcome.accepted());
    CHECK(outcome.result.finished());
    CHECK(outcome.result.winner == Cell::White);
    CHECK(loses.board().at({1, 1}) == Cell::Black);
    CHECK(loses.remaining_stones(Cell::Black) == remaining - 1);
    CHECK(loses.result().finished());
    CHECK(loses.result().reason == EndReason::Suicide);
    CHECK(loses.result().winner == Cell::White);
    CHECK(loses.legal_moves().empty());
    CHECK(loses.play(Move::pass()).error == MoveError::GameOver);

    // The second player must lose as well when making the same self-surround.
    auto reversed_cells = board.cells();
    for (auto& cell : reversed_cells) {
        if (is_player(cell)) {
            cell = opponent(cell);
        }
    }
    State white_loses(Board(reversed_cells), Cell::White);
    CHECK(white_loses.is_legal(Move::place(1, 1)));
    CHECK(white_loses.play(Move::place(1, 1)).accepted());
    CHECK(white_loses.result().reason == EndReason::Suicide);
    CHECK(white_loses.result().winner == Cell::Black);
}

} // namespace

int main() {
    return run("capture", [] {
        single_stone_capture();
        connected_group_capture_and_remaining_gap();
        simultaneous_surround_prefers_mover();
        neutral_wall_and_white_capture();
        configurable_suicide();
    });
}
