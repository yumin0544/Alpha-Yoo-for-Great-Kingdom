#include "board/State.h"
#include "test_support.h"

#include <algorithm>
#include <random>

using namespace kingdom;
using namespace kingdom::test;

namespace {

void board_setup_and_placement() {
    Board board;
    CHECK(Board::kSize == 9);
    CHECK(Board::kCellCount == 81);
    CHECK(board.at({4, 4}) == Cell::Neutral);
    CHECK(board.count(Cell::Empty) == 80);
    CHECK(board.count(Cell::Neutral) == 1);
    CHECK(Board::index({8, 8}) == 80);
    CHECK(Board::position(80) == Position{8, 8});
    CHECK(!Board::in_bounds({-1, 0}));
    CHECK(!Board::in_bounds({0, 9}));

    const auto initial = board.cells();
    CHECK(!board.place({-1, 0}, Cell::Black));
    CHECK(!board.place({9, 0}, Cell::White));
    CHECK(!board.place({4, 4}, Cell::Black));
    CHECK(board.cells() == initial);
    CHECK(board.place({0, 0}, Cell::Black));
    CHECK(!board.place({0, 0}, Cell::White));
    CHECK(board.at({0, 0}) == Cell::Black);
    CHECK(board.count(Cell::Black) == 1);
    CHECK(!board.to_string().empty());

    Board no_neutral(std::nullopt);
    CHECK(no_neutral.count(Cell::Empty) == 81);
    Board custom_neutral(Position{0, 8});
    CHECK(custom_neutral.at({0, 8}) == Cell::Neutral);
    CHECK(custom_neutral.at({4, 4}) == Cell::Empty);
    CHECK(Board(board.cells()).cells() == board.cells());
}

void groups_and_liberties() {
    Board board = make_board({{2, 2}, {2, 3}, {3, 2}, {3, 4}});
    CHECK(board.group_at({2, 2}).size() == 3);
    CHECK(board.group_at({3, 4}).size() == 1);
    CHECK(board.liberties({2, 2}).size() == 7);
    CHECK(board.group_at({8, 8}).empty());

    Board edges = make_board({{0, 8}, {1, 0}});
    CHECK(edges.group_at({0, 8}).size() == 1);
    CHECK(edges.group_at({1, 0}).size() == 1);
    CHECK(edges.liberties({0, 8}).size() == 2);

    Board wall = make_board({{4, 3}}, {}, Position{4, 4});
    CHECK(wall.liberties({4, 3}).size() == 3);
    const auto liberties = wall.liberties({4, 3});
    CHECK(std::find(liberties.begin(), liberties.end(), Position{4, 4}) == liberties.end());
}

void turns_passes_and_game_over() {
    State state;
    CHECK(state.to_play() == Cell::Black);
    CHECK(!state.result().finished());
    CHECK(state.remaining_stones(Cell::Black) == 41);
    CHECK(state.remaining_stones(Cell::White) == 41);
    CHECK(state.play(Move::place(4, 4)).error == MoveError::Occupied);
    CHECK(state.play(Move::place(-1, 0)).error == MoveError::OutOfBounds);
    CHECK(state.to_play() == Cell::Black);
    CHECK(state.remaining_stones(Cell::Black) == 41);

    CHECK(state.play(Move::place(0, 0)).accepted());
    CHECK(state.to_play() == Cell::White);
    CHECK(state.remaining_stones(Cell::Black) == 40);
    CHECK(state.play(Move::pass()).accepted());
    CHECK(state.consecutive_passes() == 1);
    CHECK(state.remaining_stones(Cell::White) == 41);
    CHECK(state.play(Move::place(0, 0)).error == MoveError::Occupied);
    CHECK(state.consecutive_passes() == 1);
    CHECK(state.to_play() == Cell::Black);
    CHECK(state.play(Move::place(0, 1)).accepted());
    CHECK(state.consecutive_passes() == 0);
    CHECK(state.to_play() == Cell::White);

    CHECK(state.play(Move::pass()).accepted());
    CHECK(state.play(Move::pass()).accepted());
    CHECK(state.result().finished());
    CHECK(state.result().reason == EndReason::TwoPasses);
    CHECK(state.result().winner == Cell::White);
    CHECK(state.legal_moves().empty());
    CHECK(!state.is_legal(Move::pass()));
    const auto finished_board = state.board().cells();
    CHECK(state.play(Move::place(8, 8)).error == MoveError::GameOver);
    CHECK(state.play(Move::pass()).error == MoveError::GameOver);
    CHECK(state.board().cells() == finished_board);
}

void exhausted_stocks_and_score_threshold() {
    State no_neutral(GameRules{}, std::nullopt);
    CHECK(no_neutral.board().count(Cell::Empty) == 81);
    CHECK(no_neutral.remaining_stones(Cell::Black) == 41);
    CHECK(no_neutral.remaining_stones(Cell::White) == 41);

    GameRules rules;
    rules.stones_per_player = 1;
    State state(rules);
    CHECK(state.play(Move::place(0, 0)).accepted());
    CHECK(state.play(Move::place(8, 8)).accepted());
    CHECK(state.remaining_stones(Cell::Black) == 0);
    CHECK(state.remaining_stones(Cell::White) == 0);
    CHECK(state.play(Move::place(0, 1)).error == MoveError::NoStones);
    const auto moves = state.legal_moves();
    CHECK(moves.size() == 1);
    CHECK(moves.front().is_pass());
    CHECK(state.play(Move::pass()).accepted());
    CHECK(state.play(Move::pass()).accepted());

    CHECK((Score{15, 12}.winner() == Cell::Black));
    CHECK((Score{15, 13}.winner() == Cell::White));
    CHECK((Score{10, 10}.winner() == Cell::White));

    State three_point_lead(make_board({{0, 2}, {1, 1}, {2, 0}}), Cell::Black);
    CHECK(three_point_lead.score().black == 3);
    CHECK(three_point_lead.play(Move::pass()).accepted());
    CHECK(three_point_lead.play(Move::pass()).accepted());
    CHECK(three_point_lead.result().winner == Cell::Black);
    CHECK(three_point_lead.result().score.black == 3);
    CHECK(three_point_lead.result().score.white == 0);
}

void complete_seeded_games() {
    for (unsigned seed = 0; seed < 20; ++seed) {
        std::mt19937 random(seed);
        State state;
        int turns = 0;
        while (!state.result().finished() && turns < 200) {
            const auto moves = state.legal_moves();
            CHECK(!moves.empty());
            std::uniform_int_distribution<std::size_t> choose(0, moves.size() - 1);
            const Move move = moves[choose(random)];
            const Cell actor = state.to_play();
            const int previous_black = state.remaining_stones(Cell::Black);
            const int previous_white = state.remaining_stones(Cell::White);
            CHECK(state.is_legal(move));
            CHECK(state.play(move).accepted());
            const int black = state.remaining_stones(Cell::Black);
            const int white = state.remaining_stones(Cell::White);
            CHECK(black >= 0 && black <= previous_black);
            CHECK(white >= 0 && white <= previous_white);
            CHECK(previous_black - black == (!move.is_pass() && actor == Cell::Black ? 1 : 0));
            CHECK(previous_white - white == (!move.is_pass() && actor == Cell::White ? 1 : 0));
            CHECK(state.board().at({4, 4}) == Cell::Neutral);
            CHECK(state.board().count(Cell::Neutral) == 1);
            ++turns;
        }
        CHECK(state.result().finished());
        CHECK(turns <= 200);
        CHECK(is_player(state.result().winner));
        CHECK(state.legal_moves().empty());
    }
}

} // namespace

int main() {
    return run("board", [] {
        board_setup_and_placement();
        groups_and_liberties();
        turns_passes_and_game_over();
        exhausted_stocks_and_score_threshold();
        complete_seeded_games();
    });
}
