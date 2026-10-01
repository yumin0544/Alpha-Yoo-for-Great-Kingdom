#include "board/State.h"
#include "test_support.h"

using namespace kingdom;
using namespace kingdom::test;

namespace {

Board square_enclosure(std::optional<Position> neutral = std::nullopt) {
    Board board(neutral);
    for (int coordinate = 2; coordinate <= 6; ++coordinate) {
        CHECK(board.place({2, coordinate}, Cell::Black));
        CHECK(board.place({6, coordinate}, Cell::Black));
    }
    for (int row = 3; row <= 5; ++row) {
        CHECK(board.place({row, 2}, Cell::Black));
        CHECK(board.place({row, 6}, Cell::Black));
    }
    return board;
}

void one_and_multiple_cell_houses() {
    Board one = make_board({{1, 2}, {2, 1}, {2, 3}, {3, 2}});
    const auto one_territory = one.territory();
    CHECK(one_territory.black == 1);
    CHECK(one_territory.white == 0);
    CHECK(owner_at(one_territory, {2, 2}) == Cell::Black);
    CHECK(owner_at(one_territory, {1, 1}) == Cell::Empty);

    // Five empty cells from the user's multi-cell example.
    Board five = make_board({{0, 3}, {1, 2}, {1, 4}, {2, 1}, {2, 4},
                             {3, 1}, {3, 4}, {4, 2}, {4, 3}, {4, 4}});
    const auto five_territory = five.territory();
    CHECK(five_territory.black == 5);
    CHECK(owner_at(five_territory, {1, 3}) == Cell::Black);
    CHECK(owner_at(five_territory, {3, 3}) == Cell::Black);

    Board white = make_board({}, {{1, 2}, {2, 1}, {2, 3}, {3, 2}});
    CHECK(white.territory().white == 1);
}

void edges_and_four_edge_exclusion() {
    Board single_edge = make_board({{0, 2}, {0, 5}, {1, 3}, {1, 4}});
    CHECK(single_edge.territory().black == 2);
    CHECK(single_edge.territory(false).black == 0);
    State edge_house(single_edge, Cell::Black);
    CHECK(edge_house.score().black == 2);
    CHECK(edge_house.play(Move::place(0, 3)).error == MoveError::OwnTerritory);
    CHECK(edge_house.score().black == 2);

    Board corner = make_board({{0, 2}, {1, 1}, {2, 0}});
    CHECK(corner.territory().black == 3);
    CHECK(owner_at(corner.territory(), {0, 0}) == Cell::Black);

    Board three_edges(std::nullopt);
    for (int row = 0; row < Board::kSize; ++row) {
        CHECK(three_edges.place({row, 2}, Cell::Black));
    }
    CHECK(three_edges.place({0, 3}, Cell::White));
    CHECK(three_edges.territory().black == 18);
    CHECK(owner_at(three_edges.territory(), {8, 0}) == Cell::Black);
    CHECK(owner_at(three_edges.territory(), {8, 8}) == Cell::Empty);

    CHECK(make_board({{4, 4}}).territory().black == 0);
    CHECK(Board().territory().black == 0);
    CHECK(Board().territory().white == 0);
}

void opponent_inside_and_neutral_usage() {
    Board opponent_inside = square_enclosure();
    CHECK(opponent_inside.place({4, 4}, Cell::White));
    CHECK(opponent_inside.territory().black == 0);
    CHECK(owner_at(opponent_inside.territory(), {3, 3}) == Cell::Empty);

    Board neutral_inside = square_enclosure(Position{4, 4});
    CHECK(neutral_inside.territory().black == 8);
    CHECK(owner_at(neutral_inside.territory(), {3, 3}) == Cell::Black);
    CHECK(owner_at(neutral_inside.territory(), {4, 4}) == Cell::Empty);
    CHECK(neutral_inside.count(Cell::Neutral) == 1);

    Board neutral_wall = make_board({{3, 5}, {4, 6}, {5, 5}}, {}, Position{4, 4});
    CHECK(neutral_wall.territory().black == 1);
    CHECK(owner_at(neutral_wall.territory(), {4, 5}) == Cell::Black);
}

void completed_houses_and_permanent_ownership() {
    const Board board = make_board({{1, 2}, {2, 1}, {2, 3}, {3, 2}});
    State opponent_turn(board, Cell::White);
    CHECK(opponent_turn.territory_owner({2, 2}) == Cell::Black);
    CHECK(!opponent_turn.is_legal(Move::place(2, 2)));
    CHECK(opponent_turn.play(Move::place(2, 2)).error == MoveError::OpponentTerritory);
    CHECK(opponent_turn.board().at({2, 2}) == Cell::Empty);
    CHECK(opponent_turn.to_play() == Cell::White);

    State owner_turn(board, Cell::Black);
    CHECK(owner_turn.score().black == 1);
    const auto before = owner_turn.board().cells();
    const auto claims = owner_turn.ownership();
    CHECK(!owner_turn.is_legal(Move::place(2, 2)));
    CHECK(owner_turn.play(Move::place(2, 2)).error == MoveError::OwnTerritory);
    CHECK(owner_turn.board().cells() == before);
    CHECK(owner_turn.ownership() == claims);
    CHECK(owner_turn.remaining_stones(Cell::Black) == 37);
    CHECK(owner_turn.to_play() == Cell::Black);
    CHECK(owner_turn.score().black == 1);
    CHECK(owner_turn.play(Move::place(0, 0)).accepted());
    CHECK(owner_turn.territory_owner({2, 2}) == Cell::Black);
    CHECK(owner_turn.score().black == 1);

    // Explicit non-standard analysis variant, not the confirmed game defaults.
    GameRules variant;
    variant.allow_own_territory_moves = true;
    State allowed(board, Cell::Black, variant);
    CHECK(allowed.play(Move::place(2, 2)).accepted());
    CHECK(allowed.board().at({2, 2}) == Cell::Black);
    CHECK(allowed.territory_owner({2, 2}) == Cell::Black);
    CHECK(allowed.score().black == 0);

    GameRules rules;
    rules.allow_own_territory_moves = false;
    State forbidden(board, Cell::Black, rules);
    CHECK(forbidden.play(Move::place(2, 2)).error == MoveError::OwnTerritory);
    CHECK(forbidden.score().black == 1);
}

} // namespace

int main() {
    return run("territory", [] {
        one_and_multiple_cell_houses();
        edges_and_four_edge_exclusion();
        opponent_inside_and_neutral_usage();
        completed_houses_and_permanent_ownership();
    });
}
