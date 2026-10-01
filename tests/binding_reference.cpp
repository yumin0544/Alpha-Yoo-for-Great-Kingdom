#include "MCTS.h"
#include "board/State.h"

#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string_view>

using namespace kingdom;

namespace {

Board make_board(std::initializer_list<Position> black,
                 std::initializer_list<Position> white = {}) {
    Board board(std::nullopt);
    for (const auto point : black) {
        if (!board.place(point, Cell::Black)) {
            throw std::logic_error("Invalid black reference fixture");
        }
    }
    for (const auto point : white) {
        if (!board.place(point, Cell::White)) {
            throw std::logic_error("Invalid white reference fixture");
        }
    }
    return board;
}

void json_string(std::string_view value) {
    std::cout << '"';
    for (const char character : value) {
        switch (character) {
        case '"': std::cout << "\\\""; break;
        case '\\': std::cout << "\\\\"; break;
        case '\n': std::cout << "\\n"; break;
        case '\r': std::cout << "\\r"; break;
        case '\t': std::cout << "\\t"; break;
        default: std::cout << character; break;
        }
    }
    std::cout << '"';
}

void emit_move(const Move& move) {
    if (move.is_pass()) {
        json_string("pass");
    } else {
        std::cout << '[' << move.point->row << ',' << move.point->col << ']';
    }
}

void emit_state(const State& state) {
    const auto score = state.score();
    const auto result = state.result();
    std::cout << "{\"board\":";
    json_string(state.board().to_string());
    std::cout << ",\"to_play\":" << static_cast<int>(state.to_play())
              << ",\"passes\":" << state.consecutive_passes()
              << ",\"remaining\":[" << state.remaining_stones(Cell::Black)
              << ',' << state.remaining_stones(Cell::White)
              << "],\"score\":[" << score.black << ',' << score.white
              << "],\"owners\":[";
    bool separator = false;
    for (const auto owner : state.ownership()) {
        if (separator) {
            std::cout << ',';
        }
        std::cout << static_cast<int>(owner);
        separator = true;
    }
    std::cout << "],\"result\":{\"winner\":" << static_cast<int>(result.winner)
              << ",\"reason\":" << static_cast<int>(result.reason)
              << ",\"score\":[" << result.score.black << ',' << result.score.white
              << "],\"captured\":" << result.captured_stones
              << ",\"finished\":" << result.finished() << "}}";
}

void emit_case(std::string_view name, State state, std::initializer_list<Move> moves) {
    std::cout << "{\"name\":";
    json_string(name);
    std::cout << ",\"errors\":[";
    bool separator = false;
    for (const auto move : moves) {
        if (separator) {
            std::cout << ',';
        }
        std::cout << static_cast<int>(state.play(move).error);
        separator = true;
    }
    std::cout << "],\"state\":";
    emit_state(state);
    std::cout << '}';
}

void emit_search(const SearchResult& result) {
    std::cout << "{\"best_move\":";
    if (result.best_move) {
        emit_move(*result.best_move);
    } else {
        std::cout << "null";
    }
    std::cout << ",\"simulations\":" << result.simulations
              << ",\"nodes\":" << result.nodes
              << ",\"total_rollout_plies\":" << result.total_rollout_plies
              << ",\"win_rate\":" << result.win_rate << ",\"moves\":[";
    bool separator = false;
    for (const auto& statistics : result.moves) {
        if (separator) {
            std::cout << ',';
        }
        std::cout << "{\"move\":";
        emit_move(statistics.move);
        std::cout << ",\"visits\":" << statistics.visits
                  << ",\"win_rate\":" << statistics.win_rate << '}';
        separator = true;
    }
    std::cout << "]}";
}

} // namespace

int main() {
    try {
        std::cout << std::boolalpha << std::setprecision(17) << "{\"cases\":[";
        emit_case("initial", State{}, {});
        std::cout << ',';
        emit_case("opening", State{}, {Move::place(0, 0), Move::pass(),
                  Move::place(4, 4), Move::place(-1, 0), Move::place(0, 1)});
        std::cout << ',';
        emit_case("capture", State(make_board({{1, 2}, {2, 1}, {3, 2}}, {{2, 2}}),
                  Cell::Black), {Move::place(2, 3), Move::pass()});
        std::cout << ',';
        const auto suicide_board = make_board({{0, 1}},
            {{0, 0}, {0, 2}, {1, 0}, {1, 2}, {2, 1}});
        emit_case("suicide", State(suicide_board, Cell::Black), {Move::place(1, 1)});
        std::cout << ',';
        GameRules forbidden;
        forbidden.suicide_rule = SuicideRule::Forbidden;
        emit_case("forbidden_suicide", State(suicide_board, Cell::Black, forbidden),
                  {Move::place(1, 1), Move::pass()});
        std::cout << ',';
        emit_case("simultaneous", State(make_board({{0, 6}, {1, 6}, {1, 8}, {2, 7}},
                  {{0, 7}, {1, 7}, {2, 8}}), Cell::Black), {Move::place(0, 8)});
        std::cout << ',';
        const auto house = make_board({{1, 2}, {2, 1}, {2, 3}, {3, 2}});
        emit_case("own_house", State(house, Cell::Black), {Move::place(2, 2)});
        std::cout << ',';
        emit_case("opponent_house", State(house, Cell::White), {Move::place(2, 2)});
        std::cout << ',';
        emit_case("two_passes", State{}, {Move::pass(), Move::pass()});
        std::cout << ',';
        GameRules limited;
        limited.stones_per_player = 1;
        emit_case("stock", State(limited), {Move::place(0, 0), Move::place(8, 8),
                  Move::place(0, 1), Move::pass(), Move::pass()});
        std::cout << "],\"search\":";
        State searched;
        if (!searched.play(Move::place(0, 0)).accepted() ||
            !searched.play(Move::pass()).accepted()) {
            throw std::logic_error("Invalid reference search opening");
        }
        MCTSOptions options;
        options.simulations = 192;
        options.exploration = 1.25;
        options.seed = 99113;
        MCTS search(options);
        emit_search(search.search(searched));
        std::cout << "}\n";
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
