#include "board/State.h"

#include <stdexcept>
#include <utility>

namespace kingdom {

State::State(GameRules rules, std::optional<Position> neutral)
    : State(Board(neutral), Cell::Black, rules) {}

State::State(Board board, Cell to_play, GameRules rules)
    : board_(std::move(board)), rules_(rules), to_play_(to_play) {
    if (!is_player(to_play_)) {
        throw std::invalid_argument("The current player must be Black or White");
    }
    if (rules_.stones_per_player < 1 ||
        rules_.stones_per_player > Board::kCellCount) {
        throw std::invalid_argument("Stone supply must be between 1 and 81");
    }
    if (rules_.suicide_rule != SuicideRule::Forbidden &&
        rules_.suicide_rule != SuicideRule::Loses) {
        throw std::invalid_argument("Unknown suicide rule");
    }
    placed_ = {board_.count(Cell::Black), board_.count(Cell::White)};
    if (placed_[0] > rules_.stones_per_player ||
        placed_[1] > rules_.stones_per_player) {
        throw std::invalid_argument("Position exceeds the configured stone supply");
    }
    claim_territory();
}

int State::player_index(Cell player) {
    if (!is_player(player)) {
        throw std::invalid_argument("Expected Black or White");
    }
    return player == Cell::Black ? 0 : 1;
}

Cell State::territory_owner(Position point) const {
    return ownership_[static_cast<std::size_t>(Board::index(point))];
}

int State::remaining_stones(Cell player) const {
    return rules_.stones_per_player -
           placed_[static_cast<std::size_t>(player_index(player))];
}

Score State::score() const noexcept {
    Score value;
    for (std::size_t i = 0; i < ownership_.size(); ++i) {
        if (board_.cells()[i] != Cell::Empty) {
            continue;
        }
        if (ownership_[i] == Cell::Black) {
            ++value.black;
        } else if (ownership_[i] == Cell::White) {
            ++value.white;
        }
    }
    return value;
}

void State::claim_territory() {
    const auto detected = board_.territory(rules_.allow_single_edge_territory);
    for (std::size_t i = 0; i < ownership_.size(); ++i) {
        if (ownership_[i] == Cell::Empty && is_player(detected.owners[i])) {
            ownership_[i] = detected.owners[i];
        }
    }
}

MoveError State::placement_error(Position point) const {
    if (!Board::in_bounds(point)) {
        return MoveError::OutOfBounds;
    }
    if (board_.at(point) != Cell::Empty) {
        return MoveError::Occupied;
    }
    const Cell owner = territory_owner(point);
    if (owner == opponent(to_play_)) {
        return MoveError::OpponentTerritory;
    }
    if (owner == to_play_ && !rules_.allow_own_territory_moves) {
        return MoveError::OwnTerritory;
    }
    if (remaining_stones(to_play_) == 0) {
        return MoveError::NoStones;
    }
    return MoveError::None;
}

MoveOutcome State::play(Move move) {
    if (result_.finished()) {
        return {MoveError::GameOver, result_};
    }
    const Cell actor = to_play_;
    if (move.is_pass()) {
        ++passes_;
        to_play_ = opponent(actor);
        if (passes_ == 2) {
            const Score final_score = score();
            result_ = {final_score.winner(), EndReason::TwoPasses, final_score, 0};
        }
        return {MoveError::None, result_};
    }

    const Position point = *move.point;
    const MoveError error = placement_error(point);
    if (error != MoveError::None) {
        return {error, result_};
    }

    // Stage the placement so a rejected suicide cannot mutate the live state.
    Board candidate = board_;
    if (!candidate.place(point, actor)) {
        throw std::logic_error("Validated placement unexpectedly failed");
    }
    std::array<bool, Board::kCellCount> examined{};
    std::array<bool, Board::kCellCount> captured{};
    int captured_count = 0;
    constexpr std::array<Position, 4> directions{{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}};
    for (Position offset : directions) {
        const Position neighbor{point.row + offset.row, point.col + offset.col};
        if (!Board::in_bounds(neighbor) || candidate.at(neighbor) != opponent(actor)) {
            continue;
        }
        const auto neighbor_index = static_cast<std::size_t>(Board::index(neighbor));
        if (examined[neighbor_index]) {
            continue;
        }
        const auto group = candidate.group_at(neighbor);
        const bool surrounded = candidate.liberties(neighbor).empty();
        for (Position member : group) {
            const auto i = static_cast<std::size_t>(Board::index(member));
            examined[i] = true;
            if (surrounded) {
                captured[i] = true;
                ++captured_count;
            }
        }
    }

    // Opponent capture takes priority even if the new friendly group has no liberty.
    const bool suicide = captured_count == 0 && candidate.liberties(point).empty();
    if (suicide && rules_.suicide_rule == SuicideRule::Forbidden) {
        return {MoveError::Suicide, result_};
    }

    board_ = std::move(candidate);
    ++placed_[static_cast<std::size_t>(player_index(actor))];
    passes_ = 0;
    to_play_ = opponent(actor);
    if (captured_count > 0) {
        for (int i = 0; i < Board::kCellCount; ++i) {
            if (captured[static_cast<std::size_t>(i)]) {
                board_.clear(Board::position(i));
            }
        }
        // A terminal capture is not a new territory-scoring turn.
        result_ = {actor, EndReason::Capture, score(), captured_count};
    } else if (suicide) {
        result_ = {opponent(actor), EndReason::Suicide, score(), 0};
    } else {
        claim_territory();
    }
    return {MoveError::None, result_};
}

bool State::is_legal(Move move) const {
    if (result_.finished()) {
        return false;
    }
    if (move.is_pass()) {
        return true;
    }
    if (placement_error(*move.point) != MoveError::None) {
        return false;
    }
    State preview = *this;
    return preview.play(move).accepted();
}

std::vector<Move> State::legal_moves() const {
    std::vector<Move> moves;
    if (result_.finished()) {
        return moves;
    }
    moves.reserve(Board::kCellCount + 1);
    for (int i = 0; i < Board::kCellCount; ++i) {
        const Position point = Board::position(i);
        const Move move = Move::place(point.row, point.col);
        if (is_legal(move)) {
            moves.push_back(move);
        }
    }
    moves.push_back(Move::pass());
    return moves;
}

} // namespace kingdom
