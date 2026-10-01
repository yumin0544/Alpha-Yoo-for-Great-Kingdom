#pragma once

#include "board/Board.h"
#include "board/GameResult.h"
#include "board/Move.h"

#include <array>
#include <optional>
#include <vector>

namespace kingdom {

enum class SuicideRule { Forbidden, Loses };

// Provisional choices for the unresolved items in docs/game_rules.md.
struct GameRules {
    SuicideRule suicide_rule = SuicideRule::Forbidden;
    bool allow_own_territory_moves = true;
    bool allow_single_edge_territory = true;
    int stones_per_player = 40;
};

enum class MoveError {
    None,
    GameOver,
    OutOfBounds,
    Occupied,
    OpponentTerritory,
    OwnTerritory,
    NoStones,
    Suicide
};

struct MoveOutcome {
    MoveError error = MoveError::None;
    GameResult result{};

    [[nodiscard]] bool accepted() const noexcept {
        return error == MoveError::None;
    }
};

// Copyable value state for future MCTS. Board mutation happens through play().
class State {
public:
    explicit State(GameRules rules = {},
                   std::optional<Position> neutral = Position{4, 4});

    // Analysis/setup constructor: the caller supplies a non-terminal position.
    explicit State(Board board, Cell to_play, GameRules rules = {});

    [[nodiscard]] const Board& board() const noexcept { return board_; }
    [[nodiscard]] Cell to_play() const noexcept { return to_play_; }
    [[nodiscard]] const GameRules& rules() const noexcept { return rules_; }
    [[nodiscard]] const Board::Ownership& ownership() const noexcept {
        return ownership_;
    }
    [[nodiscard]] Cell territory_owner(Position point) const;
    [[nodiscard]] int remaining_stones(Cell player) const;
    [[nodiscard]] int consecutive_passes() const noexcept { return passes_; }
    [[nodiscard]] const GameResult& result() const noexcept { return result_; }
    [[nodiscard]] Score score() const noexcept;

    // Rejected moves preserve every field, including the pass counter.
    [[nodiscard]] MoveOutcome play(Move move);
    [[nodiscard]] bool is_legal(Move move) const;
    [[nodiscard]] std::vector<Move> legal_moves() const;

private:
    Board board_;
    GameRules rules_;
    Cell to_play_ = Cell::Black;
    Board::Ownership ownership_{};
    std::array<int, 2> placed_{};
    int passes_ = 0;
    GameResult result_{};

    void claim_territory();
    [[nodiscard]] MoveError placement_error(Position point) const;
    [[nodiscard]] static int player_index(Cell player);
};

} // namespace kingdom
