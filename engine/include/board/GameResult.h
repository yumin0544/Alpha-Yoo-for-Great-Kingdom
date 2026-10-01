#pragma once

#include "board/Board.h"

namespace kingdom {

enum class EndReason { None, Capture, TwoPasses, Suicide };

struct Score {
    int black = 0;
    int white = 0;

    [[nodiscard]] Cell winner() const noexcept {
        return black - white >= 3 ? Cell::Black : Cell::White;
    }

    bool operator==(const Score&) const = default;
};

struct GameResult {
    Cell winner = Cell::Empty;
    EndReason reason = EndReason::None;
    Score score{};
    int captured_stones = 0;

    [[nodiscard]] bool finished() const noexcept {
        return reason != EndReason::None;
    }

    bool operator==(const GameResult&) const = default;
};

} // namespace kingdom
