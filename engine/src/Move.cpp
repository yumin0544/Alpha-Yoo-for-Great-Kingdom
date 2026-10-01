#include "board/Move.h"

namespace kingdom {

Move Move::place(int row, int col) noexcept {
    return Move{Position{row, col}};
}

Move Move::pass() noexcept {
    return Move{std::nullopt};
}

bool Move::is_pass() const noexcept {
    return !point.has_value();
}

} // namespace kingdom
