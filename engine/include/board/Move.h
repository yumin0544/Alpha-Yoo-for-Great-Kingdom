#pragma once

#include "board/Board.h"

#include <optional>

namespace kingdom {

// Public engine coordinates are zero-based. No point represents a pass.
struct Move {
    std::optional<Position> point;

    [[nodiscard]] static Move place(int row, int col) noexcept;
    [[nodiscard]] static Move pass() noexcept;
    [[nodiscard]] bool is_pass() const noexcept;
    bool operator==(const Move&) const = default;
};

} // namespace kingdom
