#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace kingdom {

enum class Cell : std::uint8_t { Empty, Black, White, Neutral };

constexpr bool is_player(Cell cell) noexcept {
    return cell == Cell::Black || cell == Cell::White;
}

constexpr Cell opponent(Cell player) noexcept {
    return player == Cell::Black ? Cell::White
         : player == Cell::White ? Cell::Black
                                 : Cell::Empty;
}

// All engine coordinates are zero-based; the default neutral stone is (4, 4).
struct Position {
    int row;
    int col;

    bool operator==(const Position&) const = default;
};

class Board {
public:
    static constexpr int kSize = 9;
    static constexpr int kCellCount = kSize * kSize;

    using Cells = std::array<Cell, kCellCount>;
    using Ownership = std::array<Cell, kCellCount>;

    struct Territory {
        Ownership owners{};
        int black = 0;
        int white = 0;
    };

    explicit Board(std::optional<Position> neutral = Position{4, 4});
    explicit Board(const Cells& cells);

    static bool in_bounds(Position pos) noexcept;
    static int index(Position pos);
    static Position position(int index);

    Cell at(Position pos) const;
    const Cells& cells() const noexcept;

    // Placement is a low-level board operation; State enforces game rules.
    bool place(Position pos, Cell player);
    // Neutral stones cannot be cleared. Clearing an empty cell is also invalid.
    void clear(Position pos);
    int count(Cell cell) const noexcept;

    std::vector<Position> group_at(Position pos) const;
    std::vector<Position> liberties(Position pos) const;

    // Counts empty cells only. Own stones and neutral stones act as walls.
    // Single-edge territory is allowed by default; false is an analysis variant.
    Territory territory(bool allow_single_edge = true) const;
    std::string to_string() const;

private:
    Cells cells_{};
};

} // namespace kingdom
