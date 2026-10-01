#include "board/Board.h"

#include <algorithm>
#include <stdexcept>

namespace kingdom {
namespace {

// Precomputed orthogonal neighbours avoid coordinate work in flood fills.
struct Neighbours {
    std::array<int, 4> indices{};
    int count = 0;
};

constexpr auto make_neighbours() {
    std::array<Neighbours, Board::kCellCount> result{};
    for (int i = 0; i < Board::kCellCount; ++i) {
        const int row = i / Board::kSize;
        const int col = i % Board::kSize;
        auto& neighbours = result[i];
        if (row > 0) neighbours.indices[neighbours.count++] = i - Board::kSize;
        if (row + 1 < Board::kSize) neighbours.indices[neighbours.count++] = i + Board::kSize;
        if (col > 0) neighbours.indices[neighbours.count++] = i - 1;
        if (col + 1 < Board::kSize) neighbours.indices[neighbours.count++] = i + 1;
    }
    return result;
}

constexpr auto kNeighbours = make_neighbours();

constexpr unsigned edge_mask(int index) noexcept {
    const int row = index / Board::kSize;
    const int col = index % Board::kSize;
    return (row == 0 ? 1U : 0U)
         | (row == Board::kSize - 1 ? 2U : 0U)
         | (col == 0 ? 4U : 0U)
         | (col == Board::kSize - 1 ? 8U : 0U);
}

// The queue holds each connected stone at most once.
int collect_group(const Board::Cells& cells, int start,
                  std::array<int, Board::kCellCount>& queue) {
    const Cell player = cells[start];
    if (!is_player(player)) return 0;

    std::array<bool, Board::kCellCount> visited{};
    int length = 1;
    queue[0] = start;
    visited[start] = true;
    for (int cursor = 0; cursor < length; ++cursor) {
        const auto& neighbours = kNeighbours[queue[cursor]];
        for (int n = 0; n < neighbours.count; ++n) {
            const int next = neighbours.indices[n];
            if (!visited[next] && cells[next] == player) {
                visited[next] = true;
                queue[length++] = next;
            }
        }
    }
    return length;
}

} // namespace

Board::Board(std::optional<Position> neutral) {
    if (neutral) cells_[index(*neutral)] = Cell::Neutral;
}

Board::Board(const Cells& cells) : cells_(cells) {
    for (Cell cell : cells_) {
        if (cell != Cell::Empty && cell != Cell::Black &&
            cell != Cell::White && cell != Cell::Neutral) {
            throw std::invalid_argument("Board contains an invalid cell value");
        }
    }
}

bool Board::in_bounds(Position pos) noexcept {
    return pos.row >= 0 && pos.row < kSize && pos.col >= 0 && pos.col < kSize;
}

int Board::index(Position pos) {
    if (!in_bounds(pos)) throw std::out_of_range("Board position is out of bounds");
    return pos.row * kSize + pos.col;
}

Position Board::position(int cell_index) {
    if (cell_index < 0 || cell_index >= kCellCount) {
        throw std::out_of_range("Board index is out of bounds");
    }
    return {cell_index / kSize, cell_index % kSize};
}

Cell Board::at(Position pos) const {
    return cells_[index(pos)];
}

const Board::Cells& Board::cells() const noexcept {
    return cells_;
}

bool Board::place(Position pos, Cell player) {
    if (!in_bounds(pos) || !is_player(player)) return false;
    auto& cell = cells_[pos.row * kSize + pos.col];
    if (cell != Cell::Empty) return false;
    cell = player;
    return true;
}

void Board::clear(Position pos) {
    auto& cell = cells_[index(pos)];
    if (!is_player(cell)) {
        throw std::invalid_argument("Only player stones can be cleared");
    }
    cell = Cell::Empty;
}

int Board::count(Cell cell) const noexcept {
    return static_cast<int>(std::count(cells_.begin(), cells_.end(), cell));
}

std::vector<Position> Board::group_at(Position pos) const {
    std::array<int, kCellCount> queue{};
    const int length = collect_group(cells_, index(pos), queue);
    std::vector<Position> result;
    result.reserve(length);
    for (int i = 0; i < length; ++i) result.push_back(position(queue[i]));
    return result;
}

std::vector<Position> Board::liberties(Position pos) const {
    std::array<int, kCellCount> queue{};
    const int length = collect_group(cells_, index(pos), queue);
    std::array<bool, kCellCount> seen{};
    std::vector<Position> result;
    for (int i = 0; i < length; ++i) {
        const auto& neighbours = kNeighbours[queue[i]];
        for (int n = 0; n < neighbours.count; ++n) {
            const int next = neighbours.indices[n];
            if (cells_[next] == Cell::Empty && !seen[next]) {
                seen[next] = true;
                result.push_back(position(next));
            }
        }
    }
    return result;
}

Board::Territory Board::territory(bool allow_single_edge) const {
    Territory result;
    for (Cell player : {Cell::Black, Cell::White}) {
        std::array<bool, kCellCount> visited{};
        std::array<int, kCellCount> queue{};
        const Cell enemy = opponent(player);

        for (int start = 0; start < kCellCount; ++start) {
            if (visited[start] || cells_[start] != Cell::Empty) continue;

            int length = 1;
            queue[0] = start;
            visited[start] = true;
            bool contains_enemy = false;
            bool has_own_boundary = false;
            unsigned edges = 0;

            for (int cursor = 0; cursor < length; ++cursor) {
                const int current = queue[cursor];
                edges |= edge_mask(current);
                contains_enemy |= cells_[current] == enemy;
                const auto& neighbours = kNeighbours[current];
                for (int n = 0; n < neighbours.count; ++n) {
                    const int next = neighbours.indices[n];
                    if (cells_[next] == player) {
                        has_own_boundary = true;
                    } else if (cells_[next] != Cell::Neutral && !visited[next]) {
                        visited[next] = true;
                        queue[length++] = next;
                    }
                }
            }

            const bool single_edge = edges != 0 && (edges & (edges - 1U)) == 0;
            if (contains_enemy || !has_own_boundary || edges == 15U ||
                (!allow_single_edge && single_edge)) {
                continue;
            }
            for (int i = 0; i < length; ++i) {
                const int cell_index = queue[i];
                if (cells_[cell_index] == Cell::Empty) {
                    result.owners[cell_index] = player;
                    if (player == Cell::Black) ++result.black;
                    else ++result.white;
                }
            }
        }
    }
    return result;
}

std::string Board::to_string() const {
    std::string result;
    result.reserve(kCellCount + kSize - 1);
    for (int row = 0; row < kSize; ++row) {
        if (row != 0) result.push_back('\n');
        for (int col = 0; col < kSize; ++col) {
            switch (cells_[row * kSize + col]) {
                case Cell::Empty: result.push_back('.'); break;
                case Cell::Black: result.push_back('x'); break;
                case Cell::White: result.push_back('o'); break;
                case Cell::Neutral: result.push_back('v'); break;
            }
        }
    }
    return result;
}

} // namespace kingdom
