#include "TacticalSolver.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace kingdom {
namespace {

using Clock = std::chrono::steady_clock;

struct Proof {
    TacticalOutcome outcome = TacticalOutcome::Unknown;
    int depth = 0;
    std::vector<Move> line;
};

TacticalOutcome invert(TacticalOutcome value) {
    return value == TacticalOutcome::Win ? TacticalOutcome::Loss
         : value == TacticalOutcome::Loss ? TacticalOutcome::Win
                                         : TacticalOutcome::Unknown;
}

int action(Move move) {
    return move.is_pass() ? Board::kCellCount : Board::index(*move.point);
}

void append_int(std::string& key, std::uint64_t value) {
    for (int i = 0; i < 8; ++i) key.push_back(static_cast<char>((value >> (i * 8)) & 255));
}

// Hash table equality checks this complete byte string, so a hash collision
// cannot turn one position's proof into another position's proof.
std::string state_key(const State& state, int depth) {
    std::string key;
    key.reserve(250);
    for (Cell cell : state.board().cells()) key.push_back(static_cast<char>(cell));
    for (Cell owner : state.ownership()) key.push_back(static_cast<char>(owner));
    key.push_back(static_cast<char>(state.to_play()));
    append_int(key, state.remaining_stones(Cell::Black));
    append_int(key, state.remaining_stones(Cell::White));
    append_int(key, state.consecutive_passes());
    const auto& rules = state.rules();
    key.push_back(static_cast<char>(rules.suicide_rule));
    key.push_back(static_cast<char>(rules.allow_own_territory_moves));
    key.push_back(static_cast<char>(rules.allow_single_edge_territory));
    append_int(key, rules.stones_per_player);
    const auto& result = state.result();
    key.push_back(static_cast<char>(result.winner));
    key.push_back(static_cast<char>(result.reason));
    append_int(key, result.score.black);
    append_int(key, result.score.white);
    append_int(key, result.captured_stones);
    append_int(key, depth);
    return key;
}

struct OrderedMoves {
    std::vector<Move> moves;
    // These are just candidate immediate wins; State::play verifies every one.
    std::vector<Move> captures;
};

OrderedMoves ordered_moves(const State& state) {
    const auto& board = state.board();
    const Cell actor = state.to_play();
    std::array<int, Board::kCellCount + 1> priority{};
    std::array<bool, Board::kCellCount> examined{};
    std::array<bool, Board::kCellCount> capture_candidate{};
    // Threats, atari escapes, contact and connections only change ordering.
    // In particular, a distant counter-capture or ladder breaker is not pruned.
    for (int i = 0; i < Board::kCellCount; ++i) {
        const Position point = Board::position(i);
        const Cell color = board.cells()[static_cast<std::size_t>(i)];
        if (!is_player(color) || examined[static_cast<std::size_t>(i)]) continue;
        const auto group = board.group_at(point);
        for (Position member : group) examined[static_cast<std::size_t>(Board::index(member))] = true;
        const auto liberties = board.liberties(point);
        for (Position liberty : liberties) {
            const auto index = static_cast<std::size_t>(Board::index(liberty));
            if (color != actor && liberties.size() == 1) {
                priority[index] += 100000;
                capture_candidate[index] = true;
            } else if (color == actor && liberties.size() == 1) {
                priority[index] += 50000;
            } else if (color != actor && liberties.size() == 2) {
                priority[index] += 20000;
                // In a ladder, extending the attack away from an existing
                // friendly wall usually keeps the chase narrow. This is only
                // a tie breaker: the other liberty and every distant defense
                // remain searchable and must be covered by a loss proof.
                int friendly_neighbors = 0;
                constexpr std::array<Position, 4> directions{{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}};
                for (Position offset : directions) {
                    Position neighbor{liberty.row + offset.row, liberty.col + offset.col};
                    if (Board::in_bounds(neighbor) && board.at(neighbor) == actor) ++friendly_neighbors;
                }
                const int edge_distance = std::min({liberty.row, liberty.col,
                    Board::kSize - 1 - liberty.row, Board::kSize - 1 - liberty.col});
                priority[index] += (Board::kSize - edge_distance) * 5 - friendly_neighbors * 50;
            } else if (color == actor && liberties.size() == 2) {
                priority[index] += 10000;
            } else if (liberties.size() <= 3) {
                priority[index] += 1000;
            } else {
                priority[index] += 10;
            }
        }
    }
    OrderedMoves result;
    if (state.remaining_stones(actor) > 0) {
        for (int i = 0; i < Board::kCellCount; ++i) {
            const Position point = Board::position(i);
            if (board.cells()[static_cast<std::size_t>(i)] != Cell::Empty) continue;
            const Cell owner = state.ownership()[static_cast<std::size_t>(i)];
            if (owner == opponent(actor) ||
                (owner == actor && !state.rules().allow_own_territory_moves)) continue;
            const Move move = Move::place(point.row, point.col);
            result.moves.push_back(move);
            if (capture_candidate[static_cast<std::size_t>(i)]) result.captures.push_back(move);
        }
    }
    result.moves.push_back(Move::pass());
    priority[Board::kCellCount] = state.consecutive_passes() == 1 && state.score().winner() == actor
                               ? 200000 : -1;
    std::stable_sort(result.moves.begin(), result.moves.end(), [&](Move lhs, Move rhs) {
        return priority[static_cast<std::size_t>(action(lhs))] >
               priority[static_cast<std::size_t>(action(rhs))];
    });
    if (state.consecutive_passes() == 1 && state.score().winner() == actor) {
        result.captures.insert(result.captures.begin(), Move::pass());
    }
    return result;
}

class Solver {
public:
    explicit Solver(TacticalSolverOptions options)
        : options_(options), start_(Clock::now()) {
        if (options.max_depth < 0 || options.max_depth > 256) {
            throw std::invalid_argument("Tactical max_depth must be between 0 and 256");
        }
        if (options.max_nodes == 0) throw std::invalid_argument("Tactical max_nodes must be positive");
    }

    TacticalSolverResult run(const State& state) {
        // The root action inventory uses the engine's public legality oracle,
        // even if the search budget expires before an action is inspected.
        const auto legal = state.legal_moves();
        std::array<TacticalOutcome, Board::kCellCount + 1> labels{};
        labels.fill(TacticalOutcome::Unknown);
        Proof proof = search(state, options_.max_depth, &labels);
        TacticalSolverResult result;
        result.outcome = proof.outcome;
        result.principal_variation = std::move(proof.line);
        result.proof_depth = proof.depth;
        result.completed_depth = exhausted_ ? 0 : options_.max_depth;
        result.nodes = nodes_;
        result.elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now() - start_).count();
        result.budget_exhausted = exhausted_;
        for (Move move : legal) {
            const auto label = labels[static_cast<std::size_t>(action(move))];
            if (label == TacticalOutcome::Win) result.winning_moves.push_back(move);
            else if (label == TacticalOutcome::Loss) result.losing_moves.push_back(move);
            else result.unknown_moves.push_back(move);
        }
        return result;
    }

private:
    TacticalSolverOptions options_;
    Clock::time_point start_;
    std::size_t nodes_ = 0;
    bool exhausted_ = false;
    std::unordered_map<std::string, Proof> memo_;

    bool consume() {
        if (nodes_ >= options_.max_nodes ||
            (options_.time_limit_ms != 0 &&
             std::chrono::duration<double, std::milli>(Clock::now() - start_).count() >=
                static_cast<double>(options_.time_limit_ms))) {
            exhausted_ = true;
            return false;
        }
        ++nodes_;
        return true;
    }

    Proof terminal(const State& state) const {
        return {state.result().winner == state.to_play() ? TacticalOutcome::Win : TacticalOutcome::Loss,
                0, {}};
    }

    Proof search(const State& state, int depth,
                 std::array<TacticalOutcome, Board::kCellCount + 1>* labels = nullptr) {
        if (!consume()) return {};
        if (state.result().finished()) return terminal(state);
        if (depth == 0) return {};
        const std::string key = state_key(state, depth);
        if (!labels) {
            const auto found = memo_.find(key);
            if (found != memo_.end()) return found->second;
        }
        const auto ordered = ordered_moves(state);
        // This scan avoids walking a huge irrelevant branch before an exact
        // immediate capture. It never infers a capture from liberty count alone.
        for (Move move : ordered.captures) {
            if (!consume()) return {};
            State child = state;
            if (!child.play(move).accepted()) continue;
            if (child.result().finished() && child.result().winner == state.to_play()) {
                Proof proof{TacticalOutcome::Win, 1, {move}};
                if (labels) (*labels)[static_cast<std::size_t>(action(move))] = TacticalOutcome::Win;
                memo_.insert_or_assign(key, proof);
                return proof;
            }
        }
        bool has_unknown = false;
        int worst_depth = 0;
        std::vector<Move> worst_line;
        std::size_t legal_count = 0;
        for (Move move : ordered.moves) {
            if (exhausted_) return {};
            State child = state;
            if (!child.play(move).accepted()) continue;
            ++legal_count;
            Proof reply = search(child, depth - 1);
            const TacticalOutcome outcome = invert(reply.outcome);
            if (labels) (*labels)[static_cast<std::size_t>(action(move))] = outcome;
            if (outcome == TacticalOutcome::Win) {
                reply.outcome = TacticalOutcome::Win;
                ++reply.depth;
                reply.line.insert(reply.line.begin(), move);
                memo_.insert_or_assign(key, reply);
                return reply;
            }
            if (outcome == TacticalOutcome::Unknown) has_unknown = true;
            else if (reply.depth + 1 > worst_depth) {
                worst_depth = reply.depth + 1;
                worst_line = std::move(reply.line);
                worst_line.insert(worst_line.begin(), move);
            }
        }
        if (exhausted_ || has_unknown || legal_count == 0) return {};
        Proof proof{TacticalOutcome::Loss, worst_depth, std::move(worst_line)};
        memo_.insert_or_assign(key, proof);
        return proof;
    }
};

} // namespace

TacticalSolverResult solve_tactics(const State& state, TacticalSolverOptions options) {
    return Solver(options).run(state);
}

} // namespace kingdom
