#include "MCTS.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace kingdom {
namespace {

int move_index(const Move& move) {
    return move.is_pass() ? Board::kCellCount : Board::index(*move.point);
}

} // namespace

MCTS::MCTS(MCTSOptions options) : options_(options), random_(options.seed) {
    if (options_.simulations == 0) {
        throw std::invalid_argument("MCTS needs at least one simulation");
    }
    if (!std::isfinite(options_.exploration) || options_.exploration < 0.0) {
        throw std::invalid_argument("Exploration must be finite and non-negative");
    }
}

std::vector<Move> MCTS::candidates(const State& state) {
    std::vector<Move> moves;
    if (state.result().finished()) return moves;

    moves.reserve(Board::kCellCount + 1);
    const Cell actor = state.to_play();
    if (state.remaining_stones(actor) > 0) {
        const auto& cells = state.board().cells();
        const auto& owners = state.ownership();
        for (int i = 0; i < Board::kCellCount; ++i) {
            const auto index = static_cast<std::size_t>(i);
            if (cells[index] != Cell::Empty || owners[index] == opponent(actor) ||
                (owners[index] == actor && !state.rules().allow_own_territory_moves)) {
                continue;
            }
            const Position point = Board::position(i);
            moves.push_back(Move::place(point.row, point.col));
        }
    }
    moves.push_back(Move::pass());
    // These are exact candidates for the default suicide-allowed rules. In the
    // Forbidden variant they are a superset; State::play rejects those suicides.
    return moves;
}

Move MCTS::take_random(std::vector<Move>& moves) {
    if (moves.empty()) throw std::logic_error("Cannot sample an empty move list");
    std::uniform_int_distribution<std::size_t> choose(0, moves.size() - 1);
    const std::size_t index = choose(random_);
    const Move move = moves[index];
    moves[index] = moves.back();
    moves.pop_back();
    return move;
}

std::size_t MCTS::select_child(const std::vector<Node>& tree, std::size_t parent) {
    const auto& node = tree[parent];
    if (node.children.empty()) throw std::logic_error("Selection needs a child");
    const double parent_log = std::log(static_cast<double>(std::max<std::size_t>(1, node.visits)));
    double best_value = -std::numeric_limits<double>::infinity();
    std::size_t best = node.children.front();
    std::size_t ties = 0;
    for (const std::size_t index : node.children) {
        const auto& child = tree[index];
        const double value = child.visits == 0
            ? std::numeric_limits<double>::infinity()
            : child.win_rate() + options_.exploration *
                  std::sqrt(parent_log / static_cast<double>(child.visits));
        if (value > best_value) {
            best_value = value;
            best = index;
            ties = 1;
        } else if (value == best_value) {
            ++ties;
            std::uniform_int_distribution<std::size_t> choose(1, ties);
            if (choose(random_) == 1) best = index;
        }
    }
    return best;
}

Cell MCTS::rollout(State& state, std::size_t& total_plies) {
    // No stones are removed before a terminal capture. There can be at most one
    // non-terminal pass between placements, followed by two final passes.
    const int max_plies = 2 * state.board().count(Cell::Empty) + 2;
    int plies = 0;
    while (!state.result().finished()) {
        if (plies >= max_plies) {
            throw std::logic_error("Rollout exceeded the game's finite turn bound");
        }
        auto moves = candidates(state);
        bool accepted = false;
        while (!moves.empty()) {
            const Move move = take_random(moves);
            if (state.play(move).accepted()) {
                accepted = true;
                break;
            }
        }
        if (!accepted) throw std::logic_error("Non-terminal state must permit a pass");
        ++plies;
        ++total_plies;
    }
    return state.result().winner;
}

SearchResult MCTS::search(const State& state) {
    using Clock = std::chrono::steady_clock;
    const auto started = Clock::now();
    SearchResult result;
    if (state.result().finished()) return result;

    std::vector<Node> tree;
    // Reserve a bounded initial arena; indices remain valid if the vector grows.
    tree.reserve(std::min<std::size_t>(options_.simulations, 4095) + 1);
    tree.emplace_back(Node::kNoParent, Move::pass(), state.to_play(), candidates(state), false);

    for (std::size_t iteration = 0; iteration < options_.simulations; ++iteration) {
        State simulation = state;
        std::size_t node_index = 0;
        while (!simulation.result().finished()) {
            bool expanded = false;
            while (!tree[node_index].untried_moves.empty()) {
                const Move move = take_random(tree[node_index].untried_moves);
                const Cell actor = simulation.to_play();
                if (!simulation.play(move).accepted()) continue;

                const std::size_t child_index = tree.size();
                const bool terminal = simulation.result().finished();
                auto moves = candidates(simulation);
                tree.emplace_back(node_index, move, actor, std::move(moves), terminal);
                // Reacquire the parent by index after a possible arena reallocation.
                tree[node_index].children.push_back(child_index);
                node_index = child_index;
                expanded = true;
                break;
            }
            if (expanded) break;

            const std::size_t child = select_child(tree, node_index);
            if (!simulation.play(tree[child].move).accepted()) {
                throw std::logic_error("A stored tree edge must remain legal");
            }
            node_index = child;
        }

        const Cell winner = rollout(simulation, result.total_rollout_plies);
        for (std::size_t index = node_index; index != Node::kNoParent; index = tree[index].parent) {
            tree[index].record_result(winner);
        }
        ++result.simulations;

        if (options_.time_limit_ms != 0) {
            const double elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now() - started).count();
            if (elapsed_ms >= static_cast<double>(options_.time_limit_ms)) break;
        }
    }

    result.nodes = tree.size();
    result.moves.reserve(tree[0].children.size());
    for (const std::size_t child : tree[0].children) {
        const auto& node = tree[child];
        result.moves.push_back({node.move, node.visits, node.win_rate()});
    }
    std::sort(result.moves.begin(), result.moves.end(), [](const auto& lhs, const auto& rhs) {
        return move_index(lhs.move) < move_index(rhs.move);
    });
    const MoveStatistics* best = nullptr;
    for (const auto& move : result.moves) {
        if (best == nullptr || move.visits > best->visits ||
            (move.visits == best->visits && move.win_rate > best->win_rate)) {
            best = &move;
        }
    }
    if (best == nullptr) throw std::logic_error("A completed search must visit a root move");
    result.best_move = best->move;
    result.win_rate = best->win_rate;
    result.elapsed_seconds = std::chrono::duration<double>(Clock::now() - started).count();
    return result;
}

} // namespace kingdom
