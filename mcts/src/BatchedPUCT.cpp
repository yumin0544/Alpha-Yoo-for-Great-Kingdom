#include "BatchedPUCT.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace kingdom {
namespace {

int action_index(const Move& move) {
    return move.is_pass() ? Board::kCellCount : Board::index(*move.point);
}

void validate_rules(const State& state) {
    const auto& rules = state.rules();
    if (rules.suicide_rule != SuicideRule::Loses || rules.allow_own_territory_moves ||
        !rules.allow_single_edge_territory || rules.stones_per_player != 41) {
        throw std::invalid_argument("Encoding v1 supports the confirmed default GameRules only");
    }
}

EncodedPUCTPosition encode(const State& state, const std::vector<Move>& moves) {
    EncodedPUCTPosition result;
    const Cell actor = state.to_play();
    const Cell enemy = opponent(actor);
    for (const auto& move : moves) {
        result.legal_mask[static_cast<std::size_t>(action_index(move))] = true;
    }
    const auto& cells = state.board().cells();
    const auto& owners = state.ownership();
    const float own_remaining = static_cast<float>(state.remaining_stones(actor) / 41.0);
    const float enemy_remaining = static_cast<float>(state.remaining_stones(enemy) / 41.0);
    const float passes = static_cast<float>(state.consecutive_passes() / 2.0);
    for (std::size_t i = 0; i < Board::kCellCount; ++i) {
        result.features[i] = cells[i] == actor ? 1.0F : 0.0F;
        result.features[81 + i] = cells[i] == enemy ? 1.0F : 0.0F;
        result.features[162 + i] = cells[i] == Cell::Neutral ? 1.0F : 0.0F;
        result.features[243 + i] = owners[i] == actor ? 1.0F : 0.0F;
        result.features[324 + i] = owners[i] == enemy ? 1.0F : 0.0F;
        result.features[405 + i] = actor == Cell::Black ? 1.0F : 0.0F;
        result.features[486 + i] = own_remaining;
        result.features[567 + i] = enemy_remaining;
        result.features[648 + i] = passes;
        result.features[729 + i] = result.legal_mask[i] ? 1.0F : 0.0F;
    }
    return result;
}

bool same_state(const State& lhs, const State& rhs) {
    const auto& a = lhs.rules();
    const auto& b = rhs.rules();
    return lhs.board().cells() == rhs.board().cells() &&
        lhs.ownership() == rhs.ownership() && lhs.to_play() == rhs.to_play() &&
        lhs.remaining_stones(Cell::Black) == rhs.remaining_stones(Cell::Black) &&
        lhs.remaining_stones(Cell::White) == rhs.remaining_stones(Cell::White) &&
        lhs.consecutive_passes() == rhs.consecutive_passes() && lhs.result() == rhs.result() &&
        a.suicide_rule == b.suicide_rule &&
        a.allow_own_territory_moves == b.allow_own_territory_moves &&
        a.allow_single_edge_territory == b.allow_single_edge_territory &&
        a.stones_per_player == b.stones_per_player;
}

double terminal_value(const State& state) {
    if (!is_player(state.result().winner)) {
        throw std::logic_error("A terminal PUCT position must have a winning player");
    }
    return state.result().winner == state.to_play() ? 1.0 : -1.0;
}

void validate_evaluation(const PUCTEvaluation& evaluation) {
    if (!std::isfinite(evaluation.value) || evaluation.value < -1.0 || evaluation.value > 1.0) {
        throw std::invalid_argument("PUCT evaluator value must be finite and in [-1, 1]");
    }
    for (const double weight : evaluation.policy) {
        if (!std::isfinite(weight) || weight < 0.0) {
            throw std::invalid_argument("PUCT policy weights must be finite and non-negative");
        }
    }
}

void checked_increment(std::size_t& value) {
    if (value == std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("Batched PUCT visit counter overflow");
    }
    ++value;
}

struct Edge {
    Move move;
    double prior;
    std::size_t visits = 0;
    double value_sum = 0.0;
    std::size_t reserved = 0;
    std::optional<std::size_t> child;
};

struct Node {
    State state;
    std::size_t visits = 0;
    std::size_t reserved = 0;
    bool expanded = false;
    bool pending = false;
    std::vector<Edge> edges;
};

struct PathEdge {
    std::size_t node;
    std::size_t edge;
};

struct PendingLeaf {
    std::size_t node;
    std::vector<PathEdge> path;
    std::vector<Move> moves;
};

bool better_move(const PUCTMoveStatistics& candidate, const PUCTMoveStatistics& incumbent) {
    if (candidate.visits != incumbent.visits) return candidate.visits > incumbent.visits;
    if (candidate.value != incumbent.value) return candidate.value > incumbent.value;
    if (candidate.prior != incumbent.prior) return candidate.prior > incumbent.prior;
    return action_index(candidate.move) < action_index(incumbent.move);
}

} // namespace

EncodedPUCTPosition encode_puct_position(const State& state) {
    validate_rules(state);
    return encode(state, state.legal_moves());
}

struct BatchedPUCT::Impl {
    PUCTOptions options;
    std::size_t leaf_batch_size;
    bool reuse;
    std::vector<Node> tree;
    BatchedPUCTStatistics stats;

    // Exclude pending leaves. If a whole child subtree is temporarily blocked,
    // try the next-scored edge instead of stalling the complete batch.
    bool select_leaf(std::size_t node_index, std::vector<PathEdge>& path,
                     std::size_t& leaf_index) {
        if (tree[node_index].pending) return false;
        if (!tree[node_index].expanded || tree[node_index].state.result().finished()) {
            leaf_index = node_index;
            return true;
        }
        const std::size_t count = tree[node_index].edges.size();
        if (count == 0) throw std::logic_error("PUCT selection needs a legal edge");
        std::vector<bool> attempted(count, false);
        for (std::size_t attempt = 0; attempt < count; ++attempt) {
            const auto& node = tree[node_index];
            const double parent_scale = std::sqrt(
                static_cast<double>(node.visits) + static_cast<double>(node.reserved));
            std::optional<std::size_t> best;
            double best_score = -std::numeric_limits<double>::infinity();
            for (std::size_t i = 0; i < count; ++i) {
                if (attempted[i]) continue;
                const auto& edge = node.edges[i];
                const double effective_visits = static_cast<double>(edge.visits) +
                    static_cast<double>(edge.reserved);
                // A virtual loss of one both widens selection and discourages
                // revisiting a path whose value has not arrived yet.
                const double value = effective_visits == 0.0 ? 0.0 :
                    (edge.value_sum - static_cast<double>(edge.reserved)) / effective_visits;
                const double exploration = edge.prior * parent_scale / (1.0 + effective_visits);
                const double score = options.c_puct >= 1.0
                    ? value / options.c_puct + exploration : value + options.c_puct * exploration;
                if (!best || score > best_score || (score == best_score &&
                    (edge.prior > node.edges[*best].prior ||
                     (edge.prior == node.edges[*best].prior &&
                      action_index(edge.move) < action_index(node.edges[*best].move))))) {
                    best = i;
                    best_score = score;
                }
            }
            if (!best) return false;
            attempted[*best] = true;
            auto child = tree[node_index].edges[*best].child;
            if (!child) {
                State child_state = tree[node_index].state;
                if (!child_state.play(tree[node_index].edges[*best].move).accepted()) {
                    throw std::logic_error("A stored PUCT edge must remain legal");
                }
                child = tree.size();
                tree.push_back({std::move(child_state), 0, 0, false, false, {}});
                tree[node_index].edges[*best].child = child;
            }
            path.push_back({node_index, *best});
            if (select_leaf(*child, path, leaf_index)) return true;
            path.pop_back();
        }
        return false;
    }

    void reserve(const PendingLeaf& pending) {
        tree[pending.node].pending = true;
        checked_increment(tree[0].reserved);
        for (const auto& step : pending.path) {
            auto& edge = tree[step.node].edges[step.edge];
            checked_increment(edge.reserved);
            checked_increment(tree[*edge.child].reserved);
        }
    }

    void release(const PendingLeaf& pending) {
        tree[pending.node].pending = false;
        --tree[0].reserved;
        for (const auto& step : pending.path) {
            auto& edge = tree[step.node].edges[step.edge];
            --edge.reserved;
            --tree[*edge.child].reserved;
        }
    }

    void backup(std::size_t leaf, const std::vector<PathEdge>& path, double value) {
        checked_increment(tree[0].visits);
        const Cell player = tree[leaf].state.to_play();
        for (const auto& step : path) {
            auto& edge = tree[step.node].edges[step.edge];
            edge.value_sum += tree[step.node].state.to_play() == player ? value : -value;
            checked_increment(edge.visits);
            checked_increment(tree[*edge.child].visits);
        }
    }

    void expand(std::size_t node_index, const std::vector<Move>& moves,
                const PUCTEvaluation& evaluation) {
        if (moves.empty()) throw std::logic_error("A non-terminal PUCT state must permit a pass");
        double largest = 0.0;
        for (const auto& move : moves) largest = std::max(largest,
            evaluation.policy[static_cast<std::size_t>(action_index(move))]);
        if (largest == 0.0) {
            throw std::invalid_argument("PUCT policy must give positive weight to a legal action");
        }
        double total = 0.0;
        for (const auto& move : moves) {
            total += evaluation.policy[static_cast<std::size_t>(action_index(move))] / largest;
        }
        auto& node = tree[node_index];
        node.edges.reserve(moves.size());
        for (const auto& move : moves) {
            const double prior = (evaluation.policy[static_cast<std::size_t>(action_index(move))]
                / largest) / total;
            node.edges.push_back({move, prior, 0, 0.0, 0, std::nullopt});
        }
        node.expanded = true;
    }

    std::vector<PUCTEvaluation> infer(const std::vector<EncodedPUCTPosition>& positions,
                                    const PUCTBatchEvaluationFunction& evaluate) {
        auto evaluations = evaluate(positions);
        if (evaluations.size() != positions.size()) {
            throw std::invalid_argument("Batched PUCT evaluator must return one result per position");
        }
        for (const auto& evaluation : evaluations) validate_evaluation(evaluation);
        checked_increment(stats.inference_batches);
        stats.max_observed_batch_size = std::max(stats.max_observed_batch_size, positions.size());
        if (positions.size() > std::numeric_limits<std::size_t>::max() - stats.network_evaluations) {
            throw std::overflow_error("Batched PUCT evaluation counter overflow");
        }
        stats.network_evaluations += positions.size();
        return evaluations;
    }
};

BatchedPUCT::BatchedPUCT(PUCTOptions options, std::size_t leaf_batch_size, bool reuse_tree)
    : impl_(std::make_unique<Impl>(Impl{options, leaf_batch_size, reuse_tree, {}, {}})) {
    // Keep the legacy validation contract without modifying its implementation.
    const PUCT validated(options);
    static_cast<void>(validated);
    if (leaf_batch_size == 0) throw std::invalid_argument("Batched PUCT needs a positive leaf batch size");
    if (options.dirichlet_epsilon != 0.0) {
        throw std::invalid_argument("Batched PUCT evaluation requires root Dirichlet noise to be disabled");
    }
}

BatchedPUCT::~BatchedPUCT() = default;
BatchedPUCT::BatchedPUCT(BatchedPUCT&&) noexcept = default;
BatchedPUCT& BatchedPUCT::operator=(BatchedPUCT&&) noexcept = default;

const PUCTOptions& BatchedPUCT::options() const noexcept { return impl_->options; }
std::size_t BatchedPUCT::leaf_batch_size() const noexcept { return impl_->leaf_batch_size; }
bool BatchedPUCT::reuse_tree() const noexcept { return impl_->reuse; }
const BatchedPUCTStatistics& BatchedPUCT::stats() const noexcept { return impl_->stats; }
void BatchedPUCT::clear() noexcept { impl_->tree.clear(); }

bool BatchedPUCT::advance(Move move) {
    auto& data = *impl_;
    checked_increment(data.stats.advance_calls);
    if (!data.reuse || data.tree.empty()) {
        clear();
        return false;
    }
    const auto& edges = data.tree[0].edges;
    const auto found = std::find_if(edges.begin(), edges.end(),
        [&](const Edge& edge) { return edge.move == move; });
    if (found == edges.end() && !data.tree[0].state.is_legal(move)) return false;
    if (found == edges.end() || !found->child) {
        clear();
        return false;
    }
    // Reachable-index compaction prevents abandoned branches from accumulating
    // across a complete game. Build first so allocation failure preserves data.
    std::vector<std::size_t> indices{*found->child};
    for (std::size_t next = 0; next < indices.size(); ++next) {
        for (const auto& edge : data.tree[indices[next]].edges) {
            if (edge.child) indices.push_back(*edge.child);
        }
    }
    std::vector<std::size_t> remap(data.tree.size(), std::numeric_limits<std::size_t>::max());
    for (std::size_t i = 0; i < indices.size(); ++i) remap[indices[i]] = i;
    std::vector<Node> retained;
    retained.reserve(indices.size());
    for (const auto index : indices) {
        retained.push_back(data.tree[index]);
        for (auto& edge : retained.back().edges) {
            if (edge.child) edge.child = remap[*edge.child];
        }
    }
    data.tree = std::move(retained);
    checked_increment(data.stats.reuse_hits);
    return true;
}

PUCTSearchResult BatchedPUCT::search(const State& state,
                                    const PUCTBatchEvaluationFunction& evaluate) {
    using Clock = std::chrono::steady_clock;
    const auto started = Clock::now();
    auto& data = *impl_;
    const auto advance_calls = data.stats.advance_calls;
    const auto reuse_hits = data.stats.reuse_hits;
    data.stats = {};
    data.stats.advance_calls = advance_calls;
    data.stats.reuse_hits = reuse_hits;
    PUCTSearchResult result;
    try {
        validate_rules(state);
        if (!data.reuse || (!data.tree.empty() && !same_state(data.tree[0].state, state))) clear();
        if (state.result().finished()) {
            clear();
            result.root_value = terminal_value(state);
            return result;
        }
        if (!evaluate) throw std::invalid_argument("PUCT requires a policy/value evaluator");
        if (data.tree.empty()) {
            data.tree.reserve(std::min<std::size_t>(data.options.simulations, 255) + 1);
            data.tree.push_back({state, 0, 0, false, false, {}});
        } else {
            data.stats.reused_nodes = data.tree.size();
            data.stats.inherited_visits = data.tree[0].visits;
        }
        if (!data.tree[0].expanded) {
            const auto moves = state.legal_moves();
            const auto evaluations = data.infer({encode(state, moves)}, evaluate);
            data.expand(0, moves, evaluations[0]);
        }
        std::vector<std::size_t> initial_visits;
        std::vector<double> initial_values;
        initial_visits.reserve(data.tree[0].edges.size());
        initial_values.reserve(data.tree[0].edges.size());
        for (const auto& edge : data.tree[0].edges) {
            initial_visits.push_back(edge.visits);
            initial_values.push_back(edge.value_sum);
        }
        const auto deadline_reached = [&]() {
            return data.options.time_limit_ms != 0 &&
                std::chrono::duration<double, std::milli>(Clock::now() - started).count() >=
                static_cast<double>(data.options.time_limit_ms);
        };
        while (result.simulations < data.options.simulations) {
            std::vector<PendingLeaf> pending;
            std::vector<EncodedPUCTPosition> positions;
            const std::size_t remaining = data.options.simulations - result.simulations;
            const std::size_t capacity = std::min(remaining, data.leaf_batch_size);
            pending.reserve(capacity);
            positions.reserve(capacity);
            while (pending.size() < capacity &&
                   pending.size() < data.options.simulations - result.simulations) {
                if (result.simulations + pending.size() != 0 && deadline_reached()) break;
                std::vector<PathEdge> path;
                std::size_t leaf = 0;
                if (!data.select_leaf(0, path, leaf)) break;
                if (data.tree[leaf].state.result().finished()) {
                    data.backup(leaf, path, terminal_value(data.tree[leaf].state));
                    checked_increment(result.simulations);
                    continue;
                }
                auto moves = data.tree[leaf].state.legal_moves();
                positions.push_back(encode(data.tree[leaf].state, moves));
                pending.push_back({leaf, std::move(path), std::move(moves)});
                data.reserve(pending.back());
            }
            if (!pending.empty()) {
                const auto evaluations = data.infer(positions, evaluate);
                for (std::size_t i = 0; i < pending.size(); ++i) {
                    data.expand(pending[i].node, pending[i].moves, evaluations[i]);
                    data.release(pending[i]);
                    data.backup(pending[i].node, pending[i].path, evaluations[i].value);
                    checked_increment(result.simulations);
                }
            }
            if (result.simulations != 0 && deadline_reached()) break;
            if (pending.empty() && result.simulations < data.options.simulations) {
                throw std::logic_error("Batched PUCT could not schedule a simulation");
            }
        }
        result.nodes = data.tree.size();
        result.network_evaluations = data.stats.network_evaluations;
        double root_sum = 0.0;
        for (std::size_t i = 0; i < data.tree[0].edges.size(); ++i) {
            const auto& edge = data.tree[0].edges[i];
            const std::size_t visits = edge.visits - initial_visits[i];
            const double value_sum = edge.value_sum - initial_values[i];
            result.moves.push_back({edge.move, edge.prior, visits,
                visits == 0 ? 0.0 : value_sum / static_cast<double>(visits)});
            root_sum += value_sum;
        }
        std::sort(result.moves.begin(), result.moves.end(), [](const auto& lhs, const auto& rhs) {
            return action_index(lhs.move) < action_index(rhs.move);
        });
        const PUCTMoveStatistics* best = nullptr;
        for (const auto& move : result.moves) {
            if (!best || better_move(move, *best)) best = &move;
        }
        if (!best || best->visits == 0) {
            throw std::logic_error("A completed PUCT search must visit a root move");
        }
        result.best_move = best->move;
        result.best_value = best->value;
        result.root_value = root_sum / static_cast<double>(result.simulations);
        result.elapsed_seconds = std::chrono::duration<double>(Clock::now() - started).count();
        return result;
    } catch (...) {
        // Discard partial backups/reservations if inference, validation, or
        // allocation fails. A subsequent call cannot reuse poisoned statistics.
        clear();
        throw;
    }
}

} // namespace kingdom
