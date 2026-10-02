#include "PUCT.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace kingdom {
namespace {

struct Edge {
    Move move;
    double prior;
    std::size_t visits = 0;
    double value_sum = 0.0;
    std::optional<std::size_t> child;

    [[nodiscard]] double value() const noexcept {
        return visits == 0 ? 0.0 : value_sum / static_cast<double>(visits);
    }
};

struct TreeNode {
    Cell player;
    std::size_t visits = 0;
    bool expanded = false;
    std::vector<Edge> edges;
};

struct PathEdge {
    std::size_t node;
    std::size_t edge;
};

int move_index(const Move& move) {
    return move.is_pass() ? Board::kCellCount : Board::index(*move.point);
}

void validate_evaluation(const PUCTEvaluation& evaluation) {
    if (!std::isfinite(evaluation.value) || evaluation.value < -1.0 ||
        evaluation.value > 1.0) {
        throw std::invalid_argument("PUCT evaluator value must be finite and in [-1, 1]");
    }
    for (const double weight : evaluation.policy) {
        if (!std::isfinite(weight) || weight < 0.0) {
            throw std::invalid_argument("PUCT policy weights must be finite and non-negative");
        }
    }
}

double expand(std::vector<TreeNode>& tree, std::size_t node_index,
              const State& state, const PUCTEvaluationFunction& evaluate,
              std::size_t& network_evaluations) {
    const auto moves = state.legal_moves();
    if (moves.empty()) {
        throw std::logic_error("A non-terminal PUCT state must permit a pass");
    }
    const PUCTEvaluation evaluation = evaluate(state);
    ++network_evaluations;
    validate_evaluation(evaluation);

    double largest = 0.0;
    for (const auto& move : moves) {
        largest = std::max(largest,
            evaluation.policy[static_cast<std::size_t>(move_index(move))]);
    }
    if (largest == 0.0) {
        throw std::invalid_argument("PUCT policy must give positive weight to a legal action");
    }
    // Scaling first keeps sums finite even for very large valid weights.
    double total = 0.0;
    for (const auto& move : moves) {
        total += evaluation.policy[static_cast<std::size_t>(move_index(move))] / largest;
    }
    auto& node = tree[node_index];
    node.edges.reserve(moves.size());
    for (const auto& move : moves) {
        const double prior =
            (evaluation.policy[static_cast<std::size_t>(move_index(move))] / largest) / total;
        node.edges.push_back({move, prior, 0, 0.0, std::nullopt});
    }
    node.expanded = true;
    return evaluation.value;
}

void apply_root_noise(TreeNode& root, const PUCTOptions& options,
                      std::mt19937_64& random) {
    if (options.dirichlet_epsilon == 0.0) return;

    std::gamma_distribution<double> gamma(options.dirichlet_alpha, 1.0);
    std::vector<double> noise(root.edges.size());
    double largest = 0.0;
    for (double& weight : noise) {
        weight = gamma(random);
        if (!std::isfinite(weight) || weight < 0.0) weight = 0.0;
        largest = std::max(largest, weight);
    }
    double total = 0.0;
    if (largest > 0.0) {
        for (double& weight : noise) {
            weight /= largest;
            total += weight;
        }
    }
    const double uniform = 1.0 / static_cast<double>(root.edges.size());
    const double epsilon = options.dirichlet_epsilon;
    double combined_total = 0.0;
    for (std::size_t i = 0; i < root.edges.size(); ++i) {
        const double sampled = total > 0.0 ? noise[i] / total : uniform;
        root.edges[i].prior = (1.0 - epsilon) * root.edges[i].prior + epsilon * sampled;
        combined_total += root.edges[i].prior;
    }
    for (auto& edge : root.edges) edge.prior /= combined_total;
}

std::size_t select_edge(const TreeNode& node, double c_puct) {
    if (node.edges.empty()) throw std::logic_error("PUCT selection needs a legal edge");
    const double parent_scale = std::sqrt(static_cast<double>(node.visits));
    std::size_t best = 0;
    double best_score = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < node.edges.size(); ++i) {
        const auto& edge = node.edges[i];
        const double exploration = edge.prior * parent_scale /
            (1.0 + static_cast<double>(edge.visits));
        // Dividing the whole score by a positive c_puct preserves selection.
        // Use that form for large coefficients to avoid intermediate overflow;
        // keep the original scale for small coefficients to avoid Q/c overflow.
        const double score = c_puct >= 1.0
            ? edge.value() / c_puct + exploration
            : edge.value() + c_puct * exploration;
        const auto& previous = node.edges[best];
        if (score > best_score ||
            (score == best_score && (edge.prior > previous.prior ||
                (edge.prior == previous.prior && move_index(edge.move) < move_index(previous.move))))) {
            best = i;
            best_score = score;
        }
    }
    return best;
}

double terminal_value(const State& state) {
    if (!is_player(state.result().winner)) {
        throw std::logic_error("A terminal PUCT position must have a winning player");
    }
    return state.result().winner == state.to_play() ? 1.0 : -1.0;
}

bool better_move(const PUCTMoveStatistics& candidate,
                 const PUCTMoveStatistics& incumbent) {
    if (candidate.visits != incumbent.visits) return candidate.visits > incumbent.visits;
    if (candidate.value != incumbent.value) return candidate.value > incumbent.value;
    if (candidate.prior != incumbent.prior) return candidate.prior > incumbent.prior;
    return move_index(candidate.move) < move_index(incumbent.move);
}

} // namespace

PUCT::PUCT(PUCTOptions options) : options_(options), random_(options.seed) {
    if (options_.simulations == 0) {
        throw std::invalid_argument("PUCT needs at least one simulation");
    }
    if (!std::isfinite(options_.c_puct) || options_.c_puct <= 0.0) {
        throw std::invalid_argument("PUCT exploration coefficient must be finite and positive");
    }
    if (!std::isfinite(options_.dirichlet_alpha) || options_.dirichlet_alpha <= 0.0) {
        throw std::invalid_argument("Dirichlet alpha must be finite and positive");
    }
    if (!std::isfinite(options_.dirichlet_epsilon) || options_.dirichlet_epsilon < 0.0 ||
        options_.dirichlet_epsilon > 1.0) {
        throw std::invalid_argument("Dirichlet epsilon must be finite and in [0, 1]");
    }
}

PUCTSearchResult PUCT::search(const State& state, const PUCTEvaluationFunction& evaluate) {
    using Clock = std::chrono::steady_clock;
    const auto started = Clock::now();
    PUCTSearchResult result;
    if (state.result().finished()) {
        result.root_value = terminal_value(state);
        return result;
    }
    if (!evaluate) throw std::invalid_argument("PUCT requires a policy/value evaluator");

    std::vector<TreeNode> tree;
    // Start with a small arena; indexed links survive subsequent growth.
    tree.reserve(std::min<std::size_t>(options_.simulations, 255) + 1);
    tree.push_back({state.to_play(), 0, false, {}});
    static_cast<void>(expand(tree, 0, state, evaluate, result.network_evaluations));
    apply_root_noise(tree[0], options_, random_);

    std::vector<PathEdge> path;
    path.reserve(2 * Board::kCellCount + 2);
    for (std::size_t iteration = 0; iteration < options_.simulations; ++iteration) {
        State simulation = state;
        std::size_t node_index = 0;
        path.clear();
        double leaf_value = 0.0;
        while (true) {
            if (simulation.result().finished()) {
                leaf_value = terminal_value(simulation);
                break;
            }
            if (!tree[node_index].expanded) {
                leaf_value = expand(tree, node_index, simulation, evaluate,
                                    result.network_evaluations);
                break;
            }

            const std::size_t edge_index = select_edge(tree[node_index], options_.c_puct);
            const Move move = tree[node_index].edges[edge_index].move;
            if (!simulation.play(move).accepted()) {
                throw std::logic_error("A stored PUCT edge must remain legal");
            }
            path.push_back({node_index, edge_index});
            auto child = tree[node_index].edges[edge_index].child;
            if (!child.has_value()) {
                child = tree.size();
                tree.push_back({simulation.to_play(), 0, false, {}});
                // Arena indices remain valid after a vector reallocation.
                tree[node_index].edges[edge_index].child = child;
            }
            node_index = *child;
        }

        ++tree[0].visits;
        for (const auto& step : path) {
            auto& edge = tree[step.node].edges[step.edge];
            const double value = tree[step.node].player == simulation.to_play()
                ? leaf_value : -leaf_value;
            ++edge.visits;
            edge.value_sum += value;
            ++tree[*edge.child].visits;
        }
        ++result.simulations;

        if (options_.time_limit_ms != 0) {
            const double elapsed_ms =
                std::chrono::duration<double, std::milli>(Clock::now() - started).count();
            if (elapsed_ms >= static_cast<double>(options_.time_limit_ms)) break;
        }
    }

    result.nodes = tree.size();
    result.moves.reserve(tree[0].edges.size());
    double root_sum = 0.0;
    for (const auto& edge : tree[0].edges) {
        result.moves.push_back({edge.move, edge.prior, edge.visits, edge.value()});
        root_sum += edge.value_sum;
    }
    std::sort(result.moves.begin(), result.moves.end(), [](const auto& lhs, const auto& rhs) {
        return move_index(lhs.move) < move_index(rhs.move);
    });
    const PUCTMoveStatistics* best = nullptr;
    for (const auto& move : result.moves) {
        if (best == nullptr || better_move(move, *best)) best = &move;
    }
    if (best == nullptr || best->visits == 0) {
        throw std::logic_error("A completed PUCT search must visit a root move");
    }
    result.best_move = best->move;
    result.best_value = best->value;
    result.root_value = root_sum / static_cast<double>(result.simulations);
    result.elapsed_seconds = std::chrono::duration<double>(Clock::now() - started).count();
    return result;
}

} // namespace kingdom
