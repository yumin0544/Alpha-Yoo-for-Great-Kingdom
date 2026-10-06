#pragma once

#include "PUCT.h"

#include <memory>

namespace kingdom {

// Observation schema 1: channel-major [10, 9, 9], then the 82-action mask.
// Both observations and node edges use the same single legal_moves() result.
struct EncodedPUCTPosition {
    std::array<float, 10 * Board::kCellCount> features{};
    std::array<bool, PUCT::kActionCount> legal_mask{};
};

[[nodiscard]] EncodedPUCTPosition encode_puct_position(const State& state);

using PUCTBatchEvaluationFunction = std::function<std::vector<PUCTEvaluation>(
    const std::vector<EncodedPUCTPosition>&)>;

struct BatchedPUCTStatistics {
    // Reset for each search; terminal searches report zero inference work.
    std::size_t network_evaluations = 0;
    std::size_t inference_batches = 0;
    std::size_t max_observed_batch_size = 0;
    // Retained nodes and visits present at the beginning of the last search.
    std::size_t reused_nodes = 0;
    std::size_t inherited_visits = 0;
    // Cumulative successful subtree promotions and advance attempts.
    std::size_t advance_calls = 0;
    std::size_t reuse_hits = 0;
};

// Synchronous batch callback, asynchronous leaf selection within each wave.
// Pending leaves are reserved before selecting other leaves. Completed values
// are backed up exactly once. This evaluation-only implementation rejects root
// Dirichlet noise, avoiding ambiguous noise when an old subtree becomes root.
// Instances are single-threaded and must keep the same evaluator/model identity
// while retaining a tree. clear() before replacing that model's weights.
class BatchedPUCT {
public:
    explicit BatchedPUCT(PUCTOptions options = {}, std::size_t leaf_batch_size = 8,
                         bool reuse_tree = true);
    ~BatchedPUCT();
    BatchedPUCT(BatchedPUCT&&) noexcept;
    BatchedPUCT& operator=(BatchedPUCT&&) noexcept;
    BatchedPUCT(const BatchedPUCT&) = delete;
    BatchedPUCT& operator=(const BatchedPUCT&) = delete;

    [[nodiscard]] PUCTSearchResult search(const State& state,
                                         const PUCTBatchEvaluationFunction& evaluate);
    // Keep and compact only the played move's subtree. Legal moves with missing
    // children reset and return false; illegal moves return false unchanged.
    [[nodiscard]] bool advance(Move move);
    void clear() noexcept;
    [[nodiscard]] const PUCTOptions& options() const noexcept;
    [[nodiscard]] std::size_t leaf_batch_size() const noexcept;
    [[nodiscard]] bool reuse_tree() const noexcept;
    [[nodiscard]] const BatchedPUCTStatistics& stats() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace kingdom
