#include "BatchedPUCT.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <stdexcept>

namespace py = pybind11;
using namespace kingdom;

namespace {

// NumPy owns both allocations. Callbacks may retain them after the C++ wave
// completes without keeping pointers into a temporary search arena.
py::tuple arrays(const std::vector<EncodedPUCTPosition>& positions) {
    const auto count = static_cast<py::ssize_t>(positions.size());
    py::array_t<float> features({count, py::ssize_t{10}, py::ssize_t{9}, py::ssize_t{9}});
    py::array_t<bool> masks({count, py::ssize_t{PUCT::kActionCount}});
    for (std::size_t i = 0; i < positions.size(); ++i) {
        std::memcpy(features.mutable_data() + i * 810, positions[i].features.data(),
                    positions[i].features.size() * sizeof(float));
        std::memcpy(masks.mutable_data() + i * PUCT::kActionCount,
                    positions[i].legal_mask.data(), positions[i].legal_mask.size() * sizeof(bool));
    }
    return py::make_tuple(std::move(features), std::move(masks));
}

std::vector<PUCTEvaluation> predictions(const py::object& returned, std::size_t count) {
    if (!py::isinstance<py::tuple>(returned) || py::len(returned) != 2) {
        throw std::invalid_argument("Batch evaluator must return (policy[N,82], values[N])");
    }
    const auto pair = returned.cast<py::tuple>();
    auto policy = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(pair[0]);
    auto values = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(pair[1]);
    if (!policy || !values || policy.ndim() != 2 || values.ndim() != 1 ||
        policy.shape(0) != static_cast<py::ssize_t>(count) ||
        policy.shape(1) != PUCT::kActionCount || values.shape(0) != policy.shape(0)) {
        throw std::invalid_argument("Batch evaluator returned unexpected policy/value shapes");
    }
    std::vector<PUCTEvaluation> result(count);
    for (std::size_t i = 0; i < count; ++i) {
        const double* row = policy.data() + i * PUCT::kActionCount;
        std::copy_n(row, PUCT::kActionCount, result[i].policy.begin());
        result[i].value = values.data()[i];
    }
    return result;
}

class PythonBatchedPUCT {
public:
    PythonBatchedPUCT(PUCTOptions options, std::size_t leaf_batch_size, bool reuse_tree)
        : searcher_(options, leaf_batch_size, reuse_tree) {}

    PUCTSearchResult search(const State& state, const py::function& evaluator) {
        const State snapshot = state;
        const auto evaluate = [&evaluator](const std::vector<EncodedPUCTPosition>& positions) {
            py::gil_scoped_acquire acquire;
            const py::tuple inputs = arrays(positions);
            return predictions(evaluator(inputs[0], inputs[1]), positions.size());
        };
        py::gil_scoped_release release;
        const std::lock_guard<std::recursive_mutex> lock(mutex_);
        check_idle();
        searching_ = true;
        struct ResetFlag {
            bool& value;
            ~ResetFlag() { value = false; }
        } reset{searching_};
        return searcher_.search(snapshot, evaluate);
    }

    bool advance(Move move) {
        py::gil_scoped_release release;
        const std::lock_guard<std::recursive_mutex> lock(mutex_);
        check_idle();
        return searcher_.advance(move);
    }

    void clear() {
        py::gil_scoped_release release;
        const std::lock_guard<std::recursive_mutex> lock(mutex_);
        check_idle();
        searcher_.clear();
    }

    BatchedPUCTStatistics stats() {
        py::gil_scoped_release release;
        const std::lock_guard<std::recursive_mutex> lock(mutex_);
        check_idle();
        return searcher_.stats();
    }

    PUCTOptions options() const { return searcher_.options(); }
    std::size_t leaf_batch_size() const { return searcher_.leaf_batch_size(); }
    bool reuse_tree() const { return searcher_.reuse_tree(); }

private:
    void check_idle() const {
        if (searching_) {
            throw std::runtime_error("Recursive access to an active BatchedPUCT is not supported");
        }
    }
    BatchedPUCT searcher_;
    std::recursive_mutex mutex_;
    bool searching_ = false;
};

} // namespace

void bind_batched_puct(py::module_& module) {
    py::class_<PythonBatchedPUCT>(module, "BatchedPUCT")
        .def(py::init<PUCTOptions, std::size_t, bool>(),
             py::arg("options") = PUCTOptions{}, py::arg("leaf_batch_size") = 8,
             py::arg("reuse_tree") = true)
        .def_property_readonly("options", &PythonBatchedPUCT::options)
        .def_property_readonly("leaf_batch_size", &PythonBatchedPUCT::leaf_batch_size)
        .def_property_readonly("reuse_tree", &PythonBatchedPUCT::reuse_tree)
        .def("search", &PythonBatchedPUCT::search, py::arg("state"), py::arg("evaluator"),
             "Batch callback: evaluator(features[N,10,9,9], masks[N,82]) -> (policy[N,82], values[N]).")
        .def("advance", &PythonBatchedPUCT::advance, py::arg("move"),
             "Advance this model's tree after each accepted actual move; compact the retained subtree.")
        .def("clear", &PythonBatchedPUCT::clear,
             "Discard retained statistics before changing evaluator/model identity or weights.")
        .def_property_readonly("stats", [](PythonBatchedPUCT& searcher) {
            const auto stats = searcher.stats();
            py::dict result;
            result["network_evaluations"] = stats.network_evaluations;
            result["inference_batches"] = stats.inference_batches;
            result["max_observed_batch_size"] = stats.max_observed_batch_size;
            result["reused_nodes"] = stats.reused_nodes;
            result["inherited_visits"] = stats.inherited_visits;
            result["advance_calls"] = stats.advance_calls;
            result["reuse_hits"] = stats.reuse_hits;
            return result;
        });
    module.def("encode_puct_batch", [](const std::vector<State>& states) {
        std::vector<EncodedPUCTPosition> encoded;
        encoded.reserve(states.size());
        {
            py::gil_scoped_release release;
            for (const auto& state : states) encoded.push_back(encode_puct_position(state));
        }
        return arrays(encoded);
    }, py::arg("states"), "C++ schema-1 encoder; copied states and independently owned batch arrays.");
}
