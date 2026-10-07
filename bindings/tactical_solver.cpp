#include "TacticalSolver.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>

namespace py = pybind11;
using namespace kingdom;

void bind_tactical_solver(py::module_& module) {
    py::enum_<TacticalOutcome>(module, "TacticalOutcome")
        .value("Unknown", TacticalOutcome::Unknown)
        .value("Win", TacticalOutcome::Win)
        .value("Loss", TacticalOutcome::Loss);
    py::class_<TacticalSolverOptions>(module, "TacticalSolverOptions")
        .def(py::init([](int max_depth, std::size_t max_nodes, int time_limit_ms) {
            if (time_limit_ms < 0) {
                throw std::invalid_argument("time_limit_ms cannot be negative");
            }
            TacticalSolverOptions options;
            options.max_depth = max_depth;
            options.max_nodes = max_nodes;
            options.time_limit_ms = static_cast<std::uint64_t>(time_limit_ms);
            return options;
        }), py::arg("max_depth") = 12, py::arg("max_nodes") = 100000,
            py::arg("time_limit_ms") = 1000)
        .def_readwrite("max_depth", &TacticalSolverOptions::max_depth)
        .def_readwrite("max_nodes", &TacticalSolverOptions::max_nodes)
        .def_readwrite("time_limit_ms", &TacticalSolverOptions::time_limit_ms);
    py::class_<TacticalSolverResult>(module, "TacticalSolverResult")
        .def_readonly("outcome", &TacticalSolverResult::outcome)
        .def_property_readonly("winning_moves", [](const TacticalSolverResult& result) {
            return result.winning_moves;
        })
        .def_property_readonly("losing_moves", [](const TacticalSolverResult& result) {
            return result.losing_moves;
        })
        .def_property_readonly("unknown_moves", [](const TacticalSolverResult& result) {
            return result.unknown_moves;
        })
        .def_property_readonly("principal_variation", [](const TacticalSolverResult& result) {
            return result.principal_variation;
        })
        .def_readonly("proof_depth", &TacticalSolverResult::proof_depth)
        .def_readonly("completed_depth", &TacticalSolverResult::completed_depth)
        .def_readonly("nodes", &TacticalSolverResult::nodes)
        .def_readonly("elapsed_ms", &TacticalSolverResult::elapsed_ms)
        .def_readonly("budget_exhausted", &TacticalSolverResult::budget_exhausted);
    module.def("solve_tactics", [](const State& state, TacticalSolverOptions options) {
        // Copy while holding the GIL; all exploration uses the private snapshot.
        const State snapshot = state;
        py::gil_scoped_release release;
        return solve_tactics(snapshot, options);
    }, py::arg("state"), py::arg("options") = TacticalSolverOptions{},
    "Prove current-player Win/Loss within depth/node/time budgets; otherwise Unknown. "
    "Depth counts individual moves (plies), not pairs of moves. Does not mutate state.");
}
