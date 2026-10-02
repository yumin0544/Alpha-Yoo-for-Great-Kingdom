#include "MCTS.h"
#include "PUCT.h"
#include "board/State.h"

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <mutex>
#include <optional>
#include <string>
#include <stdexcept>
#include <utility>

namespace py = pybind11;
using namespace kingdom;

namespace {

// State is copied while the GIL is held. Python may continue using the original
// during search; the mutex protects the RNG when this MCTS object is shared.
class PythonMCTS {
public:
    explicit PythonMCTS(MCTSOptions options) : searcher_(options) {}

    SearchResult search(const State& state) {
        const State snapshot = state;
        py::gil_scoped_release release;
        const std::lock_guard<std::mutex> lock(mutex_);
        return searcher_.search(snapshot);
    }

    MCTSOptions options() const { return searcher_.options(); }

private:
    MCTS searcher_;
    std::mutex mutex_;
};

class PythonPUCT {
public:
    explicit PythonPUCT(PUCTOptions options) : searcher_(options) {}

    PUCTSearchResult search(const State& state, const py::function& evaluator) {
        const State snapshot = state;
        // Capture by reference: the Python call frame owns the callable until
        // search returns. Python objects are created/destroyed only with the GIL.
        const auto evaluate = [&evaluator](const State& position) {
            py::gil_scoped_acquire acquire;
            const auto evaluated = evaluator(py::cast(State(position)));
            const auto output = evaluated.cast<
                std::pair<std::array<double, PUCT::kActionCount>, double>>();
            return PUCTEvaluation{output.first, output.second};
        };
        py::gil_scoped_release release;
        const std::lock_guard<std::recursive_mutex> lock(mutex_);
        if (searching_) {
            throw std::runtime_error("Recursive search on the same PUCT object is not supported");
        }
        searching_ = true;
        struct ResetFlag {
            bool& flag;
            ~ResetFlag() { flag = false; }
        } reset{searching_};
        return searcher_.search(snapshot, evaluate);
    }

    PUCTOptions options() const { return searcher_.options(); }

private:
    PUCT searcher_;
    std::recursive_mutex mutex_;
    bool searching_ = false;
};

} // namespace

PYBIND11_MODULE(my_board_engine, module) {
    module.doc() = "Great Kingdom C++ rules, pure MCTS and policy/value PUCT. Coordinates are zero-based.";
    module.attr("__version__") = "0.1.0";
    module.attr("BOARD_SIZE") = Board::kSize;
    module.attr("CELL_COUNT") = Board::kCellCount;

    py::enum_<Cell>(module, "Cell")
        .value("Empty", Cell::Empty)
        .value("Black", Cell::Black)
        .value("White", Cell::White)
        .value("Neutral", Cell::Neutral);
    py::enum_<EndReason>(module, "EndReason")
        .value("None_", EndReason::None)
        .value("Capture", EndReason::Capture)
        .value("TwoPasses", EndReason::TwoPasses)
        .value("Suicide", EndReason::Suicide);
    py::enum_<SuicideRule>(module, "SuicideRule")
        .value("Forbidden", SuicideRule::Forbidden)
        .value("Loses", SuicideRule::Loses);
    py::enum_<MoveError>(module, "MoveError")
        .value("None_", MoveError::None)
        .value("GameOver", MoveError::GameOver)
        .value("OutOfBounds", MoveError::OutOfBounds)
        .value("Occupied", MoveError::Occupied)
        .value("OpponentTerritory", MoveError::OpponentTerritory)
        .value("OwnTerritory", MoveError::OwnTerritory)
        .value("NoStones", MoveError::NoStones)
        .value("Suicide", MoveError::Suicide);
    module.def("is_player", &is_player, py::arg("cell"));
    module.def("opponent", &opponent, py::arg("player"));

    py::class_<Position>(module, "Position")
        .def(py::init([](int row, int col) { return Position{row, col}; }),
             py::arg("row"), py::arg("col"))
        .def_readwrite("row", &Position::row)
        .def_readwrite("col", &Position::col)
        .def(py::self == py::self)
        .def("__repr__", [](Position point) {
            return "Position(" + std::to_string(point.row) + ", "
                + std::to_string(point.col) + ")";
        });

    py::class_<Move>(module, "Move")
        .def_static("place", &Move::place, py::arg("row"), py::arg("col"))
        .def_static("pass_turn", &Move::pass)
        .def_property_readonly("point", [](const Move& move) { return move.point; })
        .def("is_pass", &Move::is_pass)
        .def(py::self == py::self)
        .def("__repr__", [](const Move& move) {
            if (move.is_pass()) {
                return std::string("Move.pass_turn()");
            }
            return "Move.place(" + std::to_string(move.point->row) + ", "
                + std::to_string(move.point->col) + ")";
        });

    py::class_<Board::Territory>(module, "Territory")
        .def_property_readonly("owners", [](const Board::Territory& territory) {
            return territory.owners;
        })
        .def_readonly("black", &Board::Territory::black)
        .def_readonly("white", &Board::Territory::white);

    py::class_<Board>(module, "Board")
        .def(py::init<std::optional<Position>>(),
             py::arg("neutral") = std::optional<Position>{Position{4, 4}})
        .def(py::init<const Board::Cells&>(), py::arg("cells"))
        .def_static("in_bounds", &Board::in_bounds, py::arg("point"))
        .def_static("index", &Board::index, py::arg("point"))
        .def_static("position", &Board::position, py::arg("index"))
        .def("at", &Board::at, py::arg("point"))
        .def_property_readonly("cells", [](const Board& board) { return board.cells(); })
        .def("place", &Board::place, py::arg("point"), py::arg("player"),
             "Low-level setup operation; use State.place for actual game moves.")
        .def("clear", &Board::clear, py::arg("point"))
        .def("count", &Board::count, py::arg("cell"))
        .def("group_at", &Board::group_at, py::arg("point"))
        .def("liberties", &Board::liberties, py::arg("point"))
        .def("territory", &Board::territory, py::arg("allow_single_edge") = true)
        .def("to_string", &Board::to_string)
        .def("__str__", &Board::to_string);

    py::class_<GameRules>(module, "GameRules")
        .def(py::init([](SuicideRule suicide_rule, bool allow_own_territory_moves,
                         bool allow_single_edge_territory, int stones_per_player) {
            return GameRules{suicide_rule, allow_own_territory_moves,
                             allow_single_edge_territory, stones_per_player};
        }), py::arg("suicide_rule") = SuicideRule::Loses,
            py::arg("allow_own_territory_moves") = false,
            py::arg("allow_single_edge_territory") = true,
            py::arg("stones_per_player") = 41)
        .def_readwrite("suicide_rule", &GameRules::suicide_rule)
        .def_readwrite("allow_own_territory_moves", &GameRules::allow_own_territory_moves)
        .def_readwrite("allow_single_edge_territory", &GameRules::allow_single_edge_territory)
        .def_readwrite("stones_per_player", &GameRules::stones_per_player);

    py::class_<Score>(module, "Score")
        .def(py::init([](int black, int white) { return Score{black, white}; }),
             py::arg("black") = 0, py::arg("white") = 0)
        .def_readonly("black", &Score::black)
        .def_readonly("white", &Score::white)
        .def("winner", &Score::winner)
        .def(py::self == py::self);
    py::class_<GameResult>(module, "GameResult")
        .def_readonly("winner", &GameResult::winner)
        .def_readonly("reason", &GameResult::reason)
        .def_property_readonly("score", [](const GameResult& result) { return result.score; })
        .def_readonly("captured_stones", &GameResult::captured_stones)
        .def("finished", &GameResult::finished)
        .def(py::self == py::self);
    py::class_<MoveOutcome>(module, "MoveOutcome")
        .def_readonly("error", &MoveOutcome::error)
        .def_property_readonly("result", [](const MoveOutcome& outcome) {
            return outcome.result;
        })
        .def("accepted", &MoveOutcome::accepted);

    py::class_<State>(module, "State")
        .def(py::init<GameRules, std::optional<Position>>(),
             py::arg("rules") = GameRules{},
             py::arg("neutral") = std::optional<Position>{Position{4, 4}})
        .def(py::init<Board, Cell, GameRules>(), py::arg("board"), py::arg("to_play"),
             py::arg("rules") = GameRules{},
             "Analysis setup constructor; use copy() to preserve full game state.")
        // Return detached values instead of exposing mutable C++ references.
        .def_property_readonly("board", [](const State& state) { return state.board(); })
        .def_property_readonly("to_play", &State::to_play)
        .def_property_readonly("rules", [](const State& state) { return state.rules(); })
        .def_property_readonly("ownership", [](const State& state) { return state.ownership(); })
        .def_property_readonly("consecutive_passes", &State::consecutive_passes)
        .def_property_readonly("result", [](const State& state) { return state.result(); })
        .def("territory_owner", &State::territory_owner, py::arg("point"))
        .def("remaining_stones", &State::remaining_stones, py::arg("player"))
        .def("score", &State::score)
        .def("play", &State::play, py::arg("move"))
        .def("place", [](State& state, int row, int col) {
            return state.play(Move::place(row, col));
        }, py::arg("row"), py::arg("col"))
        .def("pass_turn", [](State& state) { return state.play(Move::pass()); })
        .def("is_legal", &State::is_legal, py::arg("move"))
        .def("legal_moves", &State::legal_moves)
        .def("copy", [](const State& state) { return State(state); })
        .def("__copy__", [](const State& state) { return State(state); })
        .def("__deepcopy__", [](const State& state, const py::dict&) { return State(state); },
             py::arg("memo"));

    py::class_<MCTSOptions>(module, "MCTSOptions")
        .def(py::init([](std::size_t simulations, double exploration,
                         std::uint64_t seed, std::uint64_t time_limit_ms) {
            return MCTSOptions{simulations, exploration, seed, time_limit_ms};
        }), py::arg("simulations") = 1000,
            py::arg("exploration") = 1.4142135623730951,
            py::arg("seed") = 42, py::arg("time_limit_ms") = 0)
        .def_readwrite("simulations", &MCTSOptions::simulations)
        .def_readwrite("exploration", &MCTSOptions::exploration)
        .def_readwrite("seed", &MCTSOptions::seed)
        .def_readwrite("time_limit_ms", &MCTSOptions::time_limit_ms);
    py::class_<MoveStatistics>(module, "MoveStatistics")
        .def_property_readonly("move", [](const MoveStatistics& stats) { return stats.move; })
        .def_readonly("visits", &MoveStatistics::visits)
        .def_readonly("win_rate", &MoveStatistics::win_rate);
    py::class_<SearchResult>(module, "SearchResult")
        .def_property_readonly("best_move", [](const SearchResult& result) {
            return result.best_move;
        })
        .def_readonly("simulations", &SearchResult::simulations)
        .def_readonly("nodes", &SearchResult::nodes)
        .def_readonly("total_rollout_plies", &SearchResult::total_rollout_plies)
        .def_readonly("win_rate", &SearchResult::win_rate)
        .def_readonly("elapsed_seconds", &SearchResult::elapsed_seconds)
        .def_property_readonly("moves", [](const SearchResult& result) { return result.moves; });
    py::class_<PythonMCTS>(module, "MCTS")
        .def(py::init<MCTSOptions>(), py::arg("options") = MCTSOptions{})
        .def_property_readonly("options", &PythonMCTS::options)
        .def("search", &PythonMCTS::search, py::arg("state"),
             "Search a snapshot of state. Releases the GIL; shared searchers serialize calls.");

    py::class_<PUCTOptions>(module, "PUCTOptions")
        .def(py::init([](std::size_t simulations, double c_puct, std::uint64_t seed,
                         std::uint64_t time_limit_ms, double dirichlet_alpha,
                         double dirichlet_epsilon) {
            PUCTOptions options;
            options.simulations = simulations;
            options.c_puct = c_puct;
            options.seed = seed;
            options.time_limit_ms = time_limit_ms;
            options.dirichlet_alpha = dirichlet_alpha;
            options.dirichlet_epsilon = dirichlet_epsilon;
            return options;
        }), py::arg("simulations") = 128, py::arg("c_puct") = 1.5,
            py::arg("seed") = 42, py::arg("time_limit_ms") = 0,
            py::arg("dirichlet_alpha") = 0.3, py::arg("dirichlet_epsilon") = 0.0)
        .def_readwrite("simulations", &PUCTOptions::simulations)
        .def_readwrite("c_puct", &PUCTOptions::c_puct)
        .def_readwrite("seed", &PUCTOptions::seed)
        .def_readwrite("time_limit_ms", &PUCTOptions::time_limit_ms)
        .def_readwrite("dirichlet_alpha", &PUCTOptions::dirichlet_alpha)
        .def_readwrite("dirichlet_epsilon", &PUCTOptions::dirichlet_epsilon);
    py::class_<PUCTMoveStatistics>(module, "PUCTMoveStatistics")
        .def_property_readonly("move", [](const PUCTMoveStatistics& stats) { return stats.move; })
        .def_readonly("prior", &PUCTMoveStatistics::prior)
        .def_readonly("visits", &PUCTMoveStatistics::visits)
        .def_readonly("value", &PUCTMoveStatistics::value);
    py::class_<PUCTSearchResult>(module, "PUCTSearchResult")
        .def_property_readonly("best_move", [](const PUCTSearchResult& result) { return result.best_move; })
        .def_readonly("simulations", &PUCTSearchResult::simulations)
        .def_readonly("nodes", &PUCTSearchResult::nodes)
        .def_readonly("network_evaluations", &PUCTSearchResult::network_evaluations)
        .def_readonly("root_value", &PUCTSearchResult::root_value)
        .def_readonly("best_value", &PUCTSearchResult::best_value)
        .def_readonly("elapsed_seconds", &PUCTSearchResult::elapsed_seconds)
        .def_property_readonly("moves", [](const PUCTSearchResult& result) { return result.moves; });
    py::class_<PythonPUCT>(module, "PUCT")
        .def(py::init<PUCTOptions>(), py::arg("options") = PUCTOptions{})
        .def_property_readonly("options", &PythonPUCT::options)
        .def("search", &PythonPUCT::search, py::arg("state"), py::arg("evaluator"),
             "C++ PUCT using evaluator(state) -> (82 policy weights, current-player value).");
}
