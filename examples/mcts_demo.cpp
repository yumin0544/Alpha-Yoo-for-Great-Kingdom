#include "MCTS.h"

#include <charconv>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <system_error>

namespace {

bool parse_count(std::string_view text, std::size_t& count) {
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), count);
    return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() &&
           count > 0;
}

const char* player_name(kingdom::Cell player) {
    return player == kingdom::Cell::Black ? "Black (first)" : "White (second)";
}

void print_move(const kingdom::Move& move) {
    if (move.is_pass()) {
        std::cout << "pass";
    } else {
        std::cout << move.point->row + 1 << ' ' << move.point->col + 1;
    }
}

void print_statistics(const kingdom::SearchResult& result) {
    std::cout << " | simulations=" << result.simulations
              << ", nodes=" << result.nodes
              << ", estimated win=" << std::fixed << std::setprecision(2)
              << result.win_rate * 100.0 << '%'
              << ", elapsed=" << std::setprecision(3) << result.elapsed_seconds << "s";
    if (result.elapsed_seconds > 0.0) {
        std::cout << ", measured simulations/s=" << std::setprecision(0)
                  << static_cast<double>(result.simulations) / result.elapsed_seconds;
    }
    std::cout << '\n';
}

const char* end_reason(kingdom::EndReason reason) {
    switch (reason) {
    case kingdom::EndReason::Capture:
        return "opponent captured";
    case kingdom::EndReason::Suicide:
        return "suicide move";
    case kingdom::EndReason::TwoPasses:
        return "two consecutive passes";
    case kingdom::EndReason::None:
        return "unfinished";
    }
    return "unknown";
}

void print_usage(const char* executable) {
    std::cout << "Pure MCTS demo (x=Black, o=White, v=neutral).\n"
              << "Coordinates are one-based: row column.\n\n"
              << "Usage:\n  " << executable << " [simulations]\n  "
              << executable << " --self-play [simulations per move]\n\n"
              << "Defaults: 1000 simulations for the initial recommendation;\n"
              << "          64 simulations per move for AI versus AI.\n";
}

} // namespace

int main(int argc, char* argv[]) {
    using namespace kingdom;

    bool self_play = false;
    std::size_t simulations = 1000;
    if (argc > 1) {
        const std::string_view first = argv[1];
        if (first == "--help" || first == "-h") {
            print_usage(argv[0]);
            return argc == 2 ? 0 : 1;
        }
        if (first == "--self-play") {
            self_play = true;
            simulations = 64;
            if (argc > 3 || (argc == 3 && !parse_count(argv[2], simulations))) {
                std::cerr << "Expected a positive integer simulation count.\n";
                print_usage(argv[0]);
                return 1;
            }
        } else if (argc != 2 || !parse_count(first, simulations)) {
            std::cerr << "Expected a positive integer simulation count.\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    try {
        MCTSOptions options;
        options.simulations = simulations;
        options.seed = 42;
        MCTS searcher(options);
        State game;

        std::cout << "Pure MCTS: random complete playouts, seed=" << options.seed
                  << ", simulations per search=" << simulations << '\n'
                  << "Initial board (x=Black, o=White, v=neutral):\n"
                  << game.board().to_string() << '\n';

        if (!self_play) {
            const auto result = searcher.search(game);
            if (!result.best_move) {
                std::cerr << "No move was returned for an unfinished game.\n";
                return 1;
            }
            std::cout << "Recommended move for " << player_name(game.to_play()) << ": ";
            print_move(*result.best_move);
            print_statistics(result);
            return 0;
        }

        // No stones are removed before an immediate game end. Each placement
        // occupies a cell, and at most one pass can separate two placements.
        constexpr int max_plies = 2 * Board::kCellCount + 2;
        for (int ply = 0; ply < max_plies && !game.result().finished(); ++ply) {
            const auto result = searcher.search(game);
            if (!result.best_move) {
                std::cerr << "No move was returned for an unfinished game.\n";
                return 1;
            }
            std::cout << "Move " << ply + 1 << ", " << player_name(game.to_play()) << ": ";
            print_move(*result.best_move);
            print_statistics(result);
            if (!game.play(*result.best_move).accepted()) {
                std::cerr << "The recommended move was rejected by the engine.\n";
                return 1;
            }
        }

        if (!game.result().finished()) {
            std::cerr << "Self-play exceeded the finite game length bound.\n";
            return 1;
        }
        const auto score = game.score();
        std::cout << "\nFinal board:\n" << game.board().to_string()
                  << "\nWinner: " << player_name(game.result().winner)
                  << " (" << end_reason(game.result().reason) << ").\n"
                  << "Territory: Black=" << score.black << ", White=" << score.white
                  << "; Black needs at least 3 extra points after two passes.\n";
    } catch (const std::exception& error) {
        std::cerr << "MCTS failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
