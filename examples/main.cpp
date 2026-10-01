#include "board/State.h"

#include <array>
#include <iostream>

int main() {
    using namespace kingdom;

    State game;
    if (game.board().at({4, 4}) != Cell::Neutral || game.result().finished()) {
        std::cerr << "Unexpected initial game state.\n";
        return 1;
    }

    std::cout << "Initial board (x=Black, o=White, v=neutral):\n"
              << game.board().to_string() << '\n';

    const std::array moves{
        Move::place(0, 0),
        Move::place(8, 8),
        Move::place(0, 1),
        Move::pass(),
        Move::pass(),
    };
    for (const auto& move : moves) {
        if (!game.play(move).accepted()) {
            std::cerr << "A demo move was unexpectedly rejected.\n";
            return 1;
        }
    }

    const auto score = game.score();
    const auto& result = game.result();
    if (!result.finished() || result.reason != EndReason::TwoPasses ||
        result.winner != Cell::White || score.black != 0 || score.white != 0) {
        std::cerr << "Unexpected demo result.\n";
        return 1;
    }

    std::cout << "Final board:\n" << game.board().to_string()
              << "\nTerritory: Black=" << score.black
              << ", White=" << score.white
              << "\nWinner: White (two consecutive passes; Black needs 3 extra points).\n";
    return 0;
}
