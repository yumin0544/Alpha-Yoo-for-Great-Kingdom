#include "board/State.h"

#include <cctype>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

using namespace kingdom;

class Console {
public:
    Console() {
#ifdef _WIN32
        DWORD mode = 0;
        const HANDLE output = GetStdHandle(STD_OUTPUT_HANDLE);
        if (GetConsoleMode(output, &mode)) {
            old_output_page_ = GetConsoleOutputCP();
            SetConsoleOutputCP(CP_UTF8);
            color_ = SetConsoleMode(output, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0;
            if (color_) {
                output_ = output;
                old_output_mode_ = mode;
            }
        }
        if (GetConsoleMode(GetStdHandle(STD_INPUT_HANDLE), &mode)) {
            old_input_page_ = GetConsoleCP();
            SetConsoleCP(CP_UTF8);
        }
#endif
    }

    ~Console() {
        std::cout.flush();
#ifdef _WIN32
        if (color_) SetConsoleMode(output_, old_output_mode_);
        if (old_output_page_ != 0) SetConsoleOutputCP(old_output_page_);
        if (old_input_page_ != 0) SetConsoleCP(old_input_page_);
#endif
    }

    void point(char glyph, Cell owner) const {
        if (color_ && owner == Cell::Black) std::cout << "\x1b[94m";
        if (color_ && owner == Cell::White) std::cout << "\x1b[38;2;255;165;0m";
        std::cout << glyph;
        if (color_ && is_player(owner)) std::cout << "\x1b[0m";
    }

private:
    bool color_ = false;
#ifdef _WIN32
    HANDLE output_ = INVALID_HANDLE_VALUE;
    DWORD old_output_mode_ = 0;
    UINT old_output_page_ = 0;
    UINT old_input_page_ = 0;
#endif
};

std::string clean_command(std::string line) {
    const auto first = line.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = line.find_last_not_of(" \t\r\n");
    line = line.substr(first, last - first + 1);
    for (char& value : line) {
        value = static_cast<char>(std::tolower(static_cast<unsigned char>(value)));
    }
    return line;
}

bool is_quit(const std::string& command) {
    return command == "q" || command == "quit" || command == "exit" || command == "종료";
}

std::string_view player_name(Cell player) {
    return player == Cell::Black ? "선공" : "후공";
}

void print_help() {
    std::cout << "\n행 열 입력: 3 4 -> 3행 4열에 놓기 (1~9)\n"
              << "pass 또는 패스: 차례 넘기기\n"
              << "help 또는 도움말: 사용법 보기\n"
              << "quit 또는 q 또는 종료: 프로그램 종료\n"
              << "x=선공(파랑), o=후공(주황), #=중립 돌\n"
              << "B=선공의 집, W=후공의 집, .=빈칸\n"
              << "완성된 집에는 누구도 놓을 수 없습니다.\n"
              << "상대 돌 포획은 즉시 승리, 자충수는 즉시 패배입니다.\n"
              << "연속 패스 시 집을 셉니다. 선공은 3칸 이상 앞서야 승리합니다.\n\n";
}

void print_board(const State& game, const Console& console) {
    std::cout << "\n     1 2 3 4 5 6 7 8 9\n"
              << "   +-------------------+\n";
    for (int row = 0; row < Board::kSize; ++row) {
        std::cout << std::setw(2) << row + 1 << " | ";
        for (int col = 0; col < Board::kSize; ++col) {
            const Position point{row, col};
            const Cell cell = game.board().at(point);
            Cell owner = cell;
            char glyph = '.';
            switch (cell) {
                case Cell::Black: glyph = 'x'; break;
                case Cell::White: glyph = 'o'; break;
                case Cell::Neutral: glyph = '#'; break;
                case Cell::Empty:
                    owner = game.territory_owner(point);
                    if (owner == Cell::Black) glyph = 'B';
                    if (owner == Cell::White) glyph = 'W';
                    break;
            }
            console.point(glyph, owner);
            std::cout << ' ';
        }
        std::cout << "|\n";
    }
    const auto score = game.score();
    std::cout << "   +-------------------+\n"
              << "남은 돌: 선공 " << game.remaining_stones(Cell::Black)
              << "개 / 후공 " << game.remaining_stones(Cell::White) << "개\n"
              << "완성된 집: 선공 " << score.black << "칸 / 후공 " << score.white << "칸\n"
              << "연속 패스: " << game.consecutive_passes() << "회\n";
}

void print_error(MoveError error) {
    switch (error) {
        case MoveError::None: break;
        case MoveError::GameOver: std::cout << "이미 끝난 게임입니다.\n"; break;
        case MoveError::OutOfBounds: std::cout << "행과 열은 1~9 사이여야 합니다.\n"; break;
        case MoveError::Occupied: std::cout << "이미 돌이 있는 칸입니다.\n"; break;
        case MoveError::OpponentTerritory: std::cout << "상대 집 안에는 놓을 수 없습니다.\n"; break;
        case MoveError::OwnTerritory: std::cout << "자기 집 안에는 놓을 수 없습니다.\n"; break;
        case MoveError::NoStones: std::cout << "남은 돌이 없습니다. 패스해 주세요.\n"; break;
        case MoveError::Suicide: std::cout << "이 변형에서는 자충수가 금지되어 있습니다.\n"; break;
    }
}

void print_result(const State& game) {
    const auto& result = game.result();
    std::cout << '\n' << player_name(result.winner) << " 승리 (";
    switch (result.reason) {
        case EndReason::Capture: std::cout << "상대 돌 포획"; break;
        case EndReason::Suicide: std::cout << "자충수"; break;
        case EndReason::TwoPasses: std::cout << "연속 패스"; break;
        case EndReason::None: return;
    }
    std::cout << ")\n";
    if (result.reason == EndReason::Capture) {
        std::cout << "포획한 돌: " << result.captured_stones << "개\n";
    } else if (result.reason == EndReason::Suicide) {
        std::cout << player_name(opponent(result.winner))
                  << "이 자기 돌 무리의 마지막 빈틈을 막아 패배했습니다.\n";
    }
}

// Returns false for quit/EOF; true after a completed game.
bool play_game(State& game, const Console& console) {
    print_board(game, console);
    std::string line;
    while (!game.result().finished()) {
        std::cout << "\n차례: " << player_name(game.to_play())
                  << (game.to_play() == Cell::Black ? " (파랑/x)" : " (주황/o)")
                  << "\n행 열 또는 pass > " << std::flush;
        if (!std::getline(std::cin, line)) return false;
        const std::string command = clean_command(line);
        if (is_quit(command)) return false;
        if (command == "help" || command == "도움말") {
            print_help();
            continue;
        }

        Move move = Move::pass();
        if (command != "pass" && command != "패스") {
            std::istringstream input(command);
            int row = 0;
            int col = 0;
            std::string extra;
            if (!(input >> row >> col) || (input >> extra)) {
                std::cout << "입력을 확인해 주세요. 예: 3 4, pass, quit\n";
                continue;
            }
            if (row < 1 || row > Board::kSize || col < 1 || col > Board::kSize) {
                print_error(MoveError::OutOfBounds);
                continue;
            }
            move = Move::place(row - 1, col - 1);
        }

        const auto outcome = game.play(move);
        if (!outcome.accepted()) {
            print_error(outcome.error);
            continue;
        }
        print_board(game, console);
    }
    print_result(game);
    return true;
}

bool restart_requested() {
    std::string line;
    while (true) {
        std::cout << "\n새 게임: r 또는 새게임 / 종료: q 또는 종료 > " << std::flush;
        if (!std::getline(std::cin, line)) return false;
        const std::string command = clean_command(line);
        if (is_quit(command) || command.empty()) return false;
        if (command == "r" || command == "새게임") return true;
        std::cout << "r(새 게임) 또는 q(종료)를 입력해 주세요.\n";
    }
}

} // namespace

int main() {
    const Console console;
    try {
        std::cout << "Great Kingdom - 두 사람 직접 입력\n";
        print_help();
        while (true) {
            kingdom::State game;
            if (!play_game(game, console) || !restart_requested()) break;
            std::cout << "\n새 게임을 시작합니다.\n";
        }
        std::cout << "\n게임을 종료합니다.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "실행 오류: " << error.what() << '\n';
        return 1;
    }
}
