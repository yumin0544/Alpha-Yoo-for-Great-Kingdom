#include <iostream>
#include <array>
#include <vector>
#include <memory>
#include <cmath>
#include <random>
#include <limits>
#include <algorithm>

using namespace std;

// ============================================================
// 기본 설정
// ============================================================

constexpr int BOARD_SIZE = 9;
constexpr int MAX_CELLS = BOARD_SIZE * BOARD_SIZE;

constexpr int EMPTY = 0;
constexpr int BLACK = 1;
constexpr int WHITE = 2;
constexpr int DRAW = 3;

// 상하좌우
constexpr int DR[4] = {-1, 1, 0, 0};
constexpr int DC[4] = {0, 0, -1, 1};


// ============================================================
// 좌표 관련 함수
// ============================================================

inline bool in_board(int r, int c) {
    return r >= 0 &&
           r < BOARD_SIZE &&
           c >= 0 &&
           c < BOARD_SIZE;
}

inline int to_index(int r, int c) {
    return r * BOARD_SIZE + c;
}

inline int get_opponent(int player) {
    return (player == BLACK) ? WHITE : BLACK;
}


// ============================================================
// GameState
// ============================================================

class GameState {
public:

    // 9x9 = 81칸
    array<int, MAX_CELLS> board{};

    // 현재 차례
    int current_player = BLACK;

    // 0 : 진행 중
    // 1 : 흑 승
    // 2 : 백 승
    // 3 : 무승부
    int winner = EMPTY;


    GameState() {
        board.fill(EMPTY);
    }


    // --------------------------------------------------------
    // 특정 돌 그룹 전체 찾기
    // --------------------------------------------------------

    vector<int> get_group(int start_idx, int player) const {

        vector<int> group;

        if (start_idx < 0 ||
            start_idx >= MAX_CELLS ||
            board[start_idx] != player) {
            return group;
        }

        array<bool, MAX_CELLS> visited{};

        vector<int> stack;
        stack.push_back(start_idx);

        visited[start_idx] = true;

        while (!stack.empty()) {

            int current = stack.back();
            stack.pop_back();

            group.push_back(current);

            int r = current / BOARD_SIZE;
            int c = current % BOARD_SIZE;

            for (int d = 0; d < 4; d++) {

                int nr = r + DR[d];
                int nc = c + DC[d];

                if (!in_board(nr, nc))
                    continue;

                int next = to_index(nr, nc);

                if (board[next] == player &&
                    !visited[next]) {

                    visited[next] = true;
                    stack.push_back(next);
                }
            }
        }

        return group;
    }


    // --------------------------------------------------------
    // 돌 그룹에 자유도가 있는가?
    // --------------------------------------------------------

    bool has_liberties(int start_idx, int player) const {

        if (board[start_idx] != player)
            return false;

        array<bool, MAX_CELLS> visited{};

        vector<int> stack;
        stack.push_back(start_idx);

        visited[start_idx] = true;

        while (!stack.empty()) {

            int current = stack.back();
            stack.pop_back();

            int r = current / BOARD_SIZE;
            int c = current % BOARD_SIZE;

            for (int d = 0; d < 4; d++) {

                int nr = r + DR[d];
                int nc = c + DC[d];

                if (!in_board(nr, nc))
                    continue;

                int next = to_index(nr, nc);

                // 빈 공간 발견
                if (board[next] == EMPTY) {
                    return true;
                }

                // 같은 그룹
                if (board[next] == player &&
                    !visited[next]) {

                    visited[next] = true;
                    stack.push_back(next);
                }
            }
        }

        // 자유도가 전혀 없음
        return false;
    }


    // --------------------------------------------------------
    // 해당 빈칸이 집인가?
    //
    // 정의:
    // 해당 빈칸의 상하좌우가
    // 모두 같은 플레이어의 돌 또는 보드 경계이면 집
    //
    // 예:
    //
    // X X
    // X .
    //
    // 오른쪽 아래 '.'은 BLACK의 집
    // --------------------------------------------------------

    bool is_house_cell(int cell, int player) const {

        if (cell < 0 ||
            cell >= MAX_CELLS ||
            board[cell] != EMPTY) {
            return false;
        }

        int r = cell / BOARD_SIZE;
        int c = cell % BOARD_SIZE;

        for (int d = 0; d < 4; d++) {

            int nr = r + DR[d];
            int nc = c + DC[d];

            // 보드 밖 = 벽
            if (!in_board(nr, nc)) {
                continue;
            }

            int next = to_index(nr, nc);

            // 내부 칸인데 내 돌이 아니면 집이 아님
            if (board[next] != player) {
                return false;
            }
        }

        return true;
    }


    // --------------------------------------------------------
    // 현재 보드의 모든 집을 계산
    //
    // house_owner[i]
    //
    // 0 = 집 아님
    // 1 = 흑의 집
    // 2 = 백의 집
    // --------------------------------------------------------

    array<int, MAX_CELLS> get_house_map() const {

        array<int, MAX_CELLS> house_owner{};
        house_owner.fill(EMPTY);

        for (int i = 0; i < MAX_CELLS; i++) {

            if (board[i] != EMPTY)
                continue;

            bool black_house = is_house_cell(i, BLACK);
            bool white_house = is_house_cell(i, WHITE);

            if (black_house) {
                house_owner[i] = BLACK;
            }
            else if (white_house) {
                house_owner[i] = WHITE;
            }
        }

        return house_owner;
    }


    // --------------------------------------------------------
    // 집 개수 계산
    // --------------------------------------------------------

    int count_house(int player) const {

        auto house_map = get_house_map();

        int count = 0;

        for (int i = 0; i < MAX_CELLS; i++) {

            if (house_map[i] == player) {
                count++;
            }
        }

        return count;
    }


    // --------------------------------------------------------
    // 최종 점수 계산
    // --------------------------------------------------------

    int calculate_winner() const {

        int black_house = count_house(BLACK);
        int white_house = count_house(WHITE);
        // 백은 흑보다 2.5집의 추가점수를 갖는다.
        white_house += 2;

        if (black_house > white_house)
            return BLACK;
        return WHITE;
    }


    // --------------------------------------------------------
    // 특정 착수가 상대 돌을 둘러싸는지 검사
    // --------------------------------------------------------

    bool captures_enemy(int move) const {

        int opponent = get_opponent(current_player);

        int r = move / BOARD_SIZE;
        int c = move % BOARD_SIZE;

        for (int d = 0; d < 4; d++) {

            int nr = r + DR[d];
            int nc = c + DC[d];

            if (!in_board(nr, nc))
                continue;

            int next = to_index(nr, nc);

            if (board[next] == opponent) {

                // 상대 그룹의 자유도가 0이면
                // 상대가 둘러싸임
                if (!has_liberties(next, opponent)) {
                    return true;
                }
            }
        }

        return false;
    }


    // --------------------------------------------------------
    // 착수 합법성 검사
    // --------------------------------------------------------

    bool is_legal_move(int move) const {

        if (winner != EMPTY)
            return false;

        if (move < 0 || move >= MAX_CELLS)
            return false;

        // 이미 돌이 있음
        if (board[move] != EMPTY)
            return false;


        // ----------------------------------------------------
        // 집은 착수 금지
        // ----------------------------------------------------

        if (is_house_cell(move, BLACK) ||
            is_house_cell(move, WHITE)) {

            return false;
        }


        // ----------------------------------------------------
        // 실제로 돌을 놓아 본다
        // ----------------------------------------------------

        GameState temp = *this;

        temp.board[move] = current_player;


        // 상대를 죽일 수 있다면 합법
        if (temp.captures_enemy(move)) {
            return true;
        }


        // ----------------------------------------------------
        // 자기 돌이 자유도를 잃는 경우
        // 자살수 -> 가지치기
        // ----------------------------------------------------

        if (!temp.has_liberties(move, current_player)) {
            return false;
        }

        return true;
    }


    // --------------------------------------------------------
    // 현재 가능한 모든 착수
    // --------------------------------------------------------

    vector<int> get_legal_moves() const {

        vector<int> moves;

        if (winner != EMPTY)
            return moves;

        for (int i = 0; i < MAX_CELLS; i++) {

            if (is_legal_move(i)) {
                moves.push_back(i);
            }
        }

        return moves;
    }


    // --------------------------------------------------------
    // 실제 착수
    //
    // 반환:
    // true  = 정상적으로 착수
    // false = 불법 착수
    // --------------------------------------------------------

    bool make_move(int move) {

        if (!is_legal_move(move))
            return false;


        int player = current_player;

        // 돌 배치
        board[move] = player;


        // ----------------------------------------------------
        // 상대를 둘러쌌다면 즉시 승리
        // ----------------------------------------------------

        if (captures_enemy(move)) {

            winner = player;

            return true;
        }


        // 다음 플레이어
        current_player = get_opponent(player);

        return true;
    }


    // --------------------------------------------------------
    // 더 이상 둘 수 없으면 집 계산
    // --------------------------------------------------------

    void check_game_end() {

        if (winner != EMPTY)
            return;

        auto legal_moves = get_legal_moves();

        if (legal_moves.empty()) {

            winner = calculate_winner();
        }
    }


    // --------------------------------------------------------
    // 무작위 플레이아웃
    // --------------------------------------------------------

    int random_playout(mt19937& rng,
                       int max_turns = 200) const {

        GameState temp = *this;

        for (int turn = 0;
             turn < max_turns && temp.winner == EMPTY;
             turn++) {

            vector<int> moves =
                temp.get_legal_moves();


            // 더 이상 둘 곳이 없음
            if (moves.empty()) {

                temp.winner =
                    temp.calculate_winner();

                break;
            }


            // 무작위 수 선택
            uniform_int_distribution<int> dist(
                0,
                static_cast<int>(moves.size()) - 1
            );

            int move = moves[dist(rng)];

            temp.make_move(move);
        }


        // 안전장치
        if (temp.winner == EMPTY) {

            temp.winner =
                temp.calculate_winner();
        }

        return temp.winner;
    }


    // --------------------------------------------------------
    // 디버깅용 보드 출력
    //
    // X = BLACK
    // O = WHITE
    // H = 집
    // . = 빈칸
    // --------------------------------------------------------

    void print_board() const {

        auto house_map = get_house_map();

        cout << "\n   ";

        for (int c = 0; c < BOARD_SIZE; c++) {
            cout << c << ' ';
        }

        cout << '\n';


        for (int r = 0; r < BOARD_SIZE; r++) {

            cout << r << "  ";

            for (int c = 0; c < BOARD_SIZE; c++) {

                int cell = to_index(r, c);

                char symbol = '.';

                if (board[cell] == BLACK) {
                    symbol = 'X';
                }
                else if (board[cell] == WHITE) {
                    symbol = 'O';
                }
                else if (house_map[cell] == BLACK ||
                         house_map[cell] == WHITE) {
                    symbol = 'H';
                }

                cout << symbol << ' ';
            }

            cout << '\n';
        }


        cout << "\n현재 차례 : ";

        if (current_player == BLACK)
            cout << "BLACK";

        else
            cout << "WHITE";


        cout << '\n';

        cout << "BLACK 집 : "
             << count_house(BLACK)
             << '\n';

        cout << "WHITE 집 : "
             << count_house(WHITE)
             << '\n';
    }
};


// ============================================================
// MCTS Node
// ============================================================

class MCTSNode {

public:

    GameState state;

    MCTSNode* parent;

    vector<unique_ptr<MCTSNode>> children;

    vector<int> untried_moves;

    // 이 노드로 오는 착수
    int move_from_parent;

    // 해당 노드의 방문 횟수
    int visits;

    // 승리 횟수
    double wins;


    MCTSNode(
        const GameState& state,
        MCTSNode* parent,
        int move
    )
        : state(state),
          parent(parent),
          move_from_parent(move),
          visits(0),
          wins(0.0) {

        untried_moves =
            state.get_legal_moves();
    }


    // --------------------------------------------------------
    // 이 노드에서 '방금 수를 둔 사람'
    // --------------------------------------------------------

    int player_just_moved() const {

        return get_opponent(state.current_player);
    }


    // --------------------------------------------------------
    // UCT / UCB1
    // --------------------------------------------------------

    MCTSNode* select_child() {

        MCTSNode* best_child = nullptr;

        double best_score =
            -numeric_limits<double>::infinity();


        for (auto& child_ptr : children) {

            MCTSNode* child =
                child_ptr.get();


            if (child->visits == 0) {
                return child;
            }


            // exploitation
            double exploitation =
                child->wins /
                static_cast<double>(child->visits);


            // exploration
            double exploration =
                sqrt(
                    2.0 *
                    log(
                        static_cast<double>(visits)
                    ) /
                    static_cast<double>(
                        child->visits
                    )
                );


            double score =
                exploitation + exploration;


            if (score > best_score) {

                best_score = score;
                best_child = child;
            }
        }

        return best_child;
    }


    // --------------------------------------------------------
    // Expansion
    // --------------------------------------------------------

    MCTSNode* expand() {

        if (untried_moves.empty()) {
            return nullptr;
        }


        // 하나 꺼냄
        int move = untried_moves.back();

        untried_moves.pop_back();


        // 다음 상태 생성
        GameState next_state = state;

        bool success =
            next_state.make_move(move);


        if (!success) {
            return nullptr;
        }


        children.push_back(
            make_unique<MCTSNode>(
                next_state,
                this,
                move
            )
        );


        return children.back().get();
    }


    // --------------------------------------------------------
    // Backpropagation
    // --------------------------------------------------------

    void backpropagate(int result) {

        visits++;


        int just_moved =
            player_just_moved();


        if (result == just_moved) {

            wins += 1.0;
        }
        else if (result == DRAW) {

            wins += 0.5;
        }


        if (parent != nullptr) {

            parent->backpropagate(result);
        }
    }
};


// ============================================================
// MCTS
// ============================================================

class MCTS {

private:

    mt19937 rng;


public:

    MCTS()
        : rng(random_device{}()) {
    }


    // --------------------------------------------------------
    // MCTS 검색
    // --------------------------------------------------------

    int search(
        const GameState& root_state,
        int iterations
    ) {

        MCTSNode root(
            root_state,
            nullptr,
            -1
        );


        if (root.untried_moves.empty()) {
            return -1;
        }


        // ====================================================
        // 반복
        // ====================================================

        for (int i = 0;
             i < iterations;
             i++) {

            MCTSNode* node = &root;


            // ------------------------------------------------
            // 1. Selection
            // ------------------------------------------------

            while (
                node->untried_moves.empty() &&
                !node->children.empty()
            ) {

                node =
                    node->select_child();
            }


            // ------------------------------------------------
            // 2. Expansion
            // ------------------------------------------------

            if (!node->untried_moves.empty()) {

                MCTSNode* expanded =
                    node->expand();

                if (expanded != nullptr) {
                    node = expanded;
                }
            }


            // ------------------------------------------------
            // 3. Simulation
            // ------------------------------------------------

            int result =
                node->state.random_playout(
                    rng
                );


            // ------------------------------------------------
            // 4. Backpropagation
            // ------------------------------------------------

            node->backpropagate(result);
        }


        // ====================================================
        // 최종 선택
        //
        // 가장 많은 방문 횟수를 가진 착수 선택
        // ====================================================

        MCTSNode* best_child =
            nullptr;

        int max_visits = -1;


        for (auto& child_ptr : root.children) {

            MCTSNode* child =
                child_ptr.get();


            if (child->visits > max_visits) {

                max_visits =
                    child->visits;

                best_child = child;
            }
        }


        if (best_child == nullptr) {
            return -1;
        }


        return best_child->move_from_parent;
    }
};


// ============================================================
// main
// ============================================================

int main() {

    ios::sync_with_stdio(false);
    cin.tie(nullptr);


    GameState game;

    MCTS mcts;


    cout << "====================================\n";
    cout << "      9x9 Board Game Engine\n";
    cout << "====================================\n";


    game.print_board();


    // --------------------------------------------------------
    // AI 첫 수 계산
    // --------------------------------------------------------

    const int ITERATIONS = 10000;


    cout << "\nMCTS "
         << ITERATIONS
         << "회 탐색 중...\n";


    int best_move =
        mcts.search(
            game,
            ITERATIONS
        );


    if (best_move == -1) {

        cout << "둘 수 있는 곳이 없습니다.\n";

        return 0;
    }


    int row =
        best_move / BOARD_SIZE;

    int col =
        best_move % BOARD_SIZE;


    cout << "\nAI 추천 수 : ("
         << row
         << ", "
         << col
         << ")\n";


    // 실제 착수
    game.make_move(best_move);


    game.print_board();


    return 0;
}