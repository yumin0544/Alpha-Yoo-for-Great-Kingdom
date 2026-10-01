#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <memory>

using namespace std;

const int BOARD_SIZE = 9;
const int MAX_CELLS = BOARD_SIZE * BOARD_SIZE;
const int EMPTY = 0;
const int BLACK = 1;
const int WHITE = 2;

// 상하좌우 방향 벡터
const int dr[] = {-1, 1, 0, 0};
const int dc[] = {0, 0, -1, 1};

class GameState {
public:
    vector<int> board;
    int current_player;
    int winner; // 0: 진행중, 1: 흑 승, 2: 백 승, 3: 무승부(집 계산)

    GameState() {
        board = vector<int>(MAX_CELLS, EMPTY);
        current_player = BLACK;
        winner = 0;
    }

    // 적 상대에게 완전히 변경(반전)되는 플레이어 반환
    int get_opponent() const {
        return (current_player == BLACK) ? WHITE : BLACK;
    }

    // 특정 돌 그룹이 숨 쉴 곳(자유도)이 있는지 판별 (Flood Fill)
    bool has_liberties(int start_idx, int player) const {
        vector<bool> visited(MAX_CELLS, false);
        vector<int> stack;
        
        stack.push_back(start_idx);
        visited[start_idx] = true;

        while (!stack.empty()) {
            int curr = stack.back();
            stack.pop_back();

            int r = curr / BOARD_SIZE;
            int c = curr % BOARD_SIZE;

            for (int i = 0; i < 4; ++i) {
                int nr = r + dr[i];
                int nc = c + dc[i];
                
                if (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE) {
                    int n_idx = nr * BOARD_SIZE + nc;
                    if (board[n_idx] == EMPTY) {
                        return true; // 빈 공간이 하나라도 있으면 생존
                    }
                    if (board[n_idx] == player && !visited[n_idx]) {
                        visited[n_idx] = true;
                        stack.push_back(n_idx);
                    }
                }
            }
        }
        return false; // 빈 공간이 전혀 없음 (둘러싸임)
    }

    // 완벽한 집(진짜 눈, Eye)인지 확인하여 착수 금지 구역으로 설정
    bool is_eye(int idx, int player) const {
        int r = idx / BOARD_SIZE;
        int c = idx % BOARD_SIZE;
        
        for (int i = 0; i < 4; ++i) {
            int nr = r + dr[i];
            int nc = c + dc[i];
            if (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE) {
                int n_idx = nr * BOARD_SIZE + nc;
                if (board[n_idx] != player) return false;
            }
        }
        return true; // 상하좌우가 모두 내 돌이거나 벽인 경우
    }

    // 현재 상태에서 둘 수 있는 모든 합법적인 수 반환
    vector<int> get_legal_moves() const {
        vector<int> moves;
        if (winner != 0) return moves; // 게임 종료 상태

        for (int i = 0; i < MAX_CELLS; ++i) {
            if (board[i] == EMPTY) {
                // 1. 자신의 완벽한 집(Eye)에는 두지 않음 (가지치기 및 룰)
                if (is_eye(i, current_player)) continue;

                // 2. 자살수 방지 (단, 놓아서 적을 죽이는 경우는 허용)
                // 엔진 가속을 위해 딥 복사 대신 가상으로 놓아보고 검사
                GameState temp_state = *this;
                temp_state.board[i] = current_player;
                
                bool kills_enemy = false;
                int r = i / BOARD_SIZE;
                int c = i % BOARD_SIZE;
                
                for (int d = 0; d < 4; ++d) {
                    int nr = r + dr[d];
                    int nc = c + dc[d];
                    if (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE) {
                        int n_idx = nr * BOARD_SIZE + nc;
                        if (temp_state.board[n_idx] == get_opponent()) {
                            if (!temp_state.has_liberties(n_idx, get_opponent())) {
                                kills_enemy = true;
                                break;
                            }
                        }
                    }
                }

                if (!kills_enemy && !temp_state.has_liberties(i, current_player)) {
                    continue; // 적을 죽이지도 못하는데 내가 죽는 자리 (자살수)
                }

                moves.push_back(i);
            }
        }
        return moves;
    }

    // 돌을 두고 승패를 판정하는 핵심 로직
    void make_move(int move) {
        board[move] = current_player;
        int opponent = get_opponent();
        bool enemy_died = false;

        // 1. 적이 둘러싸였는지 (서든데스) 체크
        int r = move / BOARD_SIZE;
        int c = move % BOARD_SIZE;
        for (int d = 0; d < 4; ++d) {
            int nr = r + dr[d];
            int nc = c + dc[d];
            if (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE) {
                int n_idx = nr * BOARD_SIZE + nc;
                if (board[n_idx] == opponent && !has_liberties(n_idx, opponent)) {
                    enemy_died = true;
                    break;
                }
            }
        }

        if (enemy_died) {
            winner = current_player; // 적을 둘러쌌으므로 즉시 승리
            return;
        }

        current_player = opponent; // 턴 넘기기
    }

    // 무작위 플레이아웃 (시뮬레이션)
    int playout() {
        GameState temp = *this;
        int turn_limit = 150; // 무한 루프 방지
        
        while (temp.winner == 0 && turn_limit > 0) {
            vector<int> moves = temp.get_legal_moves();
            if (moves.empty()) {
                break; // 양측 모두 둘 곳이 없으면 (영토 굳어짐) 종료
            }
            int random_move = moves[rand() % moves.size()];
            temp.make_move(random_move);
            turn_limit--;
        }

        // 서든데스로 승부가 안 났다면 집(돌 갯수) 카운팅
        if (temp.winner == 0) {
            int black_score = 0, white_score = 0;
            for (int cell : temp.board) {
                if (cell == BLACK) black_score++;
                else if (cell == WHITE) white_score++;
            }
            if (black_score > white_score) temp.winner = BLACK;
            else if (white_score > black_score) temp.winner = WHITE;
            else temp.winner = 3; // 무승부
        }
        return temp.winner;
    }
};

// MCTS 노드
class MCTSNode {
public:
    GameState state;
    MCTSNode* parent;
    vector<unique_ptr<MCTSNode>> children;
    int move_from_parent;
    double wins;
    int visits;
    vector<int> untried_moves;

    MCTSNode(GameState s, MCTSNode* p, int move) : state(s), parent(p), move_from_parent(move), wins(0), visits(0) {
        untried_moves = state.get_legal_moves();
    }

    // UCB1 알고리즘을 이용한 자식 노드 선택
    MCTSNode* uct_select_child() {
        MCTSNode* best_child = nullptr;
        double best_score = -1e9;

        for (auto& child : children) {
            double exploit = child->wins / child->visits;
            double explore = sqrt(2.0 * log(visits) / child->visits);
            double score = exploit + explore;

            if (score > best_score) {
                best_score = score;
                best_child = child.get();
            }
        }
        return best_child;
    }

    // 새로운 노드 확장
    MCTSNode* expand() {
        int move = untried_moves.back();
        untried_moves.pop_back();

        GameState next_state = state;
        next_state.make_move(move);

        children.push_back(make_unique<MCTSNode>(next_state, this, move));
        return children.back().get();
    }

    // 결과 역전파
    void backpropagate(int result_winner) {
        visits++;
        // 부모 노드의 플레이어 입장에서 이겼는지 확인
        if (parent != nullptr) {
            if (result_winner == parent->state.current_player) {
                wins += 1.0;
            } else if (result_winner == 3) {
                wins += 0.5; // 무승부
            }
        }
        if (parent != nullptr) {
            parent->backpropagate(result_winner);
        }
    }
};

// MCTS 실행 함수
int get_best_move(const GameState& root_state, int itermax) {
    MCTSNode root(root_state, nullptr, -1);

    for (int i = 0; i < itermax; ++i) {
        MCTSNode* node = &root;

        // 1. Selection
        while (node->untried_moves.empty() && !node->children.empty()) {
            node = node->uct_select_child();
        }

        // 2. Expansion
        if (!node->untried_moves.empty()) {
            node = node->expand();
        }

        // 3. Simulation
        int result = node->state.playout();

        // 4. Backpropagation
        node->backpropagate(result);
    }

    // 방문 횟수가 가장 많은 수가 최적의 수
    int best_move = -1;
    int max_visits = -1;
    for (auto& child : root.children) {
        if (child->visits > max_visits) {
            max_visits = child->visits;
            best_move = child->move_from_parent;
        }
    }
    return best_move;
}

int main() {
    srand(time(NULL));
    GameState game;

    cout << "엔진 초기화 완료. AI 탐색을 시작합니다..." << endl;

    // AI가 흑(1)으로서 첫 수를 고민 (시뮬레이션 10,000번)
    int best_move = get_best_move(game, 10000);
    
    int r = best_move / BOARD_SIZE;
    int c = best_move % BOARD_SIZE;
    
    cout << "AI의 최적의 첫 수: (" << r << ", " << c << ")" << endl;
    
    return 0;
}