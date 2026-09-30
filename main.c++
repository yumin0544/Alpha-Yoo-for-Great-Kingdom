#include <iostream>

int dime_2_to_bit(const int p, const int q);
bool is_legal(const int player_A, const int player_B, const int house_A, const int house_B);

int main() {
    int player_A = 0;
    int player_B = 0;
    int house_A = 0;
    int house_B = 0;

    bool is_over = false;
    bool is_A_turn = true;
    int pass_cnt = 0;
    int p,q;
    while(not is_over){
        std::cin >> p >> q;
        //입력 제한
        if(p>9 || q>9){
            std::cout <<  "다시 입력해주세요\n";
            continue;
        }
        if(not is_legal(player_A,player_B,house_A,house_B)){
            std::cout <<  "다시 입력해주세요\n";
            continue;
        }
        
        //종료조건
        if(pass_cnt >= 2)
            is_over = true;
    }
    std::cin >> p >> q;

    return 0;
}

int dime_2_to_bit(const int p, const int q){
    return (p*9)+q;
}

bool is_legal(const int player_A, const int player_B, const int house_A, const int house_B){
    // out of bound
    if(player_A >= 81 || player_B >= 81) return false;
    //겹치는 부분이 있으면 false 처리
    if(player_A & player_B) return false;
    //상대의 집 안에 두었으면 false 처리
    if(player_A & house_B || player_B & house_A) return false;
    return true;
}