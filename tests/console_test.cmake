# Exercise the same console executable a person uses, with complete input sessions.
if(NOT DEFINED GAME_EXECUTABLE OR NOT EXISTS "${GAME_EXECUTABLE}")
    message(FATAL_ERROR "GAME_EXECUTABLE must identify the built console game")
endif()
if(NOT DEFINED BINARY_DIR OR BINARY_DIR STREQUAL "")
    message(FATAL_ERROR "BINARY_DIR must identify the build directory")
endif()

set(console_input_dir "${BINARY_DIR}/console-tests")
file(MAKE_DIRECTORY "${console_input_dir}")

function(run_console name input)
    set(input_path "${console_input_dir}/${name}.txt")
    file(WRITE "${input_path}" "${input}")
    execute_process(
        COMMAND "${GAME_EXECUTABLE}"
        INPUT_FILE "${input_path}"
        OUTPUT_VARIABLE output
        ERROR_VARIABLE errors
        RESULT_VARIABLE status
        TIMEOUT 10
        ENCODING UTF-8
    )
    if(NOT "${status}" STREQUAL "0")
        message(FATAL_ERROR "${name}: game did not exit cleanly (${status})\n${output}\n${errors}")
    endif()
    set("${name}_output" "${output}" PARENT_SCOPE)
endfunction()

function(expect_no_errors name output)
    foreach(error IN ITEMS
            "이미 돌이 있는 칸입니다."
            "행과 열은 1~9 사이여야 합니다."
            "입력을 확인해 주세요. 예: 3 4, pass, quit"
            "자기 집 안에는 놓을 수 없습니다."
            "상대 집 안에는 놓을 수 없습니다.")
        expect_absent("${name}" "${output}" "${error}")
    endforeach()
endfunction()

function(expect_order name output)
    set(remainder "${output}")
    foreach(expected IN LISTS ARGN)
        string(FIND "${remainder}" "${expected}" position)
        if(position EQUAL -1)
            message(FATAL_ERROR "${name}: missing or out-of-order '${expected}'\n${output}")
        endif()
        string(LENGTH "${expected}" expected_length)
        math(EXPR after_expected "${position} + ${expected_length}")
        string(SUBSTRING "${remainder}" ${after_expected} -1 remainder)
    endforeach()
endfunction()

function(expect_absent name output unexpected)
    string(FIND "${output}" "${unexpected}" position)
    if(NOT position EQUAL -1)
        message(FATAL_ERROR "${name}: unexpected '${unexpected}'\n${output}")
    endif()
endfunction()

# Invalid inputs and occupied cells must keep the current player's turn.
run_console(input_validation
    "5 5\nbad\n999999999999999999999999 1\n0 1\n10 1\n1 10\n1 1 extra\nhelp\n1 1\n1 1\n1 2\nq\n")
expect_order(input_validation "${input_validation_output}"
    "차례: 선공 (파랑/x)"
    "이미 돌이 있는 칸입니다." "차례: 선공 (파랑/x)"
    "입력을 확인해 주세요. 예: 3 4, pass, quit" "차례: 선공 (파랑/x)"
    "입력을 확인해 주세요. 예: 3 4, pass, quit" "차례: 선공 (파랑/x)"
    "행과 열은 1~9 사이여야 합니다." "차례: 선공 (파랑/x)"
    "행과 열은 1~9 사이여야 합니다." "차례: 선공 (파랑/x)"
    "행과 열은 1~9 사이여야 합니다." "차례: 선공 (파랑/x)"
    "입력을 확인해 주세요. 예: 3 4, pass, quit" "차례: 선공 (파랑/x)"
    "차례: 선공 (파랑/x)"
    "차례: 후공 (주황/o)"
    "이미 돌이 있는 칸입니다." "차례: 후공 (주황/o)"
    "차례: 선공 (파랑/x)")
expect_absent(input_validation "${input_validation_output}" "승리 (")

# Both spellings of pass finish the game; restart rebuilds the initial state.
# The final EOF also checks that closing the input stream exits cleanly.
run_console(pass_restart "1 1\npass\n패스\nr\n")
expect_order(pass_restart "${pass_restart_output}"
    " 5 | . . . . # . . . . |"
    "남은 돌: 선공 41개 / 후공 41개"
    "차례: 선공 (파랑/x)"
    "남은 돌: 선공 40개 / 후공 41개"
    "차례: 후공 (주황/o)" "차례: 선공 (파랑/x)" "후공 승리 (연속 패스)"
    "새 게임을 시작합니다."
    " 1 | . . . . . . . . . |"
    " 5 | . . . . # . . . . |"
    "남은 돌: 선공 41개 / 후공 41개" "차례: 선공 (파랑/x)")
expect_no_errors(pass_restart "${pass_restart_output}")

# A single captured stone ends the game immediately, without needing passes.
run_console(capture "2 3\n3 3\n3 2\n9 9\n4 3\n9 8\n3 4\nq\n")
expect_order(capture "${capture_output}" "차례: 선공 (파랑/x)" "선공 승리 (상대 돌 포획)")
expect_no_errors(capture "${capture_output}")

# Black can close its own last liberty; the accepted move awards White a win.
run_console(suicide
    "pass\n1 1\npass\n1 3\npass\n2 1\npass\n2 3\n1 2\n3 2\n2 2\nq\n")
expect_order(suicide "${suicide_output}" "차례: 선공 (파랑/x)" "후공 승리 (자충수)")
expect_no_errors(suicide "${suicide_output}")

# The same three-cell corner house prohibits moves by either owner.
# After the rejected own-house move, Black keeps the turn and can pass.
run_console(own_territory
    "1 3\n9 9\n2 2\n9 8\n3 1\n8 9\n1 1\npass\npass\nq\n")
expect_order(own_territory "${own_territory_output}"
    "자기 집 안에는 놓을 수 없습니다." "차례: 선공 (파랑/x)"
    "차례: 후공 (주황/o)" "선공 승리 (연속 패스)")
expect_absent(own_territory "${own_territory_output}" "상대 집 안에는 놓을 수 없습니다.")

run_console(opponent_territory "1 3\n9 9\n2 2\n9 8\n3 1\n1 1\nquit\n")
expect_order(opponent_territory "${opponent_territory_output}"
    "상대 집 안에는 놓을 수 없습니다." "차례: 후공 (주황/o)")
expect_absent(opponent_territory "${opponent_territory_output}" "승리 (")

run_console(empty_input "")
expect_order(empty_input "${empty_input_output}" "차례: 선공 (파랑/x)")
expect_absent(empty_input "${empty_input_output}" "승리 (")

message(STATUS "Console game: all scripted sessions passed")
