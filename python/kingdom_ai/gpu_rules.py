"""Independent CUDA rules for the confirmed game, with a read-only CPU oracle.

The state, transitions, legal masks and observation planes stay on the GPU.
``from_engine`` is an explicit debug/test import boundary, and ``snapshot`` is
an explicit device-to-host debug boundary. Neither is used inside GPU search.
The verified ``engine/`` implementation is not changed or called by kernels.
"""

from __future__ import annotations

from threading import Lock

import torch


GPU_STATE_WIDTH = 170
GPU_CELL_COUNT = 81
GPU_ACTION_SIZE = 82
GPU_ACTOR = 162
GPU_BLACK_STOCK = 163
GPU_WHITE_STOCK = 164
GPU_PASSES = 165
GPU_REASON = 166
GPU_WINNER = 167
GPU_CAPTURED = 168
GPU_RESERVED = 169

# The independent wire mapping deliberately uses named CPU enums when importing
# snapshots: CPU EndReason orders TwoPasses before Suicide.
GPU_NONE = 0
GPU_CAPTURE = 1
GPU_SUICIDE = 2
GPU_TWO_PASSES = 3


RULES_CUDA_SOURCE = r"""
#define GPU_STATE_WIDTH 170
#define GPU_CELL_COUNT 81
#define GPU_ACTION_SIZE 82
#define GPU_ACTOR 162
#define GPU_BLACK_STOCK 163
#define GPU_WHITE_STOCK 164
#define GPU_PASSES 165
#define GPU_REASON 166
#define GPU_WINNER 167
#define GPU_CAPTURED 168
#define GPU_RESERVED 169
#define GPU_NONE 0
#define GPU_CAPTURE 1
#define GPU_SUICIDE 2
#define GPU_TWO_PASSES 3

// One CUDA thread owns each state. All flood-fill queues have a fixed 81-cell
// bound, with each cell visited at most once; no recursion or host containers.
__device__ __forceinline__ int gpu_opponent(int player) {
    return 3 - player;
}

__device__ __forceinline__ int gpu_neighbours(int point, int* neighbours) {
    const int row = point / 9;
    const int col = point % 9;
    int count = 0;
    if (row > 0) neighbours[count++] = point - 9;
    if (row < 8) neighbours[count++] = point + 9;
    if (col > 0) neighbours[count++] = point - 1;
    if (col < 8) neighbours[count++] = point + 1;
    return count;
}

__device__ __forceinline__ unsigned int gpu_edges(int point) {
    const int row = point / 9;
    const int col = point % 9;
    return (row == 0 ? 1u : 0u) | (row == 8 ? 2u : 0u)
         | (col == 0 ? 4u : 0u) | (col == 8 ? 8u : 0u);
}

__device__ int gpu_collect_group(const int* cells, int start, int* queue) {
    const int player = cells[start];
    if (player != 1 && player != 2) return 0;
    bool visited[GPU_CELL_COUNT];
    for (int i = 0; i < GPU_CELL_COUNT; ++i) visited[i] = false;
    int length = 1;
    queue[0] = start;
    visited[start] = true;
    for (int cursor = 0; cursor < length; ++cursor) {
        int neighbours[4];
        const int count = gpu_neighbours(queue[cursor], neighbours);
        for (int n = 0; n < count; ++n) {
            const int next = neighbours[n];
            if (!visited[next] && cells[next] == player) {
                visited[next] = true;
                queue[length++] = next;
            }
        }
    }
    return length;
}

__device__ bool gpu_group_has_liberty(
    const int* cells, const int* group, int length
) {
    for (int i = 0; i < length; ++i) {
        int neighbours[4];
        const int count = gpu_neighbours(group[i], neighbours);
        for (int n = 0; n < count; ++n) {
            if (cells[neighbours[n]] == 0) return true;
        }
    }
    return false;
}

__device__ void gpu_score(const int* state, int* black, int* white) {
    int black_score = 0;
    int white_score = 0;
    for (int i = 0; i < GPU_CELL_COUNT; ++i) {
        if (state[i] != 0) continue;
        black_score += state[GPU_CELL_COUNT + i] == 1;
        white_score += state[GPU_CELL_COUNT + i] == 2;
    }
    *black = black_score;
    *white = white_score;
}

__device__ void gpu_claim_territory(int* state) {
    int detected[GPU_CELL_COUNT];
    for (int i = 0; i < GPU_CELL_COUNT; ++i) detected[i] = 0;
    // Board::territory traverses empty AND enemy cells; neutral and own stones
    // are walls. Enemy-containing components are rejected after the traversal.
    // Reproduce both player passes before updating the permanent ownership.
    for (int player = 1; player <= 2; ++player) {
        const int enemy = gpu_opponent(player);
        bool visited[GPU_CELL_COUNT];
        int queue[GPU_CELL_COUNT];
        for (int i = 0; i < GPU_CELL_COUNT; ++i) visited[i] = false;
        for (int start = 0; start < GPU_CELL_COUNT; ++start) {
            if (visited[start] || state[start] != 0) continue;
            int length = 1;
            queue[0] = start;
            visited[start] = true;
            bool contains_enemy = false;
            bool has_own_boundary = false;
            unsigned int edges = 0;
            for (int cursor = 0; cursor < length; ++cursor) {
                const int current = queue[cursor];
                edges |= gpu_edges(current);
                contains_enemy |= state[current] == enemy;
                int neighbours[4];
                const int count = gpu_neighbours(current, neighbours);
                for (int n = 0; n < count; ++n) {
                    const int next = neighbours[n];
                    if (state[next] == player) {
                        has_own_boundary = true;
                    } else if (state[next] != 3 && !visited[next]) {
                        visited[next] = true;
                        queue[length++] = next;
                    }
                }
            }
            // One, two or three board edges are allowed; four are rejected.
            if (contains_enemy || !has_own_boundary || edges == 15u) continue;
            for (int i = 0; i < length; ++i) {
                const int point = queue[i];
                if (state[point] == 0) detected[point] = player;
            }
        }
    }
    for (int i = 0; i < GPU_CELL_COUNT; ++i) {
        if (state[GPU_CELL_COUNT + i] == 0 && detected[i] != 0) {
            state[GPU_CELL_COUNT + i] = detected[i];
        }
    }
}

__device__ __forceinline__ bool gpu_legal(const int* state, int action) {
    if (state[GPU_REASON] != GPU_NONE || action < 0 || action >= GPU_ACTION_SIZE)
        return false;
    if (action == GPU_CELL_COUNT) return true;
    // Under the confirmed suicide-allowed rules, every empty unowned placement
    // with stock is accepted; a self-capture is a terminal loss, not rejection.
    const int stock_index = state[GPU_ACTOR] == 1 ? GPU_BLACK_STOCK : GPU_WHITE_STOCK;
    return state[action] == 0 && state[GPU_CELL_COUNT + action] == 0
        && state[stock_index] > 0;
}

__device__ bool gpu_play(int* state, int action) {
    if (!gpu_legal(state, action)) return false;
    const int actor = state[GPU_ACTOR];
    const int enemy = gpu_opponent(actor);
    if (action == GPU_CELL_COUNT) {
        ++state[GPU_PASSES];
        state[GPU_ACTOR] = enemy;
        if (state[GPU_PASSES] == 2) {
            int black, white;
            gpu_score(state, &black, &white);
            state[GPU_REASON] = GPU_TWO_PASSES;
            state[GPU_WINNER] = black - white >= 3 ? 1 : 2;
            state[GPU_CAPTURED] = 0;
        }
        return true;
    }

    state[action] = actor;
    bool examined[GPU_CELL_COUNT];
    bool captured[GPU_CELL_COUNT];
    for (int i = 0; i < GPU_CELL_COUNT; ++i) {
        examined[i] = false;
        captured[i] = false;
    }
    int captured_count = 0;
    int neighbours[4];
    const int count = gpu_neighbours(action, neighbours);
    int group[GPU_CELL_COUNT];
    for (int n = 0; n < count; ++n) {
        const int next = neighbours[n];
        if (state[next] != enemy || examined[next]) continue;
        const int length = gpu_collect_group(state, next, group);
        const bool surrounded = !gpu_group_has_liberty(state, group, length);
        for (int i = 0; i < length; ++i) {
            const int point = group[i];
            examined[point] = true;
            if (surrounded) {
                captured[point] = true;
                ++captured_count;
            }
        }
    }
    bool suicide = false;
    // Capture has priority, including simultaneous loss of both liberties.
    if (captured_count == 0) {
        const int length = gpu_collect_group(state, action, group);
        suicide = !gpu_group_has_liberty(state, group, length);
    }
    --state[actor == 1 ? GPU_BLACK_STOCK : GPU_WHITE_STOCK];
    state[GPU_PASSES] = 0;
    state[GPU_ACTOR] = enemy;
    if (captured_count > 0) {
        for (int i = 0; i < GPU_CELL_COUNT; ++i) {
            if (captured[i]) state[i] = 0;
        }
        state[GPU_REASON] = GPU_CAPTURE;
        state[GPU_WINNER] = actor;
        state[GPU_CAPTURED] = captured_count;
        // A terminal capture does not claim additional territory.
    } else if (suicide) {
        state[GPU_REASON] = GPU_SUICIDE;
        state[GPU_WINNER] = enemy;
        state[GPU_CAPTURED] = 0;
    } else {
        gpu_claim_territory(state);
    }
    return true;
}

__device__ __forceinline__ float gpu_terminal_value(const int* state) {
    return state[GPU_WINNER] == state[GPU_ACTOR] ? 1.0f : -1.0f;
}

__device__ void gpu_encode(const int* state, float* features, bool* mask) {
    const int actor = state[GPU_ACTOR];
    const int enemy = gpu_opponent(actor);
    const float own_stock = __fdiv_rn(
        (float)state[actor == 1 ? GPU_BLACK_STOCK : GPU_WHITE_STOCK], 41.0f
    );
    const float opponent_stock = __fdiv_rn(
        (float)state[actor == 1 ? GPU_WHITE_STOCK : GPU_BLACK_STOCK], 41.0f
    );
    for (int action = 0; action < GPU_ACTION_SIZE; ++action) {
        mask[action] = gpu_legal(state, action);
    }
    for (int i = 0; i < GPU_CELL_COUNT; ++i) {
        features[0 * GPU_CELL_COUNT + i] = state[i] == actor ? 1.0f : 0.0f;
        features[1 * GPU_CELL_COUNT + i] = state[i] == enemy ? 1.0f : 0.0f;
        features[2 * GPU_CELL_COUNT + i] = state[i] == 3 ? 1.0f : 0.0f;
        features[3 * GPU_CELL_COUNT + i] = state[GPU_CELL_COUNT + i] == actor ? 1.0f : 0.0f;
        features[4 * GPU_CELL_COUNT + i] = state[GPU_CELL_COUNT + i] == enemy ? 1.0f : 0.0f;
        features[5 * GPU_CELL_COUNT + i] = actor == 1 ? 1.0f : 0.0f;
        features[6 * GPU_CELL_COUNT + i] = own_stock;
        features[7 * GPU_CELL_COUNT + i] = opponent_stock;
        features[8 * GPU_CELL_COUNT + i] = (float)state[GPU_PASSES] * 0.5f;
        features[9 * GPU_CELL_COUNT + i] = mask[i] ? 1.0f : 0.0f;
    }
}

extern "C" __global__ void gpu_initial_batch(int* states, int n, int neutral) {
    const int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= n) return;
    int* state = states + row * GPU_STATE_WIDTH;
    for (int i = 0; i < GPU_STATE_WIDTH; ++i) state[i] = 0;
    if (neutral >= 0 && neutral < GPU_CELL_COUNT) state[neutral] = 3;
    state[GPU_ACTOR] = 1;
    state[GPU_BLACK_STOCK] = 41;
    state[GPU_WHITE_STOCK] = 41;
}

extern "C" __global__ void gpu_play_batch(
    int* states, const int* actions, bool* accepted, int n
) {
    const int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= n) return;
    accepted[row] = gpu_play(states + row * GPU_STATE_WIDTH, actions[row]);
}

extern "C" __global__ void gpu_encode_batch(
    const int* states, float* features, bool* masks, int n
) {
    const int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= n) return;
    gpu_encode(states + row * GPU_STATE_WIDTH,
               features + row * 10 * GPU_CELL_COUNT,
               masks + row * GPU_ACTION_SIZE);
}
"""


_modules = {}
_module_lock = Lock()


def _rules_module(device):
    # NVRTC compilation is lazy, and separate CUDA devices own separate modules.
    from .gpu_runtime import CudaModule
    key = torch.device(device).index
    if key is None:
        key = torch.cuda.current_device()
    with _module_lock:
        if key not in _modules:
            with torch.cuda.device(key):
                _modules[key] = CudaModule(RULES_CUDA_SOURCE)
        return _modules[key]


class GpuStateBatch:
    """CUDA int32 ``[N, 170]`` states using the confirmed default rules."""

    def __init__(self, states: torch.Tensor):
        if not isinstance(states, torch.Tensor):
            raise TypeError("states must be a torch Tensor")
        if states.device.type != "cuda" or states.dtype != torch.int32:
            raise ValueError("states must be CUDA int32 tensors")
        if states.ndim != 2 or states.shape[1] != GPU_STATE_WIDTH or states.shape[0] < 1:
            raise ValueError("states must have shape [N, 170] with N > 0")
        if not states.is_contiguous():
            raise ValueError("states must be contiguous")
        self.states = states

    @property
    def device(self):
        return self.states.device

    def __len__(self):
        return self.states.shape[0]

    @classmethod
    def initial(cls, n: int, device="cuda", neutral=(4, 4)):
        if type(n) is not int or n < 1:
            raise ValueError("n must be a positive integer")
        target = torch.device(device)
        if target.type != "cuda":
            raise ValueError("GpuStateBatch requires a CUDA device")
        if neutral is None:
            point = -1
        elif (isinstance(neutral, (tuple, list)) and len(neutral) == 2
              and all(type(value) is int and 0 <= value < 9 for value in neutral)):
            point = neutral[0] * 9 + neutral[1]
        else:
            raise ValueError("neutral must be None or a pair of coordinates in [0, 8]")
        states = torch.empty((n, GPU_STATE_WIDTH), device=target, dtype=torch.int32)
        result = cls(states)
        with torch.cuda.device(result.device):
            _rules_module(result.device).launch(
                "gpu_initial_batch", grid=((n + 127) // 128, 1, 1),
                block=(128, 1, 1), args=[states, n, point],
            )
        return result

    @classmethod
    def from_engine(cls, states, device="cuda"):
        """Import CPU oracle snapshots, retaining permanent ownership/history."""
        import my_board_engine as engine
        from .encoding import _check_rules

        target = torch.device(device)
        if target.type != "cuda":
            raise ValueError("GpuStateBatch requires a CUDA device")
        snapshots = list(states)
        if not snapshots:
            raise ValueError("At least one engine State is required")
        cells = {
            engine.Cell.Empty: 0, engine.Cell.Black: 1,
            engine.Cell.White: 2, engine.Cell.Neutral: 3,
        }
        reasons = {
            engine.EndReason.None_: GPU_NONE,
            engine.EndReason.Capture: GPU_CAPTURE,
            engine.EndReason.Suicide: GPU_SUICIDE,
            engine.EndReason.TwoPasses: GPU_TWO_PASSES,
        }
        records = []
        for state in snapshots:
            if not isinstance(state, engine.State):
                raise TypeError("Expected engine State snapshots")
            state = state.copy()
            _check_rules(state)
            record = [cells[cell] for cell in state.board.cells]
            record.extend(cells[owner] for owner in state.ownership)
            record.extend((
                cells[state.to_play], state.remaining_stones(engine.Cell.Black),
                state.remaining_stones(engine.Cell.White), state.consecutive_passes,
                reasons[state.result.reason], cells[state.result.winner],
                state.result.captured_stones, 0,
            ))
            records.append(record)
        return cls(torch.tensor(records, dtype=torch.int32, device=target))

    def clone(self):
        return type(self)(self.states.clone())

    def play(self, actions):
        """Apply one action per state on-device; rejected states stay unchanged."""
        if isinstance(actions, torch.Tensor):
            action_tensor = actions
        else:
            action_tensor = torch.as_tensor(actions)
        if action_tensor.shape != (len(self),):
            raise ValueError("actions must have shape [N]")
        if action_tensor.dtype not in (torch.int32, torch.int64):
            raise TypeError("actions must be integer action indices")
        action_tensor = action_tensor.to(device=self.device)
        if action_tensor.dtype == torch.int64:
            # Preserve rejection for huge int64 values instead of wrapping them
            # into an accidentally legal int32 action during conversion.
            action_tensor = torch.where(
                (action_tensor >= 0) & (action_tensor < GPU_ACTION_SIZE),
                action_tensor, -1,
            ).to(dtype=torch.int32)
        action_tensor = action_tensor.contiguous()
        accepted = torch.empty(len(self), device=self.device, dtype=torch.bool)
        with torch.cuda.device(self.device):
            _rules_module(self.device).launch(
                "gpu_play_batch", grid=((len(self) + 127) // 128, 1, 1),
                block=(128, 1, 1), args=[self.states, action_tensor, accepted, len(self)],
            )
        return accepted

    def encode(self):
        """Return CUDA schema-1 features ``[N,10,9,9]`` and masks ``[N,82]``."""
        features = torch.empty((len(self), 10, 9, 9), device=self.device, dtype=torch.float32)
        masks = torch.empty((len(self), GPU_ACTION_SIZE), device=self.device, dtype=torch.bool)
        with torch.cuda.device(self.device):
            _rules_module(self.device).launch(
                "gpu_encode_batch", grid=((len(self) + 127) // 128, 1, 1),
                block=(128, 1, 1), args=[self.states, features, masks, len(self)],
            )
        return features, masks

    def snapshot(self):
        """Synchronize and copy debug records to the CPU; never used in search."""
        return [{
            "cells": record[:GPU_CELL_COUNT],
            "ownership": record[GPU_CELL_COUNT:GPU_ACTOR],
            "actor": record[GPU_ACTOR],
            "black_stock": record[GPU_BLACK_STOCK],
            "white_stock": record[GPU_WHITE_STOCK],
            "passes": record[GPU_PASSES],
            "reason": record[GPU_REASON],
            "winner": record[GPU_WINNER],
            "captured_count": record[GPU_CAPTURED],
            "reserved": record[GPU_RESERVED],
        } for record in self.states.cpu().tolist()]
