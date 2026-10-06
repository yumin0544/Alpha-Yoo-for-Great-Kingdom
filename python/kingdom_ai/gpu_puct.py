"""Batched CUDA PUCT: rules, tree selection, expansion and backup stay on GPU.

Each lane owns one independent tree. Python dispatches a fixed simulation budget
and the policy/value model, but never copies a leaf board, feature or tree to CPU.
Compact inference batches require a CUDA ``nonzero`` metadata synchronization.
Root noise uses a dedicated CUDA generator; its random stream differs from the
C++ implementation even when both receive the same seed.
"""

from __future__ import annotations

import copy
import math
import threading
from dataclasses import dataclass
from numbers import Real

import torch
from torch import nn

from .encoding import ACTION_SIZE, BOARD_SIZE, INPUT_CHANNELS
from .gpu_rules import GPU_STATE_WIDTH, RULES_CUDA_SOURCE, GpuStateBatch
from .gpu_runtime import CudaModule


@dataclass(frozen=True)
class GpuPUCTOptions:
    """Opt-in root safety and value-based first-play urgency preserve defaults.

    ``tactical_checks`` prioritizes immediate wins and avoids an opponent's
    immediate winning reply when a safe candidate exists. ``fpu_reduction``
    initializes unvisited edges from the node's player-perspective value minus
    that amount; ``None`` retains the original zero initialization.
    """
    simulations: int = 32
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    seed: int = 42
    tactical_checks: bool = False
    fpu_reduction: float | None = None

    def __post_init__(self):
        if type(self.simulations) is not int or not 1 <= self.simulations <= 2**31 - 2:
            raise ValueError("simulations must be an integer in [1, 2147483646]")
        for name in ("c_puct", "dirichlet_alpha", "dirichlet_epsilon"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and real")
        if self.c_puct <= 0 or self.dirichlet_alpha <= 0:
            raise ValueError("c_puct and dirichlet_alpha must be positive")
        if not 0 <= self.dirichlet_epsilon <= 1:
            raise ValueError("dirichlet_epsilon must be in [0, 1]")
        if type(self.seed) is not int or not 0 <= self.seed <= 2**64 - 1:
            raise ValueError("seed must be an unsigned 64-bit integer")
        if type(self.tactical_checks) is not bool:
            raise TypeError("tactical_checks must be a bool")
        if self.fpu_reduction is not None and (
                isinstance(self.fpu_reduction, bool)
                or not isinstance(self.fpu_reduction, Real)
                or not math.isfinite(self.fpu_reduction)
                or self.fpu_reduction < 0):
            raise ValueError("fpu_reduction must be None or a finite non-negative real")


@dataclass(frozen=True)
class GpuSearchResult:
    actions: torch.Tensor
    visits: torch.Tensor
    policy: torch.Tensor
    values: torch.Tensor
    simulations: torch.Tensor
    network_evaluations: torch.Tensor
    priors: torch.Tensor
    best_values: torch.Tensor
    nodes: torch.Tensor


_SEARCH_CUDA_SOURCE = r'''
__device__ long long gpu_vertex(int game, int node, int capacity) {
    return (long long)game * capacity + node;
}

__device__ bool gpu_root_has_winning_reply(const int* next, int root_actor) {
    // Only a last-liberty capture or a second pass can win in one move. Avoid
    // replaying every opponent placement, then verify the identified reply with
    // the same transition routine used by search and self-play.
    int reply[GPU_STATE_WIDTH];
    if (next[GPU_PASSES] == 1) {
        for (int i = 0; i < GPU_STATE_WIDTH; ++i) reply[i] = next[i];
        if (gpu_play(reply, GPU_CELL_COUNT) && reply[GPU_REASON] != GPU_NONE &&
            reply[GPU_WINNER] == next[GPU_ACTOR]) return true;
    }
    const int stock_index = next[GPU_ACTOR] == 1 ? GPU_BLACK_STOCK : GPU_WHITE_STOCK;
    if (next[stock_index] < 1) return false;
    bool examined[GPU_CELL_COUNT];
    for (int i = 0; i < GPU_CELL_COUNT; ++i) examined[i] = false;
    int group[GPU_CELL_COUNT];
    int neighbours[4];
    for (int point = 0; point < GPU_CELL_COUNT; ++point) {
        if (next[point] != root_actor || examined[point]) continue;
        const int length = gpu_collect_group(next, point, group);
        int liberty = -1;
        bool several = false;
        for (int i = 0; i < length; ++i) {
            examined[group[i]] = true;
            const int count = gpu_neighbours(group[i], neighbours);
            for (int j = 0; j < count; ++j) {
                const int adjacent = neighbours[j];
                if (next[adjacent] != 0) continue;
                if (liberty < 0) liberty = adjacent;
                else if (liberty != adjacent) several = true;
            }
        }
        if (several || liberty < 0 || !gpu_legal(next, liberty)) continue;
        for (int i = 0; i < GPU_STATE_WIDTH; ++i) reply[i] = next[i];
        if (gpu_play(reply, liberty) && reply[GPU_REASON] == GPU_CAPTURE &&
            reply[GPU_WINNER] == next[GPU_ACTOR]) return true;
    }
    return false;
}

// Check actual transitions and the opponent's winning replies, including
// capture-before-suicide and pass scoring.
// Each CUDA thread owns one candidate; the input state and encoded features stay
// untouched. Search masks may narrow candidates without changing rule legality.
extern "C" __global__ void gpu_search_root_tactics(
    const int* input, const bool* masks, int* outcomes, int batch) {
    long long item = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= (long long)batch * GPU_ACTION_SIZE) return;
    const int game = item / GPU_ACTION_SIZE;
    const int action = item % GPU_ACTION_SIZE;
    outcomes[item] = 0;
    if (!masks[item]) return;
    const int* root = input + (long long)game * GPU_STATE_WIDTH;
    int next[GPU_STATE_WIDTH];
    for (int i = 0; i < GPU_STATE_WIDTH; ++i) next[i] = root[i];
    if (!gpu_play(next, action)) return;
    if (next[GPU_REASON] != GPU_NONE)
        outcomes[item] = next[GPU_WINNER] == root[GPU_ACTOR] ? 1 : -1;
    else if (gpu_root_has_winning_reply(next, root[GPU_ACTOR]))
        outcomes[item] = -1;
}

extern "C" __global__ void gpu_search_root_filter(
    const int* outcomes, bool* masks, int batch) {
    const int game = blockIdx.x * blockDim.x + threadIdx.x;
    if (game >= batch) return;
    const long long base = (long long)game * GPU_ACTION_SIZE;
    bool winning = false;
    bool safe = false;
    for (int action = 0; action < GPU_ACTION_SIZE; ++action) {
        if (!masks[base + action]) continue;
        winning |= outcomes[base + action] > 0;
        safe |= outcomes[base + action] >= 0;
    }
    for (int action = 0; action < GPU_ACTION_SIZE; ++action) {
        if (winning) masks[base + action] &= outcomes[base + action] > 0;
        else if (safe) masks[base + action] &= outcomes[base + action] >= 0;
    }
}

extern "C" __global__ void gpu_search_initialize(
    const int* input, int* states, int* node_count, int* leaf,
    int* path_length, bool* needs_evaluation, float* features, bool* masks,
    float* leaf_values, int batch, int capacity) {
    int game = blockIdx.x * blockDim.x + threadIdx.x;
    if (game >= batch) return;
    int* root = states + gpu_vertex(game, 0, capacity) * GPU_STATE_WIDTH;
    for (int i = 0; i < GPU_STATE_WIDTH; ++i)
        root[i] = input[(long long)game * GPU_STATE_WIDTH + i];
    node_count[game] = root[GPU_REASON] == 0 ? 1 : 0;
    leaf[game] = 0;
    path_length[game] = 0;
    needs_evaluation[game] = root[GPU_REASON] == 0;
    leaf_values[game] = root[GPU_REASON] == 0 ? 0.0f : gpu_terminal_value(root);
    if (needs_evaluation[game])
        gpu_encode(root, features + (long long)game * 810,
                   masks + (long long)game * GPU_ACTION_SIZE);
}

extern "C" __global__ void gpu_search_select(
    int* states, const double* prior, const int* visits, const double* value_sum,
    const float* node_values,
    int* children, const int* node_visits, const bool* expanded,
    const bool* legal, int* node_count, int* leaf, int* path_nodes,
    int* path_actions, int* path_length, bool* needs_evaluation,
    float* features, bool* masks, float* leaf_values,
    const double* options, int* errors, int batch, int capacity) {
    int game = blockIdx.x * blockDim.x + threadIdx.x;
    if (game >= batch) return;
    int node = 0;
    int depth = 0;
    needs_evaluation[game] = false;
    path_length[game] = 0;
    const double c_puct = options[0];
    while (true) {
        long long vertex = gpu_vertex(game, node, capacity);
        const int* current = states + vertex * GPU_STATE_WIDTH;
        if (current[GPU_REASON] != 0) {
            leaf_values[game] = gpu_terminal_value(current);
            break;
        }
        if (!expanded[vertex]) {
            needs_evaluation[game] = true;
            gpu_encode(current, features + (long long)game * 810,
                       masks + (long long)game * GPU_ACTION_SIZE);
            break;
        }
        long long base = vertex * GPU_ACTION_SIZE;
        const double scale = sqrt((double)node_visits[vertex]);
        int best = -1;
        double best_score = -1.7976931348623157e308;
        for (int action = 0; action < GPU_ACTION_SIZE; ++action) {
            if (!legal[base + action]) continue;
            const int n = visits[base + action];
            const double q = n == 0
                ? (options[2] != 0.0 ? node_values[vertex] - options[3] : 0.0)
                : value_sum[base + action] / n;
            const double exploration = prior[base + action] * scale / (1.0 + n);
            const double score = c_puct >= 1.0 ? q / c_puct + exploration
                                              : q + c_puct * exploration;
            if (best < 0 || score > best_score ||
                (score == best_score && prior[base + action] > prior[base + best])) {
                best = action;
                best_score = score;
            }
        }
        if (best < 0 || depth >= capacity - 1) {
            errors[game] |= 1;
            leaf_values[game] = 0.0f;
            break;
        }
        int child = children[base + best];
        if (child < 0) {
            child = node_count[game]++;
            if (child >= capacity) {
                errors[game] |= 1;
                leaf_values[game] = 0.0f;
                break;
            }
            int* next = states + gpu_vertex(game, child, capacity) * GPU_STATE_WIDTH;
            for (int i = 0; i < GPU_STATE_WIDTH; ++i) next[i] = current[i];
            if (!gpu_play(next, best)) {
                errors[game] |= 1;
                leaf_values[game] = 0.0f;
                break;
            }
            children[base + best] = child;
        }
        path_nodes[(long long)game * capacity + depth] = node;
        path_actions[(long long)game * capacity + depth] = best;
        ++depth;
        node = child;
    }
    leaf[game] = node;
    path_length[game] = depth;
}

extern "C" __global__ void gpu_search_expand_backup(
    const int* states, double* prior, int* visits, double* value_sum,
    float* node_values,
    const int* children, int* node_visits, bool* expanded, bool* legal,
    const int* leaf, const int* path_nodes, const int* path_actions,
    const int* path_length, const bool* needs_evaluation,
    const bool* masks, const float* probabilities, const float* leaf_values,
    const float* noise, const double* options, int* simulation_count,
    int* evaluation_count, int* errors, int batch, int capacity, int backup) {
    int game = blockIdx.x * blockDim.x + threadIdx.x;
    if (game >= batch) return;
    long long root_vertex = gpu_vertex(game, 0, capacity);
    if (states[root_vertex * GPU_STATE_WIDTH + GPU_REASON] != 0) return;
    int node = leaf[game];
    long long vertex = gpu_vertex(game, node, capacity);
    long long base = vertex * GPU_ACTION_SIZE;
    long long row = (long long)game * GPU_ACTION_SIZE;
    if (needs_evaluation[game]) {
        node_values[vertex] = leaf_values[game];
        double largest = 0.0;
        double total = 0.0;
        double noise_total = 0.0;
        int legal_count = 0;
        for (int action = 0; action < GPU_ACTION_SIZE; ++action) {
            bool valid = masks[row + action];
            legal[base + action] = valid;
            if (valid) {
                if (probabilities[row + action] > largest)
                    largest = probabilities[row + action];
                noise_total += noise[row + action];
                ++legal_count;
            }
        }
        if (largest > 0.0) {
            for (int action = 0; action < GPU_ACTION_SIZE; ++action)
                if (legal[base + action]) total += probabilities[row + action] / largest;
        }
        if (legal_count == 0 || !(total > 0.0) || !isfinite(total)) {
            errors[game] |= 2;
            total = 1.0;
            largest = 1.0;
        }
        for (int action = 0; action < GPU_ACTION_SIZE; ++action) {
            if (!legal[base + action]) {
                prior[base + action] = 0.0;
                continue;
            }
            double weight = (probabilities[row + action] / largest) / total;
            if (node == 0 && options[1] != 0.0) {
                double sampled = noise_total > 0.0 ? noise[row + action] / noise_total
                                                    : 1.0 / legal_count;
                weight = (1.0 - options[1]) * weight + options[1] * sampled;
            }
            prior[base + action] = weight;
        }
        if (node == 0 && options[1] != 0.0) {
            double combined_total = 0.0;
            for (int action = 0; action < GPU_ACTION_SIZE; ++action)
                combined_total += prior[base + action];
            for (int action = 0; action < GPU_ACTION_SIZE; ++action)
                prior[base + action] /= combined_total;
        }
        expanded[vertex] = true;
        ++evaluation_count[game];
    }
    if (!backup) return;
    const int leaf_actor = states[vertex * GPU_STATE_WIDTH + GPU_ACTOR];
    const double leaf_value = leaf_values[game];
    ++node_visits[root_vertex];
    for (int depth = 0; depth < path_length[game]; ++depth) {
        int parent = path_nodes[(long long)game * capacity + depth];
        int action = path_actions[(long long)game * capacity + depth];
        long long parent_vertex = gpu_vertex(game, parent, capacity);
        long long edge = parent_vertex * GPU_ACTION_SIZE + action;
        const double value = states[parent_vertex * GPU_STATE_WIDTH + GPU_ACTOR] == leaf_actor
            ? leaf_value : -leaf_value;
        ++visits[edge];
        value_sum[edge] += value;
        ++node_visits[gpu_vertex(game, children[edge], capacity)];
    }
    ++simulation_count[game];
}

extern "C" __global__ void gpu_search_finish(
    const int* states, const double* prior, const int* visits,
    const double* value_sum, const bool* legal, const int* simulation_count,
    long long* actions, float* policy, float* values, float* best_values,
    int batch, int capacity) {
    int game = blockIdx.x * blockDim.x + threadIdx.x;
    if (game >= batch) return;
    long long root_vertex = gpu_vertex(game, 0, capacity);
    const int* root = states + root_vertex * GPU_STATE_WIDTH;
    long long tree_base = root_vertex * GPU_ACTION_SIZE;
    long long row = (long long)game * GPU_ACTION_SIZE;
    if (root[GPU_REASON] != 0) {
        actions[game] = -1;
        values[game] = gpu_terminal_value(root);
        best_values[game] = 0.0f;
        return;
    }
    int best = -1;
    double best_q = 0.0;
    double root_sum = 0.0;
    for (int action = 0; action < GPU_ACTION_SIZE; ++action) {
        const int n = visits[tree_base + action];
        policy[row + action] = (float)n / simulation_count[game];
        root_sum += value_sum[tree_base + action];
        if (!legal[tree_base + action]) continue;
        const double q = n == 0 ? 0.0 : value_sum[tree_base + action] / n;
        if (best < 0 || n > visits[tree_base + best] ||
            (n == visits[tree_base + best] && (q > best_q ||
             (q == best_q && prior[tree_base + action] > prior[tree_base + best])))) {
            best = action;
            best_q = q;
        }
    }
    actions[game] = best;
    values[game] = (float)(root_sum / simulation_count[game]);
    best_values[game] = (float)best_q;
}
'''


class GpuPUCT:
    """Independent GPU trees and one compact CUDA model batch per simulation.

    Only confirmed default rules are supported by ``GpuStateBatch``. This class
    uses a fixed simulation count and does not support the C++ time limit option.
    Source model modes, weights, device and gradients are preserved by copying.
    """

    def __init__(self, model: nn.Module, options=None, device="cuda"):
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if options is None:
            options = GpuPUCTOptions()
        if not isinstance(options, GpuPUCTOptions):
            raise TypeError("options must be GpuPUCTOptions")
        self.options = options
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("GpuPUCT requires a CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.model = copy.deepcopy(model).to(device=self.device, dtype=torch.float32).eval()
        self._module = CudaModule(RULES_CUDA_SOURCE + _SEARCH_CUDA_SOURCE, device=self.device)
        self._generator = torch.Generator(device=self.device).manual_seed(options.seed)
        self._lock = threading.Lock()

    def reset_seed(self, seed=None):
        """Restart the dedicated root-noise stream without changing global RNGs."""
        if seed is None:
            seed = self.options.seed
        if type(seed) is not int or not 0 <= seed <= 2**64 - 1:
            raise ValueError("seed must be an unsigned 64-bit integer")
        with self._lock, torch.cuda.device(self.device):
            self._generator.manual_seed(seed)

    def _launch(self, name, batch, args):
        self._module.launch(name, grid=((batch + 127) // 128, 1, 1),
                            block=(128, 1, 1), args=args)

    def search(self, state: GpuStateBatch) -> GpuSearchResult:
        if not isinstance(state, GpuStateBatch):
            raise TypeError("Expected a GpuStateBatch")
        # The public tensor can be replaced by callers; validate its wire shape
        # and storage before passing a raw pointer to a CUDA kernel.
        GpuStateBatch(state.states)
        if state.device != self.device:
            raise ValueError("State and searcher must use the same CUDA device")
        with self._lock, torch.cuda.device(self.device), torch.inference_mode():
            return self._search(state)

    def _search(self, state):
        batch = state.states.shape[0]
        if batch < 1:
            raise ValueError("Search needs at least one state")
        capacity = self.options.simulations + 1
        device = self.device

        def zeros(shape, dtype):
            return torch.zeros(shape, dtype=dtype, device=device)

        states = zeros((batch, capacity, GPU_STATE_WIDTH), torch.int32)
        prior = zeros((batch, capacity, ACTION_SIZE), torch.float64)
        visits = zeros(prior.shape, torch.int32)
        value_sum = zeros(prior.shape, torch.float64)
        children = torch.full(prior.shape, -1, dtype=torch.int32, device=device)
        node_visits = zeros((batch, capacity), torch.int32)
        node_values = zeros((batch, capacity), torch.float32)
        expanded = zeros((batch, capacity), torch.bool)
        legal = zeros(prior.shape, torch.bool)
        node_count = zeros((batch,), torch.int32)
        leaf = zeros((batch,), torch.int32)
        path_nodes = zeros((batch, capacity), torch.int32)
        path_actions = zeros((batch, capacity), torch.int32)
        path_length = zeros((batch,), torch.int32)
        needs_evaluation = zeros((batch,), torch.bool)
        features = zeros((batch, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), torch.float32)
        masks = zeros((batch, ACTION_SIZE), torch.bool)
        probabilities = zeros((batch, ACTION_SIZE), torch.float32)
        leaf_values = zeros((batch,), torch.float32)
        noise = zeros((batch, ACTION_SIZE), torch.float32)
        options = torch.tensor([self.options.c_puct, self.options.dirichlet_epsilon,
                                self.options.fpu_reduction is not None,
                                self.options.fpu_reduction or 0.0],
                               dtype=torch.float64, device=device)
        simulation_count = zeros((batch,), torch.int32)
        evaluation_count = zeros((batch,), torch.int32)
        errors = zeros((batch,), torch.int32)
        self._launch("gpu_search_initialize", batch, [
            state.states, states, node_count, leaf, path_length, needs_evaluation,
            features, masks, leaf_values, batch, capacity])
        if self.options.tactical_checks:
            outcomes = zeros((batch, ACTION_SIZE), torch.int32)
            self._launch("gpu_search_root_tactics", batch * ACTION_SIZE, [
                state.states, masks, outcomes, batch])
            self._launch("gpu_search_root_filter", batch, [outcomes, masks, batch])

        def evaluate():
            active = needs_evaluation.nonzero(as_tuple=False).flatten()
            if active.numel() == 0:
                return
            logits, values = self.model(features.index_select(0, active))
            if logits.shape != (active.numel(), ACTION_SIZE) or values.shape != (active.numel(),):
                raise ValueError("model must return logits [N, 82] and values [N]")
            if logits.device != device or values.device != device:
                raise ValueError("model outputs must stay on the search CUDA device")
            if not logits.is_floating_point() or not values.is_floating_point():
                raise TypeError("model logits and values must be floating-point tensors")
            bad = ~torch.isfinite(logits).all(dim=1) | ~torch.isfinite(values) | (values.abs() > 1)
            logits = logits.to(dtype=torch.float32)
            values = values.to(dtype=torch.float32)
            bad |= ~torch.isfinite(logits).all(dim=1) | ~torch.isfinite(values)
            errors.index_copy_(0, active,
                errors.index_select(0, active).bitwise_or(bad.to(torch.int32) * 2))
            safe_logits = torch.where(bad[:, None], torch.zeros_like(logits), logits)
            active_masks = masks.index_select(0, active)
            weights = torch.softmax(safe_logits.masked_fill(~active_masks, -torch.inf), dim=1)
            probabilities.index_copy_(0, active, weights.masked_fill(~active_masks, 0.0))
            leaf_values.index_copy_(0, active, torch.where(bad, torch.zeros_like(values), values))

        expand_args = [states, prior, visits, value_sum, node_values, children, node_visits,
                       expanded, legal, leaf, path_nodes, path_actions, path_length,
                       needs_evaluation, masks, probabilities, leaf_values, noise,
                       options, simulation_count, evaluation_count, errors, batch, capacity]
        evaluate()
        if self.options.dirichlet_epsilon:
            concentration = torch.full((batch, ACTION_SIZE), self.options.dirichlet_alpha,
                                       dtype=torch.float64, device=device)
            samples = torch._standard_gamma(concentration, generator=self._generator)
            samples = torch.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
            # Rescale before conversion to FP32 to avoid overflow and preserve tiny alpha samples.
            samples.masked_fill_(~masks, 0.0)
            largest = samples.amax(dim=1, keepdim=True)
            noise.copy_((samples / largest.clamp_min(torch.finfo(torch.float64).tiny)).float())
        self._launch("gpu_search_expand_backup", batch, expand_args + [0])
        for _ in range(self.options.simulations):
            self._launch("gpu_search_select", batch, [
                states, prior, visits, value_sum, node_values, children, node_visits, expanded,
                legal, node_count, leaf, path_nodes, path_actions, path_length,
                needs_evaluation, features, masks, leaf_values, options, errors, batch, capacity])
            evaluate()
            self._launch("gpu_search_expand_backup", batch, expand_args + [1])
        actions = torch.full((batch,), -1, dtype=torch.int64, device=device)
        policy = zeros((batch, ACTION_SIZE), torch.float32)
        values = zeros((batch,), torch.float32)
        best_values = zeros((batch,), torch.float32)
        self._launch("gpu_search_finish", batch, [
            states, prior, visits, value_sum, legal, simulation_count,
            actions, policy, values, best_values, batch, capacity])
        # Copy one aggregate validation scalar only after all simulations.
        code = int(errors.max().item())
        if code:
            if code & 2:
                raise ValueError("PUCT model must return finite logits and finite values in [-1, 1]")
            raise RuntimeError("CUDA PUCT detected an illegal edge or exhausted tree capacity")
        return GpuSearchResult(actions=actions, visits=visits[:, 0].clone(), policy=policy,
                               values=values, simulations=simulation_count,
                               network_evaluations=evaluation_count, priors=prior[:, 0].clone(),
                               best_values=best_values, nodes=node_count)
