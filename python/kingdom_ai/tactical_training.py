"""Proof-labelled tactical fine-tuning, kept separate from game-strength claims.

Ordinary Go ideas name the curriculum; only the verified Kingdom state
transitions and a full-width solver provide targets. UNKNOWN is not a loss or
a zero-valued draw. Certified LOSS positions can supply a value-only label;
they have no justified policy imitation target. One-shot fine-tuning retains
its WIN-only contract; the online teacher explicitly opts into LOSS rows.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math
from time import perf_counter

import torch
import my_board_engine as engine

from .augmentation import augment_batch
from .encoding import encode_state, move_to_action
from .model import PolicyValueNet
from .training import TrainingBatch, TrainingSample, make_batch, train_step


@dataclass(frozen=True)
class CertifiedTacticalSample:
    case_id: str
    family_id: str
    motif: str
    sample: TrainingSample
    proof: dict
    policy_enabled: bool = True


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def model_digest(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _outcome_name(outcome):
    # pybind enum.name and string fakes in isolated unit tests share this path.
    return str(getattr(outcome, "name", outcome)).split(".")[-1].upper()


def collect_certified_samples(cases: Sequence[dict], options, *,
                              load_position: Callable | None = None,
                              solver: Callable | None = None,
                              max_cases=64, generation_seconds=60.0,
                              bound_remaining_time=False, include_loss=False):
    """Solve bounded original cases, returning samples and an audit of all cases.

    ``winning_moves`` must be root moves proved WIN by the full-width solver.
    Uniform targets over this *known safe subset* do not claim that other
    unproved legal moves are objectively losing. A per-case solver deadline is
    still required: the overall deadline is checked between solver calls.
    With ``bound_remaining_time=True`` each solver receives no more than the
    remaining overall milliseconds (engine options only). Position loading and
    OS scheduling overhead are not a hard real-time bound.
    """
    _integer(max_cases, "max_cases")
    if type(include_loss) is not bool:
        raise ValueError("include_loss must be a bool")
    if type(bound_remaining_time) is not bool:
        raise ValueError("bound_remaining_time must be a bool")
    if bound_remaining_time and not isinstance(options, engine.TacticalSolverOptions):
        raise TypeError("Remaining-budget enforcement requires engine solver options")
    if (isinstance(generation_seconds, bool) or not isinstance(generation_seconds, (int, float))
            or not math.isfinite(generation_seconds) or generation_seconds <= 0):
        raise ValueError("generation_seconds must be finite and positive")
    if load_position is None:
        from .tactical_positions import load_position
    if solver is None:
        if not hasattr(engine, "solve_tactics"):
            raise RuntimeError("Build/install the tactical solver binding before generating labels")
        solver = engine.solve_tactics
    started = perf_counter()
    samples, records, seen = [], [], set()
    for case in list(cases)[:max_cases]:
        if perf_counter() - started >= generation_seconds:
            break
        if not isinstance(case, dict):
            raise TypeError("Each tactical case must be a dictionary")
        case_id = case.get("id")
        family_id = case.get("family_id", case_id)
        if (type(case_id) is not str or not case_id or case_id in seen
                or type(family_id) is not str or not family_id):
            raise ValueError("Cases need distinct nonempty id and nonempty family_id")
        seen.add(case_id)
        state = load_position(case)
        if not isinstance(state, engine.State):
            raise TypeError("Position loader must return an engine State")
        record = {"id": case_id, "family_id": family_id, "motif": case.get("motif", "unspecified")}
        # Keep complete replayable input and provenance, not just model feature
        # planes, which cannot reconstruct permanent territory ownership.
        for field in ("position", "rules", "source"):
            if field in case:
                record[field] = deepcopy(case[field])
        record["actor"] = state.to_play.name
        record["rules"] = {
            "suicide_rule": state.rules.suicide_rule.name,
            "allow_own_territory_moves": state.rules.allow_own_territory_moves,
            "allow_single_edge_territory": state.rules.allow_single_edge_territory,
            "stones_per_player": state.rules.stones_per_player,
        }
        if state.result.finished():
            records.append({**record, "outcome": "TERMINAL", "training_label": False})
            continue
        solve_options = options
        if bound_remaining_time:
            remaining_ms = int((generation_seconds - (perf_counter() - started)) * 1000)
            if remaining_ms < 1:
                break
            per_case_ms = (min(options.time_limit_ms, remaining_ms)
                           if options.time_limit_ms else remaining_ms)
            per_case_ms = min(per_case_ms, 2 ** 31 - 1)
            solve_options = engine.TacticalSolverOptions(
                max_depth=options.max_depth, max_nodes=options.max_nodes,
                time_limit_ms=per_case_ms)
        result = solver(state.copy(), solve_options)
        outcome = _outcome_name(result.outcome)
        if outcome not in ("WIN", "LOSS", "UNKNOWN"):
            raise ValueError("Solver returned an unsupported proof outcome")
        winning_moves = list(getattr(result, "winning_moves", ()))
        record.update({"outcome": outcome, "training_label": False,
                       "winning_actions": [move_to_action(move) for move in winning_moves]})
        for field in ("nodes", "completed_depth", "proof_depth", "elapsed_ms", "budget_exhausted"):
            if hasattr(result, field):
                record[field] = getattr(result, field)
        variation = list(getattr(result, "principal_variation", ()))
        if variation:
            replayed = state.copy()
            for move in variation:
                if not replayed.play(move).accepted():
                    raise ValueError("Solver principal variation contains an illegal move")
            record["principal_variation"] = [
                "pass" if move.is_pass() else [move.point.row + 1, move.point.col + 1]
                for move in variation]
            record["pv_plies"] = len(variation)
            record["pv_end_reason"] = replayed.result.reason.name
            record["pv_winner"] = replayed.result.winner.name
            if outcome in ("WIN", "LOSS") and (
                    not replayed.result.finished()
                    or (replayed.result.winner == state.to_play) != (outcome == "WIN")):
                raise ValueError("Solver principal variation contradicts its proof outcome")
        if outcome == "UNKNOWN" or (outcome == "LOSS" and not include_loss):
            # No invented UNKNOWN target. LOSS is opt-in for online value-only
            # training; legacy one-shot fine-tuning stays WIN-only.
            records.append(record)
            continue
        encoded = encode_state(state)
        actions = record["winning_actions"]
        if outcome == "WIN" and (not actions or len(set(actions)) != len(actions)
                or any(not bool(encoded.legal_mask[action]) for action in actions)):
            raise ValueError("A nonterminal WIN proof must name distinct legal winning root moves")
        if outcome == "LOSS" and actions:
            raise ValueError("A LOSS proof cannot name winning root moves")
        policy = torch.zeros_like(encoded.legal_mask, dtype=torch.float32)
        if outcome == "WIN":
            policy[actions] = 1.0 / len(actions)
        else:
            # Storage placeholder only. Never label a PV move as best defense.
            policy[encoded.legal_mask] = 1.0 / int(encoded.legal_mask.sum())
        record["training_label"] = True
        record["policy_enabled"] = outcome == "WIN"
        record["target_kind"] = "policy_and_value" if outcome == "WIN" else "value_only"
        sample = TrainingSample(encoded.features.clone(), encoded.legal_mask.clone(),
                                policy, 1.0 if outcome == "WIN" else -1.0, encoded.to_play)
        samples.append(CertifiedTacticalSample(case_id, family_id, str(record["motif"]),
                                              sample, deepcopy(record), outcome == "WIN"))
        records.append(record)
    return samples, {"cases_available": len(cases), "cases_solved": len(records),
                     "certified_win_samples": sum(row.policy_enabled for row in samples),
                     "certified_loss_samples": sum(not row.policy_enabled for row in samples),
                     "certified_samples": len(samples),
                     "generation_seconds": perf_counter() - started,
                     "excluded_unknown": sum(row["outcome"] == "UNKNOWN" for row in records),
                     "excluded_loss": sum(row["outcome"] == "LOSS" and not row["training_label"]
                                          for row in records),
                     "records": records}


def _canonical_position(sample: TrainingSample):
    # Include all 10 planes and the pass bit, not just stone geometry.
    forms = []
    for reflected in (False, True):
        features = sample.features.flip(-1) if reflected else sample.features
        mask = sample.legal_mask[:81].reshape(9, 9)
        if reflected:
            mask = mask.flip(-1)
        for rotation in range(4):
            data = torch.rot90(features, rotation, (-2, -1)).contiguous().numpy().tobytes()
            data += torch.rot90(mask, rotation, (-2, -1)).contiguous().numpy().tobytes()
            data += sample.legal_mask[81:].numpy().tobytes()
            forms.append(data)
    return hashlib.sha256(min(forms)).hexdigest()


def _validated_samples(samples, *, allow_loss=False):
    if type(allow_loss) is not bool:
        raise ValueError("allow_loss must be a bool")
    rows = list(samples)
    for row in rows:
        if (not isinstance(row, CertifiedTacticalSample) or not isinstance(row.proof, dict)
                or row.proof.get("outcome") not in (("WIN", "LOSS") if allow_loss else ("WIN",))):
            raise ValueError("Training requires certified WIN samples")
        outcome = row.proof["outcome"]
        if (type(row.policy_enabled) is not bool
                or row.policy_enabled != (outcome == "WIN")
                or row.sample.value != (1.0 if outcome == "WIN" else -1.0)):
            raise ValueError("Certified WIN/+1 enables policy; LOSS/-1 is value-only")
        if "policy_enabled" in row.proof and (type(row.proof["policy_enabled"]) is not bool
                                              or row.proof["policy_enabled"] != row.policy_enabled):
            raise ValueError("Certificate policy mask disagrees with its target")
        if any(tensor.device.type != "cpu" for tensor in
               (row.sample.features, row.sample.legal_mask, row.sample.policy)):
            raise ValueError("Original tactical samples must be stored on CPU")
        batch = make_batch([row.sample])
        # Reuse strict replay validation for schema, legal policy and finite targets.
        from .replay import ReplayBuffer
        checker = ReplayBuffer(1)
        checker.extend([row.sample])
        proof_actions = row.proof.get("winning_actions")
        if outcome == "LOSS":
            if proof_actions != []:
                raise ValueError("Value-only LOSS certificates cannot name winning actions")
            continue
        if (not isinstance(proof_actions, list) or not proof_actions
                or any(type(action) is not int or not 0 <= action < 82 for action in proof_actions)
                or len(proof_actions) != len(set(proof_actions))
                or set(torch.nonzero(batch.policy[0] > 0).flatten().tolist()) != set(proof_actions)):
            raise ValueError("Policy support must equal the certified winning actions")
    return rows


def split_tactical_samples(samples, *, heldout_fraction=0.25, seed=42, allow_loss=False):
    """Split original families before augmentation, also joining D4 duplicates."""
    rows = _validated_samples(samples, allow_loss=allow_loss)
    if (isinstance(heldout_fraction, bool) or not isinstance(heldout_fraction, (int, float))
            or not math.isfinite(heldout_fraction) or not 0 < heldout_fraction < 1):
        raise ValueError("heldout_fraction must lie strictly between zero and one")
    _integer(seed, "seed", 0)
    if seed >= 2 ** 64:
        raise ValueError("seed must fit uint64")
    parent = list(range(len(rows)))
    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    identities, positions = {}, {}
    for index, row in enumerate(rows):
        for mapping, key in ((identities, row.family_id), (positions, _canonical_position(row.sample))):
            if key in mapping:
                parent[find(index)] = find(mapping[key])
            else:
                mapping[key] = index
    groups = {}
    for index in range(len(rows)):
        groups.setdefault(find(index), []).append(index)
    buckets = list(groups.values())
    if len(buckets) < 2:
        return rows, []
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(len(buckets), generator=generator).tolist()
    count = min(len(buckets) - 1, max(1, round(len(buckets) * heldout_fraction)))
    heldout_indices = {index for group in order[:count] for index in buckets[group]}
    return ([row for index, row in enumerate(rows) if index not in heldout_indices],
            [row for index, row in enumerate(rows) if index in heldout_indices])


def tactical_metrics(model, samples, *, allow_loss=False):
    """Raw network metrics; value-only LOSS rows are excluded from policy hits."""
    rows = _validated_samples(samples, allow_loss=allow_loss)
    if not rows:
        return {"cases": 0, "certified_top1": None, "certified_top3": None,
                "value_mse": None, "mean_value": None, "win_cases": 0, "loss_cases": 0,
                "win_value_mse": None, "loss_value_mse": None}
    device = next(model.parameters()).device
    flags = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.inference_mode():
            batch = make_batch([row.sample for row in rows], device=device)
            logits, values = model(batch.features)
            if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(values).all()):
                raise RuntimeError("Non-finite tactical evaluation outputs")
            logits = logits.masked_fill(~batch.legal_mask, -torch.inf)
            actions = logits.topk(3, dim=1).indices
            hits = batch.policy.gather(1, actions) > 0
            enabled = torch.tensor([row.policy_enabled for row in rows],
                                   dtype=torch.bool, device=device)
            value_error = (values - batch.value).square()
            wins = int(enabled.sum())
            def optional_mean(values):
                return values.mean().item() if values.numel() else None
            return {"cases": len(rows),
                    "certified_top1": optional_mean(hits[enabled, 0].float()),
                    "certified_top3": optional_mean(hits[enabled].any(dim=1).float()),
                    "value_mse": value_error.mean().item(), "mean_value": values.mean().item(),
                    "win_cases": wins, "loss_cases": len(rows) - wins,
                    "win_value_mse": optional_mean(value_error[enabled]),
                    "loss_value_mse": optional_mean(value_error[~enabled])}
    finally:
        for module, flag in flags:
            module.training = flag


def fine_tune_tactics(model, train_samples, heldout_samples=(), *, steps=256,
                     batch_size=128, learning_rate=3e-5, tactical_fraction=0.25,
                     replay=None, generator=None, seed=42, copy_model=True,
                     update_callback=None):
    """Train a candidate; never promote a champion or mutate replay contents.

    For a Trainer use ``copy_model=False`` and its public
    ``train_tactical_batch`` callback, so Adam/counters/checkpoints stay honest.
    Existing replay is sampled but not extended. With no replay the actual
    tactical fraction is 1.0, explicitly recorded in the result.
    """
    if not isinstance(model, PolicyValueNet):
        raise TypeError("model must be a PolicyValueNet")
    _integer(steps, "steps")
    _integer(batch_size, "batch_size")
    _integer(seed, "seed", 0)
    if seed >= 2 ** 64:
        raise ValueError("seed must fit uint64")
    if (isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float))
            or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("learning_rate must be finite and positive")
    if (isinstance(tactical_fraction, bool) or not isinstance(tactical_fraction, (int, float))
            or not math.isfinite(tactical_fraction) or not 0 < tactical_fraction <= 1):
        raise ValueError("tactical_fraction must lie in (0, 1]")
    if type(copy_model) is not bool or (update_callback is not None and copy_model):
        raise ValueError("An update callback requires copy_model=False")
    train_rows, heldout_rows = _validated_samples(train_samples), _validated_samples(heldout_samples)
    if not train_rows:
        raise ValueError("At least one certified WIN training sample is required")
    if ({row.case_id for row in train_rows} & {row.case_id for row in heldout_rows}
            or {row.family_id for row in train_rows} & {row.family_id for row in heldout_rows}
            or {_canonical_position(row.sample) for row in train_rows}
            & {_canonical_position(row.sample) for row in heldout_rows}):
        raise ValueError("Train and heldout identities/families/D4 positions must be disjoint")
    caller_generator = generator is not None
    if generator is None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
    if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
        raise ValueError("Training requires a caller-owned CPU generator")
    candidate = deepcopy(model) if copy_model else model
    device = next(candidate.parameters()).device
    has_replay = replay is not None and len(replay) > 0
    tactical_rows = max(1, round(batch_size * tactical_fraction)) if has_replay else batch_size
    replay_rows = batch_size - tactical_rows
    optimizer = None if update_callback is not None else torch.optim.Adam(
        candidate.parameters(), lr=learning_rate)
    generator_before = hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()
    before_digest = model_digest(candidate)
    before = {"train": tactical_metrics(candidate, train_rows),
              "heldout": tactical_metrics(candidate, heldout_rows)}
    losses = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = perf_counter()
    for _ in range(steps):
        indices = torch.randint(len(train_rows), (tactical_rows,), generator=generator).tolist()
        batch = make_batch([train_rows[index].sample for index in indices], device=device)
        if replay_rows:
            original = replay.sample(replay_rows, generator=generator, device=device)
            batch = TrainingBatch(*(torch.cat((getattr(batch, field), getattr(original, field)), dim=0)
                                    for field in ("features", "legal_mask", "policy", "value")))
        batch = augment_batch(batch, generator=generator)
        loss = (update_callback(batch) if update_callback is not None
                else train_step(candidate, optimizer, batch))
        if not isinstance(loss, dict) or any(not math.isfinite(loss.get(key, math.nan))
                                            for key in ("loss", "policy_loss", "value_loss")):
            raise RuntimeError("Training returned invalid losses")
        losses.append(loss)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = perf_counter() - started
    after = {"train": tactical_metrics(candidate, train_rows),
             "heldout": tactical_metrics(candidate, heldout_rows)}
    report = {"steps": steps, "batch_size": batch_size, "seed": seed,
              "learning_rate": learning_rate, "device": str(device),
              "update_backend": "caller_callback" if update_callback is not None else "independent_adam",
              "sampling_rng": "caller_generator" if caller_generator else "local_seed",
              "sampling_generator_sha256_before": generator_before,
              "sampling_generator_sha256_after": hashlib.sha256(
                  generator.get_state().numpy().tobytes()).hexdigest(),
              "augment_symmetries": True,
              "tactical_rows_per_batch": tactical_rows, "replay_rows_per_batch": replay_rows,
              "actual_tactical_fraction": tactical_rows / batch_size,
              "replay_size": len(replay) if has_replay else 0,
              "replay_contents_unchanged": True,
              "train_case_ids": [row.case_id for row in train_rows],
              "heldout_case_ids": [row.case_id for row in heldout_rows],
              "train_families": sorted({row.family_id for row in train_rows}),
              "heldout_families": sorted({row.family_id for row in heldout_rows}),
              "model_sha256_before": before_digest, "model_sha256_after": model_digest(candidate),
              "before": before, "after": after, "training_seconds": elapsed,
              "first_loss": losses[0], "last_loss": losses[-1],
              "mean_losses": {key: sum(row[key] for row in losses) / steps
                              for key in ("loss", "policy_loss", "value_loss")},
              "losses": losses, "promoted": False,
              "interpretation": "Proof-target fit, not independently established game strength. "
                                "Known winning moves may be a strict subset of all winning moves."}
    return candidate, report
