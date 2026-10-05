"""Batch synchronous C++ PUCT callbacks across independent game threads."""

from __future__ import annotations

from concurrent.futures import Future
from copy import deepcopy
import math
from queue import Empty, Queue
from threading import Lock, Thread, current_thread
from time import perf_counter

import my_board_engine as engine
import torch

from .encoding import ACTION_SIZE, _check_rules, encode_state, terminal_value
from .model import PolicyValueNet, masked_policy


class BatchedEvaluator:
    """Serve independent PUCT searches through one inference worker.

    Each game needs its own ``engine.PUCT`` object. Its synchronous callback
    waits on a Future, which releases the GIL while this worker gathers and
    evaluates a batch. A deadline flushes partial batches when fewer games are
    active than ``max_batch_size``. The caller's model is never modified.

    Use as a context manager or call ``close()`` after all game workers finish.
    Accepted requests drain on close; subsequent calls are rejected. A fatal
    inference failure is delivered to every outstanding and subsequent caller.
    """

    def __init__(
        self,
        model: PolicyValueNet,
        device: str | torch.device = "cpu",
        max_batch_size: int = 12,
        max_wait_ms: float = 0.0,
    ) -> None:
        if not isinstance(model, PolicyValueNet):
            raise TypeError("model must be a PolicyValueNet")
        if type(max_batch_size) is not int or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")
        if (isinstance(max_wait_ms, bool)
                or not isinstance(max_wait_ms, (int, float))
                or not math.isfinite(max_wait_ms) or max_wait_ms < 0):
            raise ValueError("max_wait_ms must be finite and non-negative")
        self.device = torch.device(device)
        self.max_batch_size = max_batch_size
        self.max_wait_ms = float(max_wait_ms)
        # Only the dedicated worker uses this detached, fixed inference model.
        # Explicit float32 keeps CPU/CUDA comparisons free of mixed precision.
        self.model = deepcopy(model).to(device=self.device, dtype=torch.float32)
        self.model.eval()
        self._requests: Queue = Queue()
        self._sentinel = object()
        self._lock = Lock()
        self._closed = False
        self._failure: BaseException | None = None
        self._pending = 0
        self._network_evaluations = 0
        self._inference_batches = 0
        self._max_observed_batch_size = 0
        self._batch_seconds = 0.0
        self._thread = Thread(
            target=self._serve, name="kingdom-batched-inference", daemon=True
        )
        self._thread.start()

    def __enter__(self) -> BatchedEvaluator:
        with self._lock:
            self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _check_open(self) -> None:
        # Call with self._lock held so failure/close cannot race submission.
        if self._failure is not None:
            raise self._failure
        if self._closed:
            raise RuntimeError("BatchedEvaluator is closed")

    def __call__(self, state: engine.State) -> tuple[list[float], float]:
        if not isinstance(state, engine.State):
            raise TypeError("state must be a my_board_engine.State")
        snapshot = state.copy()
        _check_rules(snapshot)
        future: Future = Future()
        with self._lock:
            self._check_open()
            if snapshot.result.finished():
                return [0.0] * ACTION_SIZE, terminal_value(snapshot)
            self._pending += 1
            # An unbounded queue cannot block while the lifecycle lock is held.
            self._requests.put((snapshot, future))
        return future.result()

    @property
    def stats(self) -> dict[str, int | float]:
        """Snapshot successful inference work, excluding terminal requests.

        ``batch_seconds`` includes device transfers, forward evaluation,
        synchronized CPU outputs and policy masking. It excludes queue waiting
        and CPU state encoding. ``reset_stats`` can exclude warmup work.
        """
        with self._lock:
            return {
                "network_evaluations": self._network_evaluations,
                "inference_batches": self._inference_batches,
                "mean_batch_size": (
                    self._network_evaluations / self._inference_batches
                    if self._inference_batches else 0.0
                ),
                "max_observed_batch_size": self._max_observed_batch_size,
                "batch_seconds": self._batch_seconds,
            }

    def reset_stats(self) -> None:
        """Clear measurements between runs when no request is outstanding."""
        with self._lock:
            if self._pending:
                raise RuntimeError("Cannot reset stats while requests are outstanding")
            self._network_evaluations = 0
            self._inference_batches = 0
            self._max_observed_batch_size = 0
            self._batch_seconds = 0.0

    def close(self) -> None:
        """Reject new requests, finish accepted ones, and join the worker."""
        with self._lock:
            if not self._closed:
                self._closed = True
                # All previously accepted requests precede this sentinel.
                self._requests.put(self._sentinel)
        if current_thread() is not self._thread:
            self._thread.join()

    def _evaluate_batch(self, requests):
        encoded = [encode_state(state, device="cpu") for state, _ in requests]
        features = torch.stack([item.features for item in encoded])
        legal_masks = torch.stack([item.legal_mask for item in encoded])
        started = perf_counter()
        logits, values = self.model(features.to(device=self.device))
        size = len(requests)
        if logits.shape != (size, ACTION_SIZE) or values.shape != (size,):
            raise ValueError("model returned unexpected policy/value shapes")
        if not logits.is_floating_point() or not values.is_floating_point():
            raise ValueError("model policy and values must be floating-point tensors")
        # One batched device-to-host copy replaces per-row item()/tolist()
        # synchronization. Masking and validation operate on ordinary CPU data.
        outputs = torch.cat((logits, values.unsqueeze(1)), dim=1).to(
            device="cpu", dtype=torch.float32
        )
        policy = masked_policy(outputs[:, :ACTION_SIZE], legal_masks)
        cpu_values = outputs[:, ACTION_SIZE]
        if (not bool(torch.isfinite(cpu_values).all())
                or bool((cpu_values.abs() > 1.0).any())):
            raise ValueError("model values must be finite and in [-1, 1]")
        predictions = list(zip(policy.tolist(), cpu_values.tolist()))
        return predictions, perf_counter() - started

    def _serve(self) -> None:
        active = []
        try:
            # inference_mode is thread-local and must be entered here.
            with torch.inference_mode():
                while True:
                    first = self._requests.get()
                    if first is self._sentinel:
                        return
                    active = [first]
                    stop_after_batch = False
                    deadline = perf_counter() + self.max_wait_ms / 1000.0
                    while len(active) < self.max_batch_size:
                        remaining = deadline - perf_counter()
                        try:
                            item = (
                                self._requests.get(timeout=remaining)
                                if remaining > 0 else self._requests.get_nowait()
                            )
                        except Empty:
                            break
                        if item is self._sentinel:
                            stop_after_batch = True
                            break
                        active.append(item)
                    predictions, elapsed = self._evaluate_batch(active)
                    with self._lock:
                        self._network_evaluations += len(active)
                        self._inference_batches += 1
                        self._max_observed_batch_size = max(
                            self._max_observed_batch_size, len(active)
                        )
                        self._batch_seconds += elapsed
                        self._pending -= len(active)
                    # Counters are finalized before result() wakes its caller.
                    for (_, future), prediction in zip(active, predictions):
                        future.set_result(prediction)
                    active = []
                    if stop_after_batch:
                        return
        except BaseException as error:
            # Atomically prevent submissions before draining every queue item.
            with self._lock:
                self._failure = error
                self._closed = True
                while True:
                    try:
                        item = self._requests.get_nowait()
                    except Empty:
                        break
                    if item is not self._sentinel:
                        active.append(item)
                self._pending = 0
            for _, future in active:
                if not future.done():
                    future.set_exception(error)
