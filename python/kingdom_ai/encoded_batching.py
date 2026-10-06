"""Batch already encoded C++ leaf requests without Python engine calls."""

from __future__ import annotations

from concurrent.futures import Future
from copy import deepcopy
import math
from queue import Empty, Queue
from threading import Lock, Thread, current_thread
from time import perf_counter

import numpy as np
import torch

from .encoding import ACTION_SIZE, BOARD_SIZE, INPUT_CHANNELS
from .model import PolicyValueNet, masked_policy


class EncodedBatchedEvaluator:
    """Merge leaf batches from independent searches into one model service.

    Inputs are owned snapshots of float32 ``[N,10,9,9]`` features and bool
    ``[N,82]`` masks. No engine state is reconstructed or encoded in Python.
    Each request must fit ``max_batch_size`` (a *row*, not request, limit).
    Returned float64 policy/value arrays belong to the calling request.
    Accepted requests drain on close. A fatal model failure releases all
    current and future callers; invalid input alone does not poison service.
    """

    def __init__(
        self, model: PolicyValueNet, device: str | torch.device = "cpu",
        max_batch_size: int = 96, max_wait_ms: float = 0.0,
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
        self.model = deepcopy(model).to(device=self.device, dtype=torch.float32)
        self.model.eval()
        self._requests: Queue = Queue()
        self._sentinel = object()
        self._lock = Lock()
        self._closed = False
        self._failure: BaseException | None = None
        self._pending = 0
        self._clear_counters()
        self._thread = Thread(target=self._serve, name="kingdom-encoded-inference", daemon=True)
        self._thread.start()

    def __enter__(self) -> EncodedBatchedEvaluator:
        with self._lock:
            self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _check_open(self) -> None:
        if self._failure is not None:
            raise self._failure
        if self._closed:
            raise RuntimeError("EncodedBatchedEvaluator is closed")

    def __call__(self, features: np.ndarray, legal_masks: np.ndarray):
        if not isinstance(features, np.ndarray) or not isinstance(legal_masks, np.ndarray):
            raise TypeError("features and legal_masks must be numpy arrays")
        if features.dtype != np.float32 or legal_masks.dtype != np.bool_:
            raise TypeError("features must be float32 and legal_masks must be bool")
        if features.ndim != 4 or features.shape[1:] != (INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE):
            raise ValueError("features must have shape [N, 10, 9, 9]")
        size = features.shape[0]
        if size < 1 or size > self.max_batch_size:
            raise ValueError("request rows must be between 1 and max_batch_size")
        if legal_masks.shape != (size, ACTION_SIZE):
            raise ValueError("legal_masks must have shape [N, 82]")
        # The C++ callback may hand out a view into temporary storage. Retain
        # independent contiguous snapshots until inference and result delivery.
        features = np.array(features, dtype=np.float32, order="C", copy=True)
        legal_masks = np.array(legal_masks, dtype=np.bool_, order="C", copy=True)
        if not np.isfinite(features).all() or (features < 0).any() or (features > 1).any():
            raise ValueError("encoded features must be finite and in [0, 1]")
        if not legal_masks.any(axis=1).all():
            raise ValueError("nonterminal requests must have a legal action in every row")
        future: Future = Future()
        with self._lock:
            self._check_open()
            self._pending += 1
            self._requests.put((features, legal_masks, future, perf_counter()))
        return future.result()

    def _clear_counters(self) -> None:
        self._network_evaluations = 0
        self._inference_batches = 0
        self._submitted_requests = 0
        self._max_observed_batch_size = 0
        self._batch_seconds = 0.0
        self._host_preparation_seconds = 0.0
        self._transfer_forward_seconds = 0.0
        self._policy_seconds = 0.0
        self._queue_seconds = 0.0

    @property
    def stats(self) -> dict[str, int | float]:
        """Successful rows/batches and summed service timings, not wall shares.

        ``transfer_forward_seconds`` includes H2D, forward and synchronized
        D2H together; it does not pretend to separate asynchronous CUDA work.
        ``queue_seconds`` sums per-request waiting and can exceed wall time.
        CPU engine encoding happens before this service and is not included.
        """
        with self._lock:
            return {
                "network_evaluations": self._network_evaluations,
                "inference_batches": self._inference_batches,
                "submitted_requests": self._submitted_requests,
                "mean_batch_size": self._network_evaluations / self._inference_batches
                    if self._inference_batches else 0.0,
                "mean_request_size": self._network_evaluations / self._submitted_requests
                    if self._submitted_requests else 0.0,
                "max_observed_batch_size": self._max_observed_batch_size,
                "batch_seconds": self._batch_seconds,
                "host_preparation_seconds": self._host_preparation_seconds,
                "transfer_forward_seconds": self._transfer_forward_seconds,
                "policy_seconds": self._policy_seconds,
                "queue_seconds": self._queue_seconds,
            }

    def reset_stats(self) -> None:
        with self._lock:
            if self._pending:
                raise RuntimeError("Cannot reset stats while requests are outstanding")
            self._clear_counters()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._requests.put(self._sentinel)
        if current_thread() is not self._thread:
            self._thread.join()

    def _evaluate_batch(self, requests):
        started = perf_counter()
        features = torch.from_numpy(np.concatenate([item[0] for item in requests], axis=0))
        legal_masks = torch.from_numpy(np.concatenate([item[1] for item in requests], axis=0))
        prepared = perf_counter()
        logits, values = self.model(features.to(device=self.device))
        size = features.shape[0]
        if not isinstance(logits, torch.Tensor) or not isinstance(values, torch.Tensor):
            raise TypeError("model policy and values must be tensors")
        if logits.shape != (size, ACTION_SIZE) or values.shape != (size,):
            raise ValueError("model returned unexpected policy/value shapes")
        if not logits.is_floating_point() or not values.is_floating_point():
            raise TypeError("model policy and values must be floating-point tensors")
        outputs = torch.cat((logits, values.unsqueeze(1)), dim=1).to(
            device="cpu", dtype=torch.float32
        )
        transferred = perf_counter()
        policy = masked_policy(outputs[:, :ACTION_SIZE], legal_masks)
        cpu_values = outputs[:, ACTION_SIZE]
        if (not bool(torch.isfinite(cpu_values).all())
                or bool((cpu_values.abs() > 1.0).any())):
            raise ValueError("model values must be finite and in [-1, 1]")
        policies = policy.numpy().astype(np.float64, copy=True)
        values_array = cpu_values.numpy().astype(np.float64, copy=True)
        predictions = []
        offset = 0
        for features_array, _, _, _ in requests:
            end = offset + len(features_array)
            predictions.append((policies[offset:end].copy(), values_array[offset:end].copy()))
            offset = end
        finished = perf_counter()
        return predictions, size, (finished - started, prepared - started,
                                   transferred - prepared, finished - transferred)

    def _serve(self) -> None:
        active = []
        carry = None
        try:
            with torch.inference_mode():
                while True:
                    first = carry if carry is not None else self._requests.get()
                    carry = None
                    if first is self._sentinel:
                        return
                    active = [first]
                    size = len(first[0])
                    stop_after_batch = False
                    deadline = perf_counter() + self.max_wait_ms / 1000.0
                    while size < self.max_batch_size:
                        remaining = deadline - perf_counter()
                        try:
                            item = (self._requests.get(timeout=remaining) if remaining > 0
                                    else self._requests.get_nowait())
                        except Empty:
                            break
                        if item is self._sentinel:
                            stop_after_batch = True
                            break
                        if size + len(item[0]) > self.max_batch_size:
                            carry = item
                            break
                        active.append(item)
                        size += len(item[0])
                    inference_started = perf_counter()
                    queue_seconds = sum(inference_started - item[3] for item in active)
                    predictions, size, timings = self._evaluate_batch(active)
                    with self._lock:
                        self._network_evaluations += size
                        self._inference_batches += 1
                        self._submitted_requests += len(active)
                        self._max_observed_batch_size = max(self._max_observed_batch_size, size)
                        self._batch_seconds += timings[0]
                        self._host_preparation_seconds += timings[1]
                        self._transfer_forward_seconds += timings[2]
                        self._policy_seconds += timings[3]
                        self._queue_seconds += queue_seconds
                        self._pending -= len(active)
                    for (_, _, future, _), prediction in zip(active, predictions):
                        future.set_result(prediction)
                    active = []
                    if stop_after_batch:
                        return
        except BaseException as error:
            with self._lock:
                self._failure = error
                self._closed = True
                if carry is not None:
                    active.append(carry)
                while True:
                    try:
                        item = self._requests.get_nowait()
                    except Empty:
                        break
                    if item is not self._sentinel:
                        active.append(item)
                self._pending = 0
            for _, _, future, _ in active:
                if not future.done():
                    future.set_exception(error)
