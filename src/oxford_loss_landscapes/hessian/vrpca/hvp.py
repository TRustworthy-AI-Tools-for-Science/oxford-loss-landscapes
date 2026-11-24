"""Hessian-vector product utilities for VR-PCA solvers."""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Sequence

import torch

from .types import TensorList, TensorSequence
from .utils import (
    collect_parameters,
    flatten_tensors,
    normalize_vector,
    unflatten_like,
    with_no_grad_tensors,
)


def _apply_to_device(obj: Any, device: torch.device) -> Any:
    """Recursively move tensors, sequences, dicts, or graph objects to ``device``."""
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.to(device=device)
    if isinstance(obj, dict):
        return {key: _apply_to_device(val, device) for key, val in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [_apply_to_device(item, device) for item in obj]
        return type(obj)(converted) if isinstance(obj, tuple) else converted
    if hasattr(obj, "to") and callable(obj.to):
        return obj.to(device)
    return obj


def _select_batch(data: Any, indices: Optional[Sequence[int]]) -> Any:
    """Slice tensor/sequence batches by ``indices``; pass dicts/graphs through if full-batch."""
    if indices is None or data is None:
        return data
    if isinstance(data, torch.Tensor):
        return data[indices]
    if isinstance(data, (list, tuple)):
        sliced = [_select_batch(item, indices) for item in data]
        return type(data)(sliced) if isinstance(data, tuple) else sliced
    if isinstance(data, dict):
        # Single-graph dict batches (e.g. MACE AtomicData) cannot be sliced by
        # naive row indexing; return full dict when indices select sample 0.
        return data
    return data


class HVPBudgetTracker:
    """Keep track of full and minibatch Hessian-vector products."""

    def __init__(self) -> None:
        self.full_calls = 0
        self.minibatch_calls = 0
        self.minibatch_equivalent = 0.0

    def add_full(self, count: int = 1) -> None:
        self.full_calls += count

    def add_minibatch(self, batch_size: int, total_size: int) -> None:
        total_size = max(total_size, 1)
        self.minibatch_calls += 1
        self.minibatch_equivalent += float(batch_size) / float(total_size)

    def total_equivalent(self) -> float:
        return float(self.full_calls) + float(self.minibatch_equivalent)


class HessianVectorProductOracle:
    """Callable Hessian-vector products for a model/loss pair.

    Supports standard tensor batches as well as graph/dict inputs (such as
    MACE, UMA, or CHGNet MLFF batches) via ``forward_fn`` or ``loss_callable``,
    and safely handles multi-head/multi-task models with unused parameters.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: Optional[Callable] = None,
        inputs: Any = None,
        targets: Any = None,
        *,
        all_params: bool = True,
        scale: float = 1.0,
        forward_fn: Optional[Callable[[torch.nn.Module, Any], Any]] = None,
        loss_callable: Optional[Callable[[Optional[Sequence[int]]], torch.Tensor]] = None,
        dataset_size: Optional[int] = None,
    ) -> None:
        if loss_callable is None and loss_fn is None:
            raise ValueError("Provide either `loss_fn` or `loss_callable`.")

        self.model = model
        self.loss_fn = loss_fn
        self.forward_fn = forward_fn
        self.loss_callable = loss_callable
        self.params = collect_parameters(model, all_params)
        self.device = self.params[0].device
        self.inputs = _apply_to_device(inputs, self.device)
        self.targets = _apply_to_device(targets, self.device)
        self._explicit_dataset_size = dataset_size
        self._spectral_norm_estimate: Optional[float] = None
        self.scale = float(scale)

    @property
    def num_parameters(self) -> int:
        return int(sum(param.numel() for param in self.params))

    @property
    def dataset_size(self) -> int:
        if self._explicit_dataset_size is not None:
            return int(self._explicit_dataset_size)
        if isinstance(self.inputs, torch.Tensor):
            return int(self.inputs.shape[0])
        if isinstance(self.inputs, (list, tuple)) and self.inputs:
            first = self.inputs[0]
            if isinstance(first, torch.Tensor):
                return int(first.shape[0])
        if self.loss_callable is not None or isinstance(self.inputs, dict) or hasattr(self.inputs, "to"):
            return 1
        raise TypeError(
            "inputs must be a tensor, sequence of tensors, dict/graph batch, "
            "or paired with an explicit `dataset_size`/`loss_callable`"
        )

    def hvp(self, vector_list: TensorSequence, batch_indices: Optional[Sequence[int]] = None) -> TensorList:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        if self.loss_callable is not None:
            loss = self.loss_callable(batch_indices)
        else:
            if (
                batch_indices is not None
                and self.dataset_size > 1
                and (
                    isinstance(self.inputs, dict)
                    or (hasattr(self.inputs, "to") and not isinstance(self.inputs, torch.Tensor))
                )
            ):
                raise ValueError(
                    "Minibatching graph/dict inputs with dataset_size > 1 requires "
                    "an index-aware `loss_callable`."
                )
            xb = _apply_to_device(_select_batch(self.inputs, batch_indices), self.device)
            yb = _apply_to_device(_select_batch(self.targets, batch_indices), self.device)
            outputs = self.forward_fn(self.model, xb) if self.forward_fn is not None else self.model(xb)
            loss = self.loss_fn(outputs, yb) if yb is not None else self.loss_fn(outputs)

        raw_grads = torch.autograd.grad(
            loss,
            self.params,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )

        # Keep only parameters participating in the graph for second differentiation
        active_indices = [idx for idx, g in enumerate(raw_grads) if g is not None]
        if not active_indices:
            return [torch.zeros_like(param) for param in self.params]

        active_grads = [raw_grads[idx] for idx in active_indices]
        active_params = [self.params[idx] for idx in active_indices]
        active_vecs = [
            vector_list[idx].to(device=self.params[idx].device, dtype=self.params[idx].dtype)
            for idx in active_indices
        ]

        raw_hvp = torch.autograd.grad(
            active_grads,
            active_params,
            grad_outputs=active_vecs,
            retain_graph=False,
            allow_unused=True,
        )

        full_hvp: TensorList = [torch.zeros_like(param) for param in self.params]
        for pos, idx in enumerate(active_indices):
            if raw_hvp[pos] is not None:
                full_hvp[idx] = raw_hvp[pos]

        if self.scale != 1.0:
            return [tensor.mul(self.scale) for tensor in full_hvp]
        return full_hvp

    def full_hvp(self, vector_list: TensorSequence, tracker: Optional[HVPBudgetTracker] = None) -> TensorList:
        if tracker is not None:
            tracker.add_full()
        return self.hvp(vector_list, batch_indices=None)

    def minibatch_hvp(self, vector_list: TensorSequence, indices: Sequence[int], tracker: Optional[HVPBudgetTracker] = None) -> TensorList:
        if tracker is not None:
            tracker.add_minibatch(batch_size=len(indices), total_size=self.dataset_size)
        return self.hvp(vector_list, batch_indices=indices)

    def estimate_spectral_norm(self, power_iters: int = 5) -> float:
        vector = torch.randn(self.num_parameters, device=self.device, dtype=self.params[0].dtype)
        vector = normalize_vector(vector)
        norm_estimate = 0.0
        for _ in range(power_iters):
            vector_list = unflatten_like(vector, self.params)
            hvp = self.hvp(vector_list)
            hvp_flat = flatten_tensors(with_no_grad_tensors(hvp))
            norm_estimate = float(hvp_flat.norm().item())
            vector = normalize_vector(hvp_flat)
        self._spectral_norm_estimate = norm_estimate
        return norm_estimate

    def adaptive_step_size(self, default: float = 1e-2) -> float:
        if self._spectral_norm_estimate is None or math.isclose(self._spectral_norm_estimate, 0.0):
            self.estimate_spectral_norm()
        if self._spectral_norm_estimate is None or self._spectral_norm_estimate <= 0:
            return default
        return 0.5 / float(self._spectral_norm_estimate)
