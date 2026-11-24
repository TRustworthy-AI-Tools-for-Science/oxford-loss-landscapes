"""Public API entry points for VR-PCA Hessian solvers."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np
import torch

from .algorithms import top_eigenpair_vrpca
from .config import VRPCAConfig, VRPCAResult
from .hvp import HessianVectorProductOracle, _apply_to_device


def _resolve_device(module: torch.nn.Module, use_cuda: bool) -> torch.device:
    if use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return next(module.parameters()).device


def _move_module(module: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    return module.to(device=device)


def top_hessian_eigenpair_vrpca(
    net: torch.nn.Module,
    inputs: Any = None,
    targets: Any = None,
    criterion: Optional[Callable] = None,
    *,
    all_params: bool = True,
    use_cuda: bool = False,
    config: Optional[VRPCAConfig] = None,
    forward_fn: Optional[Callable[[torch.nn.Module, Any], Any]] = None,
    loss_callable: Optional[Callable[[Optional[Sequence[int]]], torch.Tensor]] = None,
    dataset_size: Optional[int] = None,
) -> VRPCAResult:
    """Compute the dominant Hessian eigenpair using VR-PCA."""

    device = _resolve_device(net, use_cuda)
    net = _move_module(net, device)
    inputs = _apply_to_device(inputs, device)
    targets = _apply_to_device(targets, device)

    oracle = HessianVectorProductOracle(
        model=net,
        loss_fn=criterion,
        inputs=inputs,
        targets=targets,
        all_params=all_params,
        forward_fn=forward_fn,
        loss_callable=loss_callable,
        dataset_size=dataset_size,
    )
    return top_eigenpair_vrpca(oracle, config=config)


def min_hessian_eigenpair_vrpca(
    net: torch.nn.Module,
    inputs: Any = None,
    targets: Any = None,
    criterion: Optional[Callable] = None,
    *,
    all_params: bool = True,
    use_cuda: bool = False,
    config: Optional[VRPCAConfig] = None,
    forward_fn: Optional[Callable[[torch.nn.Module, Any], Any]] = None,
    loss_callable: Optional[Callable[[Optional[Sequence[int]]], torch.Tensor]] = None,
    dataset_size: Optional[int] = None,
) -> VRPCAResult:
    """Compute the minimum Hessian eigenpair using VR-PCA by running on ``-H``."""

    device = _resolve_device(net, use_cuda)
    net = _move_module(net, device)
    inputs = _apply_to_device(inputs, device)
    targets = _apply_to_device(targets, device)

    oracle = HessianVectorProductOracle(
        model=net,
        loss_fn=criterion,
        inputs=inputs,
        targets=targets,
        all_params=all_params,
        scale=-1.0,
        forward_fn=forward_fn,
        loss_callable=loss_callable,
        dataset_size=dataset_size,
    )
    result = top_eigenpair_vrpca(oracle, config=config)
    return replace(result, eigenvalue=-result.eigenvalue)


def min_max_hessian_eigs_vrpca(
    net: torch.nn.Module,
    inputs: Any = None,
    targets: Any = None,
    criterion: Optional[Callable] = None,
    *,
    all_params: bool = True,
    use_cuda: bool = False,
    config: Optional[VRPCAConfig] = None,
    compute_min: bool = False,
    forward_fn: Optional[Callable[[torch.nn.Module, Any], Any]] = None,
    loss_callable: Optional[Callable[[Optional[Sequence[int]]], torch.Tensor]] = None,
    dataset_size: Optional[int] = None,
) -> Tuple[float, Optional[float], np.ndarray, Optional[np.ndarray], float]:
    """Return an interface similar to ``min_max_hessian_eigs`` using VR-PCA.

    Set ``compute_min=True`` to also estimate the minimum eigenpair with VR-PCA running on the
    negated Hessian. The returned tuple mirrors the classical helper, with the iteration counter
    replaced by the sum of VR-PCA HVP-equivalent calls.
    """

    result = top_hessian_eigenpair_vrpca(
        net=net,
        inputs=inputs,
        targets=targets,
        criterion=criterion,
        all_params=all_params,
        use_cuda=use_cuda,
        config=config,
        forward_fn=forward_fn,
        loss_callable=loss_callable,
        dataset_size=dataset_size,
    )
    min_value: Optional[float] = None
    min_vector: Optional[np.ndarray] = None
    total_cost = result.hvp_equivalent_calls

    if compute_min:
        min_result = min_hessian_eigenpair_vrpca(
            net=net,
            inputs=inputs,
            targets=targets,
            criterion=criterion,
            all_params=all_params,
            use_cuda=use_cuda,
            config=config,
            forward_fn=forward_fn,
            loss_callable=loss_callable,
            dataset_size=dataset_size,
        )
        min_value = min_result.eigenvalue
        min_vector = min_result.eigenvector.detach().cpu().numpy()
        total_cost += min_result.hvp_equivalent_calls

    return (
        result.eigenvalue,
        min_value,
        result.eigenvector.detach().cpu().numpy(),
        min_vector,
        total_cost,
    )
