"""
Functions for approximating loss/return landscapes in one and two dimensions.
"""
import copy
import datetime
import multiprocessing
import os
import traceback
import typing
import numpy as np
import torch.nn
from tqdm import trange

try:
    import ray
except ImportError:
    ray = None

from .model_interface.model_wrapper import ModelWrapper, wrap_model
from .model_interface.model_parameters import ModelParameters, rand_n_like, orthogonal_to
from .metrics.metric import Metric

USE_PARALLEL = False


def _scale_direction(direction: ModelParameters, target_norm: float) -> ModelParameters:
    """Scale ``direction`` in-place to have L2 norm ``target_norm`` (no-op if zero)."""
    dir_norm = direction.model_norm(2)
    if dir_norm > 1e-12:
        direction.mul_(target_norm / dir_norm)
    return direction


def _normalize_direction(
    direction: ModelParameters,
    ref_point: ModelParameters,
    normalization: typing.Optional[str],
) -> ModelParameters:
    """Apply requested parameter-space normalization in-place."""
    if normalization == 'model':
        direction.model_normalize_(ref_point)
    elif normalization == 'layer':
        direction.layer_normalize_(ref_point)
    elif normalization == 'filter':
        direction.filter_normalize_(ref_point)
    elif normalization is None:
        pass
    else:
        raise AttributeError(
            'Unsupported normalization argument. Supported values are model, layer, and filter'
        )
    return direction


def _set_params_offset(
    target_params: ModelParameters,
    base_tensors: typing.Sequence[torch.Tensor],
    alpha: float = 0.0,
    dir_one: typing.Optional[ModelParameters] = None,
    beta: float = 0.0,
    dir_two: typing.Optional[ModelParameters] = None,
) -> None:
    """Set ``target_params`` in-place to ``base + alpha * dir_one + beta * dir_two``.

    Computing every grid point directly from the immutable baseline snapshot
    eliminates cumulative floating-point drift across steps and guarantees that
    ``(alpha=0, beta=0)`` evaluates at the exact bit-for-bit original weights.
    """
    with torch.no_grad():
        for idx, base in enumerate(base_tensors):
            if (dir_one is None or alpha == 0.0) and (dir_two is None or beta == 0.0):
                target_params.parameters[idx].copy_(base)
                continue
            val = base.clone()
            if dir_one is not None and alpha != 0.0:
                val.add_(dir_one[idx], alpha=float(alpha))
            if dir_two is not None and beta != 0.0:
                val.add_(dir_two[idx], alpha=float(beta))
            target_params.parameters[idx].copy_(val)


def _centered_grid_coords(steps: int) -> np.ndarray:
    """Return symmetric 1D grid coordinates in ``[-1, +1]`` centered at ``0``.

    - For odd ``steps``, ``coords[steps // 2] == 0.0`` exactly.
    - For even ``steps``, ``coords`` is strictly symmetric around ``0.0``.
    - For ``steps <= 1``, returns ``np.array([0.0])``.
    """
    steps_int = int(steps)
    if steps_int <= 1:
        return np.array([0.0], dtype=float)
    coords = np.linspace(-1.0, 1.0, steps_int, dtype=float)
    if steps_int % 2 == 1:
        coords[steps_int // 2] = 0.0
    return coords


def _evaluate_plane(
    start_point,
    dir_one,
    dir_two,
    steps,
    metric,
    model_wrapper,
    distance=1.0,
    export=False,
    *,
    centered: bool = False,
):
    """Evaluate a 2D planar region in parameter space with zero cumulative drift.

    When ``centered=True``, ``start_point`` is the unperturbed center ``theta_0``
    and ``dir_one``/``dir_two`` are half-span vectors; grid coordinates span
    ``[-1, +1] x [-1, +1]`` so ``(steps // 2, steps // 2)`` evaluates at exact
    ``theta_0`` for odd ``steps`` and the grid is strictly symmetric around
    ``theta_0`` for even ``steps``.

    When ``centered=False`` (legacy direct-call mode), ``start_point`` is the
    grid corner and ``dir_one``/``dir_two`` are per-step increments evaluated at
    ``start_point + i * dir_one + j * dir_two`` for ``i, j in range(steps)``.
    """
    steps_int = int(steps)
    target_params = model_wrapper.get_module_parameters()
    base_tensors = [p.detach().clone() for p in start_point]
    coords = _centered_grid_coords(steps_int) if centered else np.arange(steps_int, dtype=float)

    data_matrix = []
    for alpha in trange(steps_int, desc='Calculating Surface...'):
        a_val = float(coords[alpha])
        data_column = []
        for beta in range(steps_int):
            b_val = float(coords[beta])
            _set_params_offset(
                target_params,
                base_tensors,
                alpha=a_val,
                dir_one=dir_one,
                beta=b_val,
                dir_two=dir_two,
            )
            data_column.append(metric(model_wrapper))
        data_matrix.append(data_column)

    if export:
        _export_plane_to_npy(data_matrix, distance)
    else:
        return np.array(data_matrix)


def _export_plane_to_npy(data_matrix, distance):
    """
    Exports the data_matrix as a .npy binary file in a folder called 'results' in the current working directory.
    """
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_LOLxAI"

    toml_filename = f"{filename}.toml"
    with open(os.path.join(results_dir, toml_filename), "w", encoding="utf-8") as f:
        f.write(f'distance = {float(distance)}\n')
        f.write(f'coordinate_min = {-float(distance)}\n')
        f.write(f'coordinate_max = {float(distance)}\n')

    data_filename = f"{filename}.npy"
    data_file_path = os.path.join(results_dir, data_filename)
    np.save(data_file_path, data_matrix)

    print(f"Saved plane data to {data_file_path}")


def _evaluate_plane_parallel(
    start_point,
    dir_one,
    dir_two,
    steps,
    metric,
    model_wrapper,
    distance=1.0,
    export=False,
    num_workers=None,
    *,
    centered: bool = False,
):
    """Parallel version of _evaluate_plane using Ray remote workers.

    Each row of the evaluation grid is dispatched to an independent Ray worker
    and computed directly from the baseline tensors (zero cumulative drift,
    exact match with ``_evaluate_plane``). Falls back to sequential evaluation
    if Ray is unavailable or fails.
    """
    def _sequential_eval(sp, d1, d2, steps_, metric_, wrapper_):
        return _evaluate_plane(
            sp,
            d1,
            d2,
            int(steps_),
            metric_,
            wrapper_,
            distance=distance,
            export=export,
            centered=centered,
        )

    if ray is None or not ray.is_initialized():
        try:
            if ray is not None:
                ray.init(ignore_reinit_error=True)
        except Exception:
            return _sequential_eval(start_point, dir_one, dir_two, steps, metric, model_wrapper)

    if ray is None:
        return _sequential_eval(start_point, dir_one, dir_two, steps, metric, model_wrapper)

    try:
        available_cpus = int(ray.available_resources().get("CPU", multiprocessing.cpu_count()))
    except Exception:
        available_cpus = multiprocessing.cpu_count()

    steps_int = int(steps)
    if num_workers is None:
        num_workers = min(steps_int, max(1, available_cpus))

    @ray.remote
    def eval_row(base_params, dir_one_params, dir_two_params, alpha_val, beta_coords, metric_fn, wrapper_):
        local_wrapper = copy.deepcopy(wrapper_)
        local_params = local_wrapper.get_module_parameters()
        data_column = []
        for b_val in beta_coords:
            with torch.no_grad():
                for idx, base in enumerate(base_params):
                    if alpha_val == 0.0 and b_val == 0.0:
                        local_params.parameters[idx].copy_(base)
                    else:
                        val = base.clone()
                        if alpha_val != 0.0:
                            val.add_(dir_one_params[idx], alpha=float(alpha_val))
                        if b_val != 0.0:
                            val.add_(dir_two_params[idx], alpha=float(b_val))
                        local_params.parameters[idx].copy_(val)
            data_column.append(metric_fn(local_wrapper))
        return data_column

    coords = _centered_grid_coords(steps_int) if centered else np.arange(steps_int, dtype=float)
    base_data = [p.detach().clone() for p in start_point]
    dir_one_data = [p.detach().clone() for p in dir_one]
    dir_two_data = [p.detach().clone() for p in dir_two]

    tasks = [
        eval_row.remote(
            base_data,
            dir_one_data,
            dir_two_data,
            float(coords[i]),
            coords.tolist(),
            metric,
            model_wrapper,
        )
        for i in range(steps_int)
    ]

    try:
        results = ray.get(tasks)
    except Exception:
        print("Warning: Ray remote execution failed, falling back to sequential evaluation.")
        traceback.print_exc()
        return _sequential_eval(start_point, dir_one, dir_two, steps, metric, model_wrapper)

    data_matrix = np.array(results)
    if export:
        _export_plane_to_npy(data_matrix, distance)
    else:
        return data_matrix


def point(model: typing.Union[torch.nn.Module, ModelWrapper], metric: Metric) -> tuple:
    """
    Returns the computed value of the evaluation function applied to the model
    or agent at a specific point in parameter space.
    """
    return metric(wrap_model(model))


def linear_interpolation(
    model_start: typing.Union[torch.nn.Module, ModelWrapper],
    model_end: typing.Union[torch.nn.Module, ModelWrapper],
    metric: Metric,
    steps=100,
    deepcopy_model=False,
    distance=1.0,
    restore_model: bool = True,
) -> np.ndarray:
    """
    Evaluate ``metric`` at ``steps`` evenly spaced points along the line
    connecting ``model_start`` to ``model_start + distance * (model_end - model_start)``.
    """
    model_start_wrapper = wrap_model(copy.deepcopy(model_start) if deepcopy_model else model_start)
    end_model_wrapper = wrap_model(copy.deepcopy(model_end) if deepcopy_model else model_end)

    snapshot = model_start_wrapper.snapshot_parameters() if (not deepcopy_model and restore_model) else None
    try:
        start_point = model_start_wrapper.get_module_parameters()
        end_point = end_model_wrapper.get_module_parameters()
        base_tensors = [p.detach().clone() for p in start_point]
        delta = end_point - start_point

        steps_int = int(steps)
        ts = np.array([0.0], dtype=float) if steps_int <= 1 else np.linspace(0.0, float(distance), steps_int)

        data_values = []
        for idx in trange(steps_int, desc='Calculating...'):
            _set_params_offset(start_point, base_tensors, alpha=float(ts[idx]), dir_one=delta)
            data_values.append(metric(model_start_wrapper))
    finally:
        if snapshot is not None:
            model_start_wrapper.restore_parameters(snapshot)

    return np.array(data_values)


def random_line(
    model_start: typing.Union[torch.nn.Module, ModelWrapper],
    metric: Metric,
    distance=0.1,
    steps=100,
    normalization='filter',
    deepcopy_model=False,
    restore_model: bool = True,
) -> np.ndarray:
    """
    Evaluate ``metric`` along a symmetric 1D random direction spanning
    ``[-distance, +distance]`` centered at ``model_start``.

    For odd ``steps``, ``result[steps // 2]`` evaluates at the exact bit-for-bit
    unperturbed parameters of ``model_start``. For even ``steps``, grid points
    are strictly symmetric around ``model_start``. When ``deepcopy_model=False``
    and ``restore_model=True`` (default), original parameters are restored in a
    ``finally`` block, making this safe for non-deepcopyable modules such as
    ``e3nn`` / TorchScript MLFFs and ASE calculators.
    """
    model_start_wrapper = wrap_model(copy.deepcopy(model_start) if deepcopy_model else model_start)
    snapshot = model_start_wrapper.snapshot_parameters() if (not deepcopy_model and restore_model) else None

    try:
        start_point = model_start_wrapper.get_module_parameters()
        base_tensors = [p.detach().clone() for p in start_point]
        direction = rand_n_like(start_point)
        _normalize_direction(direction, start_point, normalization)

        # Scale full half-span to start_point.model_norm() * distance
        _scale_direction(direction, start_point.model_norm(2) * float(distance))

        steps_int = int(steps)
        coords = _centered_grid_coords(steps_int)

        data_values = []
        for idx in trange(steps_int, desc='Calculating...'):
            _set_params_offset(start_point, base_tensors, alpha=float(coords[idx]), dir_one=direction)
            data_values.append(metric(model_start_wrapper))
    finally:
        if snapshot is not None:
            model_start_wrapper.restore_parameters(snapshot)

    return np.array(data_values)


def planar_interpolation(
    model_start: typing.Union[torch.nn.Module, ModelWrapper],
    model_end_one: typing.Union[torch.nn.Module, ModelWrapper],
    model_end_two: typing.Union[torch.nn.Module, ModelWrapper],
    metric: Metric,
    distance=1.0,
    steps=20,
    deepcopy_model=False,
    eigen_models=False,
    use_ray=False,
    num_workers=None,
    restore_model: bool = True,
) -> np.ndarray:
    """
    Evaluate ``metric`` over a 2D planar subspace centered at ``model_start``
    and spanned by ``model_end_one`` and ``model_end_two``.

    For odd ``steps``, ``result[steps // 2, steps // 2]`` evaluates at the exact
    unperturbed parameters of ``model_start``.
    """
    model_start_wrapper = wrap_model(copy.deepcopy(model_start) if deepcopy_model else model_start)
    model_end_one_wrapper = wrap_model(copy.deepcopy(model_end_one) if deepcopy_model else model_end_one)
    model_end_two_wrapper = wrap_model(copy.deepcopy(model_end_two) if deepcopy_model else model_end_two)

    snapshot = model_start_wrapper.snapshot_parameters() if (not deepcopy_model and restore_model) else None
    try:
        start_point = model_start_wrapper.get_module_parameters()
        steps_int = int(steps)

        if eigen_models:
            half_one = model_end_one_wrapper.get_module_parameters() * float(distance)
            half_two = model_end_two_wrapper.get_module_parameters() * float(distance)
        else:
            half_one = (model_end_one_wrapper.get_module_parameters() - start_point) * float(distance)
            half_two = (model_end_two_wrapper.get_module_parameters() - start_point) * float(distance)

        if use_ray or USE_PARALLEL:
            return _evaluate_plane_parallel(
                start_point,
                half_one,
                half_two,
                steps_int,
                metric,
                model_start_wrapper,
                distance=distance,
                num_workers=num_workers,
                centered=True,
            )
        return _evaluate_plane(
            start_point,
            half_one,
            half_two,
            steps_int,
            metric,
            model_start_wrapper,
            distance=distance,
            centered=True,
        )
    finally:
        if snapshot is not None:
            model_start_wrapper.restore_parameters(snapshot)


def hessian_plane(
    model: typing.Union[torch.nn.Module, ModelWrapper],
    metric: Metric,
    criterion=None,
    inputs=None,
    targets=None,
    *,
    distance: float = 1.0,
    steps: int = 20,
    deepcopy_model: bool = False,
    vrpca_config=None,
    export: bool = False,
    restore_model: bool = True,
    forward_fn=None,
    loss_callable=None,
    dataset_size: typing.Optional[int] = None,
) -> np.ndarray:
    """
    Evaluate ``metric`` along a planar subspace centered at ``model`` and
    spanned by the dominant parameter-Hessian eigenvector (computed via VR-PCA)
    and a random orthogonal direction.

    For odd ``steps``, ``result[steps // 2, steps // 2]`` evaluates at the exact
    bit-for-bit unperturbed parameters of ``model``. Supports both standard
    tensor batches and graph/dict MLFF inputs via ``forward_fn`` or
    ``loss_callable``, and safely pads frozen (``requires_grad=False``)
    parameters with zeros.
    """
    from .hessian.vrpca import top_hessian_eigenpair_vrpca, VRPCAConfig
    from .hessian.vrpca.utils import collect_parameters, unflatten_like

    if isinstance(model, ModelWrapper):
        if len(model.modules) != 1:
            raise ValueError("hessian_plane with VR-PCA requires exactly one module.")
        raw_module = model.modules[0]
    else:
        raw_module = model

    cfg = vrpca_config or VRPCAConfig()
    result = top_hessian_eigenpair_vrpca(
        net=raw_module,
        inputs=inputs,
        targets=targets,
        criterion=criterion,
        config=cfg,
        forward_fn=forward_fn,
        loss_callable=loss_callable,
        dataset_size=dataset_size,
    )

    # Map differentiable eigenvector back onto full parameter list (pad frozen params with 0)
    diff_params = collect_parameters(raw_module, all_params=True)
    eigvec_diff = unflatten_like(result.eigenvector.detach(), diff_params)
    diff_map = {id(p): v for p, v in zip(diff_params, eigvec_diff)}

    model_wrapper = wrap_model(copy.deepcopy(model) if deepcopy_model else model)
    snapshot = model_wrapper.snapshot_parameters() if (not deepcopy_model and restore_model) else None

    try:
        start_point = model_wrapper.get_module_parameters()
        orig_params = wrap_model(model).get_module_parameters() if deepcopy_model else start_point
        full_eigvec = [
            diff_map[id(orig_p)].clone().to(device=cur_p.device, dtype=cur_p.dtype)
            if id(orig_p) in diff_map
            else torch.zeros_like(cur_p)
            for orig_p, cur_p in zip(orig_params, start_point)
        ]
        half_one = ModelParameters(full_eigvec)
        half_two = orthogonal_to(half_one)
        with torch.no_grad():
            for orig_p, direction_p in zip(orig_params, half_two):
                if not orig_p.requires_grad:
                    direction_p.zero_()

        target_norm = start_point.model_norm(2) * float(distance)
        _scale_direction(half_one, target_norm)
        _scale_direction(half_two, target_norm)

        return _evaluate_plane(
            start_point,
            half_one,
            half_two,
            int(steps),
            metric,
            model_wrapper,
            distance=distance,
            export=export,
            centered=True,
        )
    finally:
        if snapshot is not None:
            model_wrapper.restore_parameters(snapshot)


def random_plane(
    model: typing.Union[torch.nn.Module, ModelWrapper],
    metric: Metric,
    distance=1,
    steps=20,
    normalization='filter',
    deepcopy_model=False,
    export=False,
    use_ray=False,
    num_workers=None,
    restore_model: bool = True,
) -> np.ndarray:
    """
    Evaluate ``metric`` over a symmetric 2D random plane spanning
    ``[-distance, +distance]^2`` centered at ``model``.

    For odd ``steps``, ``result[steps // 2, steps // 2]`` evaluates at the exact
    bit-for-bit unperturbed parameters of ``model``. For even ``steps``, the
    grid is strictly symmetric around ``model``. When ``deepcopy_model=False``
    and ``restore_model=True`` (default), original parameters are restored in a
    ``finally`` block.
    """
    model_start_wrapper = wrap_model(copy.deepcopy(model) if deepcopy_model else model)
    snapshot = model_start_wrapper.snapshot_parameters() if (not deepcopy_model and restore_model) else None

    try:
        start_point = model_start_wrapper.get_module_parameters()
        half_one = rand_n_like(start_point)
        half_two = orthogonal_to(half_one)

        _normalize_direction(half_one, start_point, normalization)
        _normalize_direction(half_two, start_point, normalization)

        target_norm = start_point.model_norm(2) * float(distance)
        _scale_direction(half_one, target_norm)
        _scale_direction(half_two, target_norm)

        if use_ray or USE_PARALLEL:
            return _evaluate_plane_parallel(
                start_point,
                half_one,
                half_two,
                int(steps),
                metric,
                model_start_wrapper,
                distance=distance,
                export=export,
                num_workers=num_workers,
                centered=True,
            )
        return _evaluate_plane(
            start_point,
            half_one,
            half_two,
            int(steps),
            metric,
            model_start_wrapper,
            distance=distance,
            export=export,
            centered=True,
        )
    finally:
        if snapshot is not None:
            model_start_wrapper.restore_parameters(snapshot)
