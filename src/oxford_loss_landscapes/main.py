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


def _evaluate_plane(start_point, dir_one, dir_two, steps, metric, model_wrapper, distance=1.0, export=False):
    """
    Helper function to evaluate a planar region in parameter space.
    Evaluates a (steps x steps) grid with snake traversal so the center index
    ``(steps // 2, steps // 2)`` aligns with the original model parameters for
    odd ``steps``.
    """
    data_matrix = []

    for i in trange(steps, desc='Calculating Surface...'):
        data_column = []

        for j in range(steps):
            if i % 2 == 0:
                data_column.append(metric(model_wrapper))
                if j < steps - 1:
                    start_point.add_(dir_two)
            else:
                data_column.insert(0, metric(model_wrapper))
                if j < steps - 1:
                    start_point.sub_(dir_two)

        data_matrix.append(data_column)
        if i < steps - 1:
            start_point.add_(dir_one)

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

    data_filename = f"{filename}.npy"
    data_file_path = os.path.join(results_dir, data_filename)
    np.save(data_file_path, data_matrix)

    print(f"Saved plane data to {data_file_path}")


def _evaluate_plane_parallel(
    start_point, dir_one, dir_two, steps, metric, model_wrapper,
    distance=1.0, export=False, num_workers=None,
):
    """Parallel version of _evaluate_plane using Ray remote workers.

    Each row of the evaluation grid is dispatched to an independent Ray worker.
    Falls back to sequential evaluation if Ray is unavailable or fails.
    """
    def _sequential_eval(sp, d1, d2, steps_, metric_, wrapper_):
        return _evaluate_plane(sp, d1, d2, int(steps_), metric_, wrapper_, distance=distance, export=export)

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

    if num_workers is None:
        num_workers = min(steps, max(1, available_cpus))

    @ray.remote
    def eval_row(row_start_params, dir_two_params, steps_, metric_fn, wrapper_, row_idx):
        local_wrapper = copy.deepcopy(wrapper_)
        local_params = local_wrapper.get_module_parameters()
        with torch.no_grad():
            for idx in range(len(local_params)):
                local_params.parameters[idx].copy_(row_start_params[idx])
        data_column = []
        for j in range(int(steps_)):
            val = metric_fn(local_wrapper)
            data_column.append(val)
            if j < int(steps_) - 1:
                with torch.no_grad():
                    for idx in range(len(local_params)):
                        local_params.parameters[idx].add_(dir_two_params[idx])
        return data_column

    row_params = []
    current = start_point
    dir_two_data = [p.data.clone().detach() for p in dir_two]
    for i in range(int(steps)):
        row_params.append([p.data.clone().detach() for p in current])
        current = current + dir_one

    tasks = [
        eval_row.remote(row_params[i], dir_two_data, steps, metric, model_wrapper, i)
        for i in range(int(steps))
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
    connecting ``model_start`` to ``model_end``.
    """
    model_start_wrapper = wrap_model(copy.deepcopy(model_start) if deepcopy_model else model_start)
    end_model_wrapper = wrap_model(copy.deepcopy(model_end) if deepcopy_model else model_end)

    snapshot = model_start_wrapper.snapshot_parameters() if (not deepcopy_model and restore_model) else None
    try:
        start_point = model_start_wrapper.get_module_parameters()
        end_point = end_model_wrapper.get_module_parameters()
        span_steps = max(int(steps) - 1, 1)
        direction = (end_point - start_point) * (float(distance) / span_steps)

        data_values = []
        for idx in trange(int(steps), desc='Calculating...'):
            data_values.append(metric(model_start_wrapper))
            if idx < int(steps) - 1:
                start_point.add_(direction)
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

    For odd ``steps``, ``result[steps // 2]`` evaluates at the exact unperturbed
    parameters of ``model_start``. When ``deepcopy_model=False`` and
    ``restore_model=True`` (default), original parameters are restored in a
    ``finally`` block, making this safe for non-deepcopyable modules such as
    ``e3nn`` / TorchScript MLFFs and ASE calculators.
    """
    model_start_wrapper = wrap_model(copy.deepcopy(model_start) if deepcopy_model else model_start)
    snapshot = model_start_wrapper.snapshot_parameters() if (not deepcopy_model and restore_model) else None

    try:
        start_point = model_start_wrapper.get_module_parameters()
        direction = rand_n_like(start_point)
        _normalize_direction(direction, start_point, normalization)

        # Scale full half-span to start_point.model_norm() * distance
        _scale_direction(direction, start_point.model_norm(2) * float(distance))

        steps_int = int(steps)
        if steps_int <= 1:
            return np.array([metric(model_start_wrapper)])

        # Shift to -distance and step by (2 * distance) / (steps - 1) so
        # index 0 is -distance, center (steps // 2 for odd) is 0, and last is +distance.
        start_point.sub_(direction)
        step_vec = direction * (2.0 / float(steps_int - 1))

        data_values = []
        for idx in trange(steps_int, desc='Calculating...'):
            data_values.append(metric(model_start_wrapper))
            if idx < steps_int - 1:
                start_point.add_(step_vec)
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
    Evaluate ``metric`` over a 2D planar subspace defined by ``model_start`` and
    two endpoint models ``model_end_one`` and ``model_end_two``.
    """
    model_start_wrapper = wrap_model(copy.deepcopy(model_start) if deepcopy_model else model_start)
    model_end_one_wrapper = wrap_model(copy.deepcopy(model_end_one) if deepcopy_model else model_end_one)
    model_end_two_wrapper = wrap_model(copy.deepcopy(model_end_two) if deepcopy_model else model_end_two)

    snapshot = model_start_wrapper.snapshot_parameters() if (not deepcopy_model and restore_model) else None
    try:
        start_point = model_start_wrapper.get_module_parameters()
        steps_int = int(steps)
        if steps_int <= 1:
            return np.array([[metric(model_start_wrapper)]])
        span_steps = steps_int - 1

        if eigen_models:
            half_one = model_end_one_wrapper.get_module_parameters() * float(distance)
            half_two = model_end_two_wrapper.get_module_parameters() * float(distance)
        else:
            half_one = (model_end_one_wrapper.get_module_parameters() - start_point) * float(distance)
            half_two = (model_end_two_wrapper.get_module_parameters() - start_point) * float(distance)

        start_point.sub_(half_one)
        start_point.sub_(half_two)
        dir_one = half_one * (2.0 / float(span_steps))
        dir_two = half_two * (2.0 / float(span_steps))

        if use_ray or USE_PARALLEL:
            return _evaluate_plane_parallel(
                start_point, dir_one, dir_two, steps_int, metric, model_start_wrapper, num_workers=num_workers
            )
        return _evaluate_plane(start_point, dir_one, dir_two, steps_int, metric, model_start_wrapper)
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
    Evaluate ``metric`` along a planar subspace spanned by the dominant
    parameter-Hessian eigenvector (computed via VR-PCA) and a random orthogonal
    direction.

    Supports both standard tensor batches and graph/dict MLFF inputs via
    ``forward_fn`` or ``loss_callable``, and safely pads frozen
    (``requires_grad=False``) parameters with zeros.
    """
    from .hessian.vrpca import top_hessian_eigenpair_vrpca, VRPCAConfig
    from .hessian.vrpca.utils import collect_parameters, unflatten_like

    if isinstance(model, ModelWrapper):
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

        target_norm = start_point.model_norm(2) * float(distance)
        _scale_direction(half_one, target_norm)
        _scale_direction(half_two, target_norm)

        steps_int = int(steps)
        if steps_int <= 1:
            data = np.array([[metric(model_wrapper)]])
            if export:
                _export_plane_to_npy(data, distance)
                return None
            return data
        span_steps = steps_int - 1
        start_point.sub_(half_one)
        start_point.sub_(half_two)
        dir_one = half_one * (2.0 / float(span_steps))
        dir_two = half_two * (2.0 / float(span_steps))

        return _evaluate_plane(start_point, dir_one, dir_two, steps_int, metric, model_wrapper, distance, export)
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
    unperturbed parameters of ``model``. When ``deepcopy_model=False`` and
    ``restore_model=True`` (default), original parameters are restored in a
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

        steps_int = int(steps)
        if steps_int <= 1:
            data = np.array([[metric(model_start_wrapper)]])
            if export:
                _export_plane_to_npy(data, distance)
                return None
            return data
        span_steps = steps_int - 1
        start_point.sub_(half_one)
        start_point.sub_(half_two)
        dir_one = half_one * (2.0 / float(span_steps))
        dir_two = half_two * (2.0 / float(span_steps))

        if use_ray or USE_PARALLEL:
            return _evaluate_plane_parallel(
                start_point, dir_one, dir_two, steps_int, metric, model_start_wrapper, distance, export, num_workers
            )
        return _evaluate_plane(start_point, dir_one, dir_two, steps_int, metric, model_start_wrapper, distance, export)
    finally:
        if snapshot is not None:
            model_start_wrapper.restore_parameters(snapshot)
