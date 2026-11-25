"""Regression tests for 0-D params, line/plane alignment, restoration, and MLFF graphs."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn


class _SurrogateMultiHeadMLFF(nn.Module):
    """Minimal multi-head dict-input surrogate with 0-D scalar & frozen params."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(3, 6)
        self.head_active = nn.Linear(6, 1)
        self.head_unused = nn.Linear(6, 1)
        # 0-D scalar parameter (common in MACE/UMA/CHGNet scale/shift layers)
        self.energy_scale = nn.Parameter(torch.tensor(1.5))
        # Frozen parameter (e.g. fixed atomic reference energies)
        self.atomic_shift = nn.Parameter(torch.tensor(-2.0), requires_grad=False)

    def forward(self, batch: dict) -> dict:
        pos = batch["positions"]
        feat = torch.tanh(self.encoder(pos))
        per_atom = self.head_active(feat) * self.energy_scale + self.atomic_shift
        return {"energy": per_atom.sum()}


def test_filter_and_model_normalize_handle_0d_params_and_exact_norm():
    from oxford_loss_landscapes.model_interface.model_parameters import (
        ModelParameters,
        rand_n_like,
    )

    torch.manual_seed(0)
    model = _SurrogateMultiHeadMLFF()
    ref = ModelParameters([p for p in model.parameters() if p.is_floating_point()])

    # 1. filter_normalize_ must not crash on 0-D scalar parameters
    d_filter = rand_n_like(ref)
    d_filter.filter_normalize_(ref)
    for param, ref_p in zip(d_filter, ref):
        if param.ndim == 0:
            assert torch.isclose(param.abs(), ref_p.abs(), atol=1e-5)

    # 2. model_normalize_ must match ref.model_norm(2) in a single pass
    d_model = rand_n_like(ref)
    d_model.model_normalize_(ref)
    assert d_model.model_norm(2) == pytest.approx(ref.model_norm(2), rel=1e-5)

    # 3. layer_normalize_ must handle zero-norm layers without NaN
    zero_ref = ModelParameters([torch.zeros(4), torch.ones(3, 3)])
    d_layer = rand_n_like(zero_ref)
    d_layer.layer_normalize_(zero_ref)
    assert torch.all(torch.isfinite(d_layer[0]))
    assert torch.all(torch.isfinite(d_layer[1]))


def test_random_line_respects_distance_centers_and_restores_weights():
    import oxford_loss_landscapes as oll
    from oxford_loss_landscapes.metrics.sl_metrics import Loss

    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 1))
    X = torch.randn(16, 4)
    y = torch.randn(16, 1)
    criterion = nn.MSELoss()
    metric = Loss(criterion, X, y)

    orig_weights = [p.detach().clone() for p in model.parameters()]
    orig_loss = criterion(model(X), y).item()

    torch.manual_seed(7)
    line_small = oll.random_line(
        model, metric, distance=0.01, steps=11, normalization="filter", deepcopy_model=False
    )
    # Weights must be restored identically even with deepcopy_model=False
    for p, saved in zip(model.parameters(), orig_weights):
        assert torch.allclose(p, saved)

    torch.manual_seed(7)
    line_large = oll.random_line(
        model, metric, distance=0.5, steps=11, normalization="filter", deepcopy_model=False
    )
    for p, saved in zip(model.parameters(), orig_weights):
        assert torch.allclose(p, saved)

    assert line_small[5] == pytest.approx(orig_loss, abs=1e-5)
    assert line_large[5] == pytest.approx(orig_loss, abs=1e-5)
    assert np.ptp(line_large) > np.ptp(line_small) * 5.0


def test_random_plane_exact_center_and_inplace_restore():
    import oxford_loss_landscapes as oll
    from oxford_loss_landscapes.metrics.metric import Metric
    from oxford_loss_landscapes.metrics.sl_metrics import Loss

    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(3, 6), nn.Tanh(), nn.Linear(6, 1))
    X = torch.randn(12, 3)
    y = torch.randn(12, 1)
    criterion = nn.MSELoss()
    metric = Loss(criterion, X, y)

    orig_weights = [p.detach().clone() for p in model.parameters()]
    orig_loss = criterion(model(X), y).item()

    # Larger odd grid (21x21) on float32 must hit exact bit-for-bit origin (zero drift)
    plane = oll.random_plane(
        model, metric, distance=0.25, steps=21, normalization="filter", deepcopy_model=False
    )
    assert plane.shape == (21, 21)
    assert plane[10, 10] == pytest.approx(orig_loss, abs=1e-12)
    for p, saved in zip(model.parameters(), orig_weights):
        assert torch.allclose(p, saved)

    # Pure quadratic bowl centered at orig_weights must be strictly symmetric
    # for EVEN steps (no half-step shift!) and hit exact 0.0 at center for ODD steps.
    class _QuadraticDistanceMetric(Metric):
        def __call__(self, model_wrapper):
            return float(
                sum(
                    torch.sum((cur - ref) ** 2).item()
                    for cur, ref in zip(model_wrapper.get_module_parameters(), orig_weights)
                )
            )

    quad_metric = _QuadraticDistanceMetric()
    even_plane = oll.random_plane(
        model, quad_metric, distance=0.5, steps=6, normalization="filter", deepcopy_model=False
    )
    assert even_plane.shape == (6, 6)
    assert np.allclose(even_plane, even_plane[::-1, ::-1], atol=1e-5)

    even_line = oll.random_line(
        model, quad_metric, distance=0.5, steps=6, normalization="filter", deepcopy_model=False
    )
    assert even_line.shape == (6,)
    assert np.allclose(even_line, even_line[::-1], atol=1e-5)

    odd_line = oll.random_line(
        model, quad_metric, distance=0.5, steps=21, normalization="filter", deepcopy_model=False
    )
    assert odd_line[10] == 0.0
    assert np.allclose(odd_line, odd_line[::-1], atol=1e-5)


def test_vrpca_and_hessian_plane_on_multihead_dict_mlff():
    import oxford_loss_landscapes as oll
    from oxford_loss_landscapes.metrics.metric import Metric

    torch.manual_seed(3)
    model = _SurrogateMultiHeadMLFF()
    batch = {"positions": torch.randn(5, 3)}
    target_energy = torch.tensor(-5.0)
    orig_weights = [p.detach().clone() for p in model.parameters()]

    def loss_callable(_indices):
        out = model(batch)
        return (out["energy"] - target_energy) ** 2

    class _EnergyLossMetric(Metric):
        def __call__(self, model_wrapper):
            net = model_wrapper.modules[0]
            with torch.no_grad():
                return float(((net(batch)["energy"] - target_energy) ** 2).item())

    cfg = oll.VRPCAConfig(batch_size=1, epochs=4, seed=42)
    result = oll.top_hessian_eigenpair_vrpca(
        net=model,
        loss_callable=loss_callable,
        dataset_size=1,
        config=cfg,
    )
    assert isinstance(result.eigenvalue, float)
    assert np.isfinite(result.eigenvalue)

    plane = oll.hessian_plane(
        model,
        metric=_EnergyLossMetric(),
        loss_callable=loss_callable,
        dataset_size=1,
        distance=0.1,
        steps=5,
        deepcopy_model=False,
        vrpca_config=cfg,
    )
    assert plane.shape == (5, 5)
    assert np.all(np.isfinite(plane))
    center_expected = float(((model(batch)["energy"] - target_energy) ** 2).item())
    assert plane[2, 2] == pytest.approx(center_expected, abs=1e-5)
    for p, saved in zip(model.parameters(), orig_weights):
        assert torch.allclose(p, saved)


def test_force_loss_vrpca_float64_hessian_and_metrics():
    """Verify conservative force-matching loss ||-grad_q E_theta - F_*||^2,
    fd_grad HVP fallback, float64 classical Hessians, and metric state safety."""
    import oxford_loss_landscapes as oll
    from oxford_loss_landscapes.hessian.hessian import (
        hessian_vector_product,
        small_hessian,
    )
    from oxford_loss_landscapes.metrics.sl_metrics import (
        LossGradient,
        LossPerturbations,
    )

    torch.manual_seed(11)
    model = _SurrogateMultiHeadMLFF().double()
    model.trainable_shift = nn.Parameter(torch.tensor(0.25, dtype=torch.float64))

    pos_base = torch.randn(4, 3, dtype=torch.float64)
    target_forces = torch.randn(4, 3, dtype=torch.float64)

    def force_loss_callable(_indices):
        pos = pos_base.detach().clone().requires_grad_(True)
        energy = model({"positions": pos})["energy"] + model.trainable_shift
        forces = -torch.autograd.grad(energy, pos, create_graph=True)[0]
        return torch.mean((forces - target_forces) ** 2)

    cfg_exact = oll.VRPCAConfig(batch_size=1, epochs=5, seed=0, hvp_mode="autograd")
    cfg_fd = oll.VRPCAConfig(
        batch_size=1, epochs=5, seed=0, hvp_mode="fd_grad", fd_eps=1e-6
    )
    orig_params = [p.detach().clone() for p in model.parameters()]
    with torch.no_grad():
        res_exact = oll.top_hessian_eigenpair_vrpca(
            net=model, loss_callable=force_loss_callable, dataset_size=1, config=cfg_exact
        )
        res_fd = oll.top_hessian_eigenpair_vrpca(
            net=model, loss_callable=force_loss_callable, dataset_size=1, config=cfg_fd
        )
    assert res_exact.eigenvalue == pytest.approx(res_fd.eigenvalue, rel=1e-3, abs=1e-4)
    for p, saved in zip(model.parameters(), orig_params):
        assert torch.allclose(p, saved)
    assert not model.atomic_shift.requires_grad

    # Classical small_hessian and hessian_vector_product on float64 model with frozen param
    loss_f = force_loss_callable(None)
    dense_h = small_hessian(model, loss_f)
    assert dense_h.dtype == np.float64
    assert not model.atomic_shift.requires_grad

    loss_f2 = force_loss_callable(None)
    hvp_op = hessian_vector_product(model, loss_f2)
    probe = np.ones(dense_h.shape[0], dtype=np.float64)
    assert np.allclose(hvp_op.matvec(probe), dense_h @ probe, atol=1e-6)

    # LossGradient must preserve requires_grad flags & LossPerturbations must scale by alpha
    mlp = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 1))
    x = torch.randn(6, 3)
    y = torch.randn(6, 1)
    mse = nn.MSELoss()
    _ = oll.point(mlp, LossGradient(mse, x, y))
    assert all(p.requires_grad for p in mlp.parameters())

    orig_mlp = [p.detach().clone() for p in mlp.parameters()]
    torch.manual_seed(0)
    pert_small = oll.point(mlp, LossPerturbations(mse, x, y, n_directions=4, alpha=1e-3))
    torch.manual_seed(0)
    pert_large = oll.point(mlp, LossPerturbations(mse, x, y, n_directions=4, alpha=1e-1))
    assert np.max(np.abs(pert_large)) > np.max(np.abs(pert_small)) * 5.0
    for p, saved in zip(mlp.parameters(), orig_mlp):
        assert torch.equal(p, saved)
