"""Regression tests for bugs fixed in May 2026 code review."""

import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))


def _data(n=20, d=4):
    torch.manual_seed(1)
    X = torch.randn(n, d)
    y = torch.randn(n, 1)
    return X, y


# ---------------------------------------------------------------------------
# 1. ModelWrapper.parameters() / named_parameters() itertools.chain fix
# ---------------------------------------------------------------------------

class TestModelWrapperIteration:
    def test_parameters_yields_tensors(self):
        from oxford_loss_landscapes.model_interface.model_wrapper import SimpleModelWrapper
        model = _small_model()
        wrapper = SimpleModelWrapper(model)
        params = list(wrapper.parameters())
        assert len(params) > 0, "parameters() returned nothing"
        for p in params:
            assert isinstance(p, torch.Tensor), f"Expected Tensor, got {type(p)}"

    def test_named_parameters_yields_name_tensor_pairs(self):
        from oxford_loss_landscapes.model_interface.model_wrapper import SimpleModelWrapper
        model = _small_model()
        wrapper = SimpleModelWrapper(model)
        named = list(wrapper.named_parameters())
        assert len(named) > 0
        for name, param in named:
            assert isinstance(name, str)
            assert isinstance(param, torch.Tensor)

    def test_parameters_count_matches_model(self):
        from oxford_loss_landscapes.model_interface.model_wrapper import SimpleModelWrapper
        model = _small_model()
        wrapper = SimpleModelWrapper(model)
        assert len(list(wrapper.parameters())) == len(list(model.parameters()))


# ---------------------------------------------------------------------------
# 2. ModelParameters validation, as_numpy, rand_n_like device
# ---------------------------------------------------------------------------

class TestModelParameters:
    def test_init_rejects_non_list(self):
        from oxford_loss_landscapes.model_interface.model_parameters import ModelParameters
        with pytest.raises(AttributeError):
            ModelParameters("not a list")

    def test_init_rejects_list_of_non_tensors(self):
        from oxford_loss_landscapes.model_interface.model_parameters import ModelParameters
        with pytest.raises(AttributeError):
            ModelParameters([1, 2, 3])

    def test_init_accepts_list_of_tensors(self):
        from oxford_loss_landscapes.model_interface.model_parameters import ModelParameters
        params = [torch.randn(3, 4), torch.randn(4)]
        mp = ModelParameters(params)
        assert len(mp) == 2

    def test_as_numpy_works_with_requires_grad(self):
        from oxford_loss_landscapes.model_interface.model_parameters import ModelParameters
        t = torch.randn(3, 4, requires_grad=True)
        mp = ModelParameters([t])
        arr = mp.as_numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (12,)

    def test_rand_n_like_matches_device(self):
        from oxford_loss_landscapes.model_interface.model_parameters import ModelParameters, rand_n_like
        # CPU test (GPU not always available)
        params = [torch.randn(3, 4), torch.randn(4)]
        mp = ModelParameters(params)
        rnd = rand_n_like(mp)
        for ref, r in zip(params, rnd._get_parameters()):
            assert r.device == ref.device
            assert r.shape == ref.shape
            assert r.dtype == ref.dtype


# ---------------------------------------------------------------------------
# 3. LossGradient now functional
# ---------------------------------------------------------------------------

class TestLossGradient:
    def test_returns_numpy_array_of_correct_shape(self):
        from oxford_loss_landscapes.metrics.sl_metrics import LossGradient
        from oxford_loss_landscapes.model_interface.model_wrapper import SimpleModelWrapper
        model = _small_model()
        X, y = _data()
        wrapper = SimpleModelWrapper(model)
        metric = LossGradient(nn.MSELoss(), X, y)
        grad = metric(wrapper)
        assert isinstance(grad, np.ndarray)
        n_params = sum(p.numel() for p in model.parameters())
        assert grad.shape == (n_params,)

    def test_gradient_is_nonzero(self):
        from oxford_loss_landscapes.metrics.sl_metrics import LossGradient
        from oxford_loss_landscapes.model_interface.model_wrapper import SimpleModelWrapper
        model = _small_model()
        X, y = _data()
        wrapper = SimpleModelWrapper(model)
        metric = LossGradient(nn.MSELoss(), X, y)
        grad = metric(wrapper)
        assert np.any(grad != 0), "Gradient is all zeros — computation likely failed"

    def test_gradient_matches_manual_computation(self):
        from oxford_loss_landscapes.metrics.sl_metrics import LossGradient
        from oxford_loss_landscapes.model_interface.model_wrapper import SimpleModelWrapper
        model = _small_model()
        X, y = _data()

        # Manual gradient
        model.zero_grad()
        loss = nn.MSELoss()(model(X), y)
        loss.backward()
        manual_grad = np.concatenate([p.grad.numpy().flatten() for p in model.parameters()])

        wrapper = SimpleModelWrapper(model)
        metric = LossGradient(nn.MSELoss(), X, y)
        computed_grad = metric(wrapper)

        assert np.allclose(computed_grad, manual_grad, rtol=1e-4)

    def test_requires_grad_restored_after_call(self):
        from oxford_loss_landscapes.metrics.sl_metrics import LossGradient
        from oxford_loss_landscapes.model_interface.model_wrapper import wrap_model
        model = _small_model()
        X, y = _data()
        wrapper = wrap_model(model)  # wrap_model sets requires_grad=False
        metric = LossGradient(nn.MSELoss(), X, y)
        metric(wrapper)
        for p in model.parameters():
            assert not p.requires_grad, "requires_grad was not restored to False after LossGradient call"


# ---------------------------------------------------------------------------
# 4. TransformerModelWrapper.get_module_parameters() no longer drops embeddings
# ---------------------------------------------------------------------------

class TestTransformerModelWrapperParams:
    def test_param_count_matches_model(self):
        from oxford_loss_landscapes.model_interface.model_wrapper import TransformerModelWrapper

        class FakeTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(10, 4)
                self.linear = nn.Linear(4, 1)

            def forward(self, x):
                return self.linear(self.embedding(x))

        model = FakeTransformer()
        wrapper = TransformerModelWrapper(model)
        mp = wrapper.get_module_parameters()
        assert len(mp) == len(list(model.parameters())), (
            "get_module_parameters() dropped parameters (embedding filter still active?)"
        )


# ---------------------------------------------------------------------------
# 5. get_hessian() / get_eigenstuff() handle LinearOperator for medium models
# ---------------------------------------------------------------------------

class TestHessianLinearOperator:
    def test_get_eigenstuff_accepts_linear_operator(self):
        from scipy.sparse.linalg import LinearOperator
        from oxford_loss_landscapes.hessian.hessian import get_eigenstuff

        n = 10
        A = np.diag(np.arange(1.0, n + 1))
        op = LinearOperator((n, n), matvec=lambda v: A @ v)

        eigenvalues, eigenvectors = get_eigenstuff(op, num_eigs_returned=2)
        assert len(eigenvalues) == 2
        # Top-2 should be n and n-1
        assert max(eigenvalues) == pytest.approx(float(n), rel=1e-2)

    def test_get_hessian_medium_model_returns_something_usable(self):
        from oxford_loss_landscapes.hessian.hessian import get_hessian, get_eigenstuff
        from scipy.sparse.linalg import LinearOperator

        # Model with just over SMALL_MATRIX_SIZE (1000) params: 25*25+25 = 650 per layer × 2 → 1326 total
        model = nn.Sequential(nn.Linear(25, 25), nn.ReLU(), nn.Linear(25, 25), nn.ReLU(), nn.Linear(25, 1))
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 1000  # must be in "medium" branch

        X = torch.randn(10, 25)
        y = torch.randn(10, 1)
        loss = nn.MSELoss()(model(X), y)

        H = get_hessian(model, loss)
        # For medium models, get_hessian returns a LinearOperator
        assert isinstance(H, LinearOperator)

        # get_eigenstuff must handle it
        eigenvalues, eigenvectors = get_eigenstuff(H, num_eigs_returned=2)
        assert len(eigenvalues) == 2
        assert len(eigenvectors) == 2


# ---------------------------------------------------------------------------
# 6. _evaluate_plane grid alignment (centre = original params)
# ---------------------------------------------------------------------------

class TestEvaluatePlaneAlignment:
    def test_centre_of_grid_is_original_model_loss(self):
        from oxford_loss_landscapes.model_interface.model_wrapper import SimpleModelWrapper
        from oxford_loss_landscapes.metrics.sl_metrics import Loss
        from oxford_loss_landscapes import random_plane

        torch.manual_seed(42)
        model = _small_model()
        X, y = _data()
        criterion = nn.MSELoss()

        # Record the loss at the original parameters
        original_loss = criterion(model(X), y).item()

        torch.manual_seed(7)  # seed the direction sampling for determinism
        wrapper = SimpleModelWrapper(model)
        metric = Loss(criterion, X, y)
        steps = 11  # odd so centre index is unambiguous
        plane = random_plane(wrapper, metric, distance=0.5, steps=steps)

        assert plane.shape == (steps, steps)
        centre = plane[steps // 2, steps // 2]
        corner = plane[0, 0]

        # Centre should be close to the original loss (float32 accumulation ~1e-3 tolerance)
        assert abs(centre - original_loss) < 1e-2, (
            f"Centre loss {centre:.6f} is too far from original loss {original_loss:.6f} — grid misaligned"
        )
        # Corners should be meaningfully different from centre (grid must actually perturb the model)
        assert abs(corner - centre) > 1e-3, (
            f"Corner loss {corner:.6f} ≈ centre loss {centre:.6f} — perturbation may be too small or grid is flat"
        )


# ---------------------------------------------------------------------------
# 7. hessian_plane produces correct shape and is importable at top level
# ---------------------------------------------------------------------------

class TestHessianPlane:
    def test_returns_correct_shape(self):
        import oxford_loss_landscapes as oll

        model = _small_model()
        X, y = _data()
        criterion = nn.MSELoss()

        from oxford_loss_landscapes.model_interface.model_wrapper import SimpleModelWrapper
        from oxford_loss_landscapes.metrics.sl_metrics import Loss
        from oxford_loss_landscapes.hessian.vrpca import VRPCAConfig

        wrapper = SimpleModelWrapper(model)
        metric = Loss(criterion, X, y)
        cfg = VRPCAConfig(batch_size=20, epochs=5)
        steps = 7
        plane = oll.hessian_plane(wrapper, metric, criterion, X, y, steps=steps, vrpca_config=cfg)

        assert isinstance(plane, np.ndarray)
        assert plane.shape == (steps, steps)

    def test_importable_from_top_level(self):
        import oxford_loss_landscapes as oll
        assert hasattr(oll, "hessian_plane")
        assert callable(oll.hessian_plane)


# ---------------------------------------------------------------------------
# 8. VR-PCA types accessible at top level
# ---------------------------------------------------------------------------

class TestVRPCATopLevel:
    def test_vrpca_config_importable(self):
        import oxford_loss_landscapes as oll
        assert hasattr(oll, "VRPCAConfig")
        cfg = oll.VRPCAConfig(batch_size=32, epochs=10)
        assert cfg.batch_size == 32

    def test_vrpca_result_importable(self):
        import oxford_loss_landscapes as oll
        assert hasattr(oll, "VRPCAResult")

    def test_top_hessian_eigenpair_vrpca_importable(self):
        import oxford_loss_landscapes as oll
        assert hasattr(oll, "top_hessian_eigenpair_vrpca")
        assert callable(oll.top_hessian_eigenpair_vrpca)

    def test_min_max_hessian_eigs_vrpca_importable(self):
        import oxford_loss_landscapes as oll
        assert hasattr(oll, "min_max_hessian_eigs_vrpca")
        assert callable(oll.min_max_hessian_eigs_vrpca)

    def test_top_hessian_eigenpair_vrpca_runs(self):
        import oxford_loss_landscapes as oll

        model = _small_model()
        X, y = _data()
        criterion = nn.MSELoss()

        cfg = oll.VRPCAConfig(batch_size=20, epochs=5)
        result = oll.top_hessian_eigenpair_vrpca(
            net=model, inputs=X, targets=y, criterion=criterion, config=cfg
        )

        assert isinstance(result, oll.VRPCAResult)
        assert isinstance(result.eigenvalue, float)
        n_params = sum(p.numel() for p in model.parameters())
        assert result.eigenvector.shape == (n_params,)
        assert result.eigenvector.norm().item() == pytest.approx(1.0, abs=1e-3)
