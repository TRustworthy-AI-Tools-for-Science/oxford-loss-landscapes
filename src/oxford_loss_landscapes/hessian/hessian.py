import time
import warnings

import torch
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

# Hessian Eigenvectors and Values
from numpy import linalg as LA

# Threshold for large matrix size
SMALL_MATRIX_SIZE = int(1e3)
MEDIUM_MATRIX_SIZE = int(1e6)
LARGE_MATRIX_SIZE = int(1e10)

################################################################################
#                              Supporting Functions
################################################################################

def get_eigenstuff(hessian, num_eigs_returned=2, method='numpy'):
    """Calculate eigenvalues and eigenvectors of a Hessian matrix or LinearOperator."""
    from scipy.sparse.linalg import LinearOperator as SciPyLinearOperator

    # LinearOperator (e.g. from hessian_vector_product for medium models) must use eigsh
    if isinstance(hessian, SciPyLinearOperator):
        n = hessian.shape[0]
        if n < num_eigs_returned:
            raise ValueError(f"Operator size {n} too small for {num_eigs_returned} eigenvalues.")
        eigenvalues, eigenvectors = eigsh(hessian, k=num_eigs_returned, which='LM', tol=1e-2)
        eigenvalues = [float(ev) for ev in eigenvalues]
        eigenvectors = [eigenvectors[:, i] for i in range(eigenvectors.shape[1])]
        return eigenvalues, eigenvectors

    if method == 'numpy':
        # Warn for very large matrices
        if hessian.shape[0] > LARGE_MATRIX_SIZE:
            warnings.warn("Matrix is very large for numpy.linalg.eig. Consider using method='scipy'.")

        # Use eigh for symmetric Hessians to guarantee real eigenvalues/vectors
        eigenvalues, eigenvectors = LA.eigh(hessian)

        # Sort eigenvalues in ascending order and select the largest num_eigs_returned
        sorted_indices = np.argsort(eigenvalues)[-num_eigs_returned:]
        eigenvalues = [float(eigenvalues[i]) for i in sorted_indices]
        eigenvectors = [eigenvectors[:, i].flatten() for i in sorted_indices]

    elif method == 'scipy':
        # Validation checks
        if hessian.shape[0] != hessian.shape[1]:
            raise ValueError("Matrix must be square for scipy eigsh decomposition.")
        if hessian.shape[0] < num_eigs_returned:
            raise ValueError(f"Matrix size {hessian.shape[0]} too small for {num_eigs_returned} eigenvalues.")

        # Warn for edge cases
        if hessian.shape[0] < 10 * num_eigs_returned:
            warnings.warn(f"Matrix size {hessian.shape[0]} small relative to requested eigenvalues. "
                         "Consider method='numpy' for better accuracy.")

        eigenvalues, eigenvectors = eigsh(hessian, k=num_eigs_returned, which='LM', tol=1e-2)
        eigenvalues = [float(ev) for ev in eigenvalues]
        eigenvectors = [eigenvectors[:, i] for i in range(eigenvectors.shape[1])]

    elif method == 'vrpca':
        raise NotImplementedError("VR-PCA method not implemented in this function.")

    return eigenvalues, eigenvectors

def _select_hessian_params(net, all_params: bool = True):
    """Return differentiable floating-point parameters without mutating frozen flags."""
    params = [
        p
        for p in net.parameters()
        if p.requires_grad and p.is_floating_point() and (all_params or p.ndim > 1)
    ]
    if not params:
        raise ValueError("Model has no differentiable floating-point parameters for Hessian computation.")
    return params


def _numpy_dtype_for_params(params):
    """Use float64 numpy vectors if any parameter is float64, else float32."""
    if any(p.dtype == torch.float64 for p in params):
        return np.float64
    return np.float32


def get_hessian(model, loss, method='numpy'):
    """Return the Hessian for a given model and pre-computed loss.

    For small models (< 1 000 params) returns a dense numpy ndarray.
    For medium models (< 1e10 params) returns a scipy LinearOperator that
    supports Hessian-vector products; pass it directly to ``get_eigenstuff``
    or ``scipy.sparse.linalg.eigsh``.
    """

    params = _select_hessian_params(model, all_params=True)
    n_params = sum(p.numel() for p in params)
    if n_params < SMALL_MATRIX_SIZE:
        return small_hessian(model, loss, method=method)
    elif n_params < LARGE_MATRIX_SIZE:
        if isinstance(loss, (int, float)):
            loss = torch.tensor(loss, dtype=params[0].dtype, device=params[0].device, requires_grad=True)
        elif not loss.requires_grad:
            loss.requires_grad_(True)
        return hessian_vector_product(net=model, loss=loss, use_cuda=False, all_params=True)
    else:
        raise NotImplementedError("Hessian computation for very large models not implemented.")
    

def small_hessian(model, loss, num_eigs_returned=2, method='numpy'):
    """Calculate the dense Hessian matrix over differentiable floating-point parameters."""

    params = _select_hessian_params(model, all_params=True)
    n_params = sum(p.numel() for p in params)
    if n_params > 1000:
        raise RuntimeError(
            f"Model has {n_params} differentiable parameters. "
            "Direct Hessian computation is not recommended for large models. "
            "Use Hessian-vector product methods for efficiency."
        )

    model.zero_grad()

    with torch.enable_grad():
        if isinstance(loss, (int, float)):
            loss = torch.tensor(loss, dtype=params[0].dtype, device=params[0].device, requires_grad=True)
        elif not loss.requires_grad:
            loss.requires_grad_(True)

        grads = torch.autograd.grad(
            loss, params, create_graph=True, retain_graph=True, allow_unused=True
        )

        processed_grads = []
        for param, g in zip(params, grads):
            if g is not None:
                processed_grads.append(g.contiguous().view(-1))
            else:
                processed_grads.append(
                    torch.zeros(param.numel(), dtype=param.dtype, device=param.device)
                )

        grads_flat = torch.cat(processed_grads)

        hessian_rows = []
        for g_idx in grads_flat:
            if not g_idx.requires_grad:
                zero_row = torch.zeros(n_params, dtype=g_idx.dtype, device=g_idx.device)
                hessian_rows.append(zero_row)
                continue

            grad2 = torch.autograd.grad(
                g_idx, params, retain_graph=True, allow_unused=True
            )
            processed_grad2 = []
            for param, g2 in zip(params, grad2):
                if g2 is not None:
                    processed_grad2.append(g2.contiguous().view(-1))
                else:
                    processed_grad2.append(
                        torch.zeros(param.numel(), dtype=param.dtype, device=param.device)
                    )
            hessian_rows.append(torch.cat(processed_grad2))

    hessian = torch.stack(hessian_rows).detach().cpu().numpy()
    return hessian


def hessian_vector_product(net, loss, use_cuda=False, all_params=True, rank=0, verbose=False):
    """Implementation of the hessian vector product algorithm using a pre-computed loss."""
    hess_vec_prod, params, N = create_hessian_vector_product_from_loss(
        net, loss, use_cuda, all_params
    )
    hess_vec_prod.count = 0

    if verbose and rank == 0:
        print("Rank %d: computing max eigenvalue" % rank)

    def hvp_wrapper(vec):
        return hess_vec_prod(vec, verbose=verbose, rank=rank)

    np_dtype = _numpy_dtype_for_params(params)
    hessian = LinearOperator((N, N), matvec=hvp_wrapper, dtype=np_dtype)
    return hessian


def npvec_to_tensorlist(vec, params):
    """Convert numpy vector to list of tensors matching each parameter's shape, dtype, and device."""
    loc = 0
    result = []
    for p in params:
        numel = p.data.numel()
        chunk = np.asarray(vec[loc:loc + numel])
        result.append(
            torch.as_tensor(chunk, dtype=p.dtype, device=p.device).view(p.data.shape)
        )
        loc += numel
    assert loc == vec.size, f'Vector length {vec.size} does not match total parameter count {loc}'
    return result


def gradtensor_to_npvec(net_or_params, all_params=True):
    """Extract gradients from parameters and return a concatenated numpy vector."""
    if isinstance(net_or_params, (list, tuple)):
        params = list(net_or_params)
    else:
        params = _select_hessian_params(net_or_params, all_params=all_params)

    np_dtype = _numpy_dtype_for_params(params)
    grad_parts = []
    for p in params:
        if p.grad is None:
            grad_parts.append(np.zeros(p.data.numel(), dtype=np_dtype))
        else:
            grad_parts.append(p.grad.detach().cpu().numpy().astype(np_dtype, copy=False).ravel())

    return np.concatenate(grad_parts)


################################################################################
#                  For computing Hessian-vector products
################################################################################
def eval_hess_vec_prod_from_loss(vec, params, net, loss, use_cuda=False):
    """
    Evaluate Hessian-vector product using a pre-computed loss and store result in network gradients.
    """
    if use_cuda:
        net.cuda()
        vec = [v.cuda() for v in vec]

    was_training = net.training
    net.eval()
    net.zero_grad()
    try:
        with torch.enable_grad():
            if not loss.requires_grad:
                raise ValueError("Loss tensor must have requires_grad=True for Hessian computation")

            grad_f = torch.autograd.grad(
                loss, inputs=params, create_graph=True, allow_unused=True, retain_graph=True
            )

            inner_products = []
            for i, v in enumerate(vec):
                if grad_f[i] is not None and grad_f[i].requires_grad and v is not None:
                    inner_products.append((grad_f[i] * v.to(device=grad_f[i].device, dtype=grad_f[i].dtype)).sum())

            if inner_products:
                prod = sum(inner_products)
                prod.backward(retain_graph=True)
    finally:
        if was_training:
            net.train(True)


def eval_hess_vec_prod(vec, params, net, loss_func, inputs, outputs, use_cuda=False):
    """
    Evaluate Hessian-vector product and store result in network gradients.
    """
    if use_cuda:
        net.cuda()
        vec = [v.cuda() for v in vec]

    device = params[0].device
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    if outputs is not None and hasattr(outputs, "to"):
        outputs = outputs.to(device)

    was_training = net.training
    net.eval()
    net.zero_grad()
    try:
        with torch.enable_grad():
            pred_outputs = net(inputs)
            loss = loss_func(pred_outputs, outputs) if outputs is not None else loss_func(pred_outputs)
            grad_f = torch.autograd.grad(
                loss, inputs=params, create_graph=True, allow_unused=True
            )

            inner_products = []
            for i, v in enumerate(vec):
                if grad_f[i] is not None and grad_f[i].requires_grad and v is not None:
                    inner_products.append((grad_f[i] * v.to(device=grad_f[i].device, dtype=grad_f[i].dtype)).sum())

            if inner_products:
                prod = sum(inner_products)
                prod.backward()
    finally:
        if was_training:
            net.train(True)

################################################################################
#                  For computing Eigenvalues of Hessian
################################################################################
def create_hessian_vector_product_from_loss(net, loss, use_cuda=False, all_params=True):
    """
    Create a Hessian-vector product function for a given model using a pre-computed loss.
    """
    params = _select_hessian_params(net, all_params=all_params)
    N = sum(p.numel() for p in params)

    def hess_vec_prod(vec, verbose=False, rank=0):
        if not hasattr(hess_vec_prod, 'count'):
            hess_vec_prod.count = 0

        hess_vec_prod.count += 1
        vec_tensors = npvec_to_tensorlist(vec, params)
        start_time = time.time()
        eval_hess_vec_prod_from_loss(vec_tensors, params, net, loss, use_cuda)
        prod_time = time.time() - start_time
        if verbose and rank == 0:
            print("Iter: %d  time: %f" % (hess_vec_prod.count, prod_time))
        return gradtensor_to_npvec(params, all_params)

    return hess_vec_prod, params, N


def create_hessian_vector_product(net, inputs, outputs, criterion, use_cuda=False, all_params=True):
    """
    Create a Hessian-vector product function for a given model and data.
    """
    params = _select_hessian_params(net, all_params=all_params)
    N = sum(p.numel() for p in params)

    def hess_vec_prod(vec, verbose=False, rank=0):
        if not hasattr(hess_vec_prod, 'count'):
            hess_vec_prod.count = 0

        hess_vec_prod.count += 1
        vec_tensors = npvec_to_tensorlist(vec, params)
        start_time = time.time()
        eval_hess_vec_prod(vec_tensors, params, net, criterion, inputs, outputs, use_cuda)
        prod_time = time.time() - start_time
        if verbose and rank == 0:
            print("Iter: %d  time: %f" % (hess_vec_prod.count, prod_time))
        return gradtensor_to_npvec(params, all_params)

    return hess_vec_prod, params, N


def min_max_hessian_eigs(net, inputs, outputs, criterion, rank=0, use_cuda=False, verbose=False, all_params=True,
                         backend=None, vrpca_config=None, compute_min=False):
    """
        Compute the largest and the smallest eigenvalues of the Hessian marix.

        Args:
            net: the trained model.
            inputs: nn inputs.
            outputs: desired nn outputs.
            criterion: loss function.
            rank: rank of the working node.
            use_cuda: use GPU
            verbose: print more information
            all_params: use all nn parameters

        Returns:
            maxeig: max eigenvalue
            mineig: min eigenvalue
            maxeigvec: max eigenvector
            mineigvec: min eigenvector
            hess_vec_prod.count: number of iterations for calculating max and min eigenvalues
    """
    if backend == "vrpca":
        from oxford_loss_landscapes.hessian.vrpca.api import min_max_hessian_eigs_vrpca
        return min_max_hessian_eigs_vrpca(
            net=net,
            inputs=inputs,
            targets=outputs,
            criterion=criterion,
            all_params=all_params,
            use_cuda=use_cuda,
            config=vrpca_config,
            compute_min=compute_min,
        )

    # Create the Hessian-vector product function
    hess_vec_prod, params, N = create_hessian_vector_product(
        net, inputs, outputs, criterion, use_cuda, all_params
    )
    
    # Reset counter
    hess_vec_prod.count = 0
    
    if verbose and rank == 0: 
        print("Rank %d: computing max eigenvalue" % rank)

    # Create a wrapper that matches the expected interface for eigsh
    def hvp_wrapper(vec):
        return hess_vec_prod(vec, verbose=verbose, rank=rank)
    
    A = LinearOperator((N, N), matvec=hvp_wrapper)
    
    # Use scipy's eigsh directly instead of get_eigenstuff for LinearOperator
    eigvals, eigvecs = eigsh(A, k=1, which='LM', tol=1e-2)

    maxeig = eigvals[0]
    maxeigvec = eigvecs[:, 0]
    if verbose and rank == 0: print('max eigenvalue = %f' % maxeig)

    # If the largest eigenvalue is positive, shift matrix so that any negative eigenvalue is now the largest
    # We assume the smallest eigenvalue is zero or less, and so this shift is more than what we need
    shift = maxeig*1.0
    def shifted_hess_vec_prod(vec):
        return hvp_wrapper(vec) - shift*vec

    if verbose and rank == 0: print("Rank %d: Computing shifted eigenvalue" % rank)

    A = LinearOperator((N, N), matvec=shifted_hess_vec_prod)

    # Use scipy's eigsh directly instead of get_eigenstuff for LinearOperator
    eigvals, eigvecs = eigsh(A, k=1, which='LM', tol=1e-2)

    eigvals = eigvals + shift
    mineig = eigvals[0]
    mineigvec = eigvecs[:, 0]
    if verbose and rank == 0: print('min eigenvalue = ' + str(mineig))

    if maxeig <= 0 and mineig > 0:
        maxeig, mineig = mineig, maxeig
        maxeigvec, mineigvec = mineigvec, maxeigvec

    return maxeig, mineig, maxeigvec, mineigvec, hess_vec_prod.count

def get_hessian_eigenstuff(model, loss, num_eigs_returned=2, method='numpy', landscape_type='minimum'):
    """Get the eigenvalues and eigenvectors of the Hessian matrix."""
    
    if landscape_type not in ['minimum', 'saddle']:
        raise ValueError("landscape_type must be 'minimum' or 'saddle'")
    
    if landscape_type == 'minimum':
        try:
            hessian = get_hessian(model, loss, method=method)
            eigenvalues, eigenvectors = get_eigenstuff(hessian, num_eigs_returned, method)
            return eigenvalues, eigenvectors
        except Exception as e:
            warnings.warn(f"Failed to compute Hessian eigenvalues: {e}. Returning None values.")
            return None, None
    elif landscape_type == 'saddle':
        raise NotImplementedError("Min-max eigenvalue computation not implemented.")
    else:
        return None, None