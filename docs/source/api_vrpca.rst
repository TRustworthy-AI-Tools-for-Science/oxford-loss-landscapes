VR-PCA Eigensolver
==================

Variance-Reduced PCA (VR-PCA) based eigenpair solvers for large-scale
Hessian analysis. These are exposed at the top-level package and under
``oxford_loss_landscapes.hessian.vrpca``.

.. code-block:: python

   import oxford_loss_landscapes as oll

   config = oll.VRPCAConfig(batch_size=64, epochs=20, tol=1e-4)
   result = oll.top_hessian_eigenpair_vrpca(
       net=model, inputs=inputs, targets=targets, criterion=criterion,
       config=config,
   )
   print(result.eigenvalue, result.converged)

Configuration
-------------

.. autoclass:: oxford_loss_landscapes.VRPCAConfig
   :members:
   :undoc-members:

Results
-------

.. autoclass:: oxford_loss_landscapes.VRPCAResult
   :members:

Solver Functions
----------------

.. autofunction:: oxford_loss_landscapes.top_hessian_eigenpair_vrpca
.. autofunction:: oxford_loss_landscapes.min_hessian_eigenpair_vrpca
.. autofunction:: oxford_loss_landscapes.min_max_hessian_eigs_vrpca
