Features
========

Core Loss Landscape Functions
------------------------------

All functions are available directly from ``import oxford_loss_landscapes as oll``.

``point(model, metric)``
   Evaluate the loss metric at the model's current parameter values.

``linear_interpolation(model_start, model_end, metric, distance, steps)``
   Compute loss along the straight line connecting two models in parameter
   space. Returns a 1-D NumPy array of length ``steps``.

``random_line(model, metric, distance, steps, normalization)``
   Compute loss along a single random direction from the current parameters.
   Supports ``'model'``, ``'layer'``, and ``'filter'`` normalization.

``planar_interpolation(model_start, model_end_1, model_end_2, metric, steps)``
   Compute loss over the 2-D plane spanned by three model checkpoints.

``random_plane(model, metric, distance, steps, normalization, export, use_ray)``
   Compute loss over a random 2-D plane centred on the current parameters.
   Pass ``export=True`` to write results to ``./results/`` for the
   interactive dashboard. Pass ``use_ray=True`` to evaluate rows in
   parallel via Ray (falls back to sequential if Ray is not installed).

``hessian_plane(model, metric, loss, steps, distance)``
   Compute loss over the plane spanned by the two leading Hessian
   eigenvectors — the directions of sharpest curvature.

Parallel Evaluation
-------------------

``random_plane`` and ``hessian_plane`` support parallel row evaluation
via `Ray <https://www.ray.io/>`_.

.. code-block:: python

   plane = oll.random_plane(model_wrapper, metric, distance=1.0, steps=25,
                            use_ray=True, num_workers=4)

If Ray is not installed the function silently falls back to sequential
evaluation. The global flag ``oxford_loss_landscapes.main.USE_PARALLEL``
can also enable parallelism by default.

Hessian Analysis
-----------------

``oxford_loss_landscapes.hessian`` provides tools to compute eigenvalues
and eigenvectors of the loss Hessian.

Classical solver
~~~~~~~~~~~~~~~~

.. code-block:: python

   from oxford_loss_landscapes.hessian import min_max_hessian_eigs

   max_eig, min_eig, max_vec, min_vec, iters = min_max_hessian_eigs(
       net=model, inputs=inputs, outputs=targets, criterion=criterion,
   )

VR-PCA solver
~~~~~~~~~~~~~

The Variance-Reduced PCA solver scales to large models and is exposed at
the top-level package:

.. code-block:: python

   import oxford_loss_landscapes as oll

   config = oll.VRPCAConfig(batch_size=64, epochs=20, tol=1e-4)
   result = oll.top_hessian_eigenpair_vrpca(
       net=model, inputs=inputs, targets=targets, criterion=criterion,
       config=config,
   )
   print(f"max eigenvalue ≈ {result.eigenvalue:.4f}, converged={result.converged}")

The ``backend`` dispatch parameter on ``min_max_hessian_eigs`` allows
drop-in replacement of the classical solver with VR-PCA:

.. code-block:: python

   max_eig, min_eig, _, _, cost = min_max_hessian_eigs(
       net=model, inputs=inputs, outputs=targets, criterion=criterion,
       backend="vrpca", vrpca_config=config, compute_min=True,
   )

Hessian trace estimation:

.. code-block:: python

   from oxford_loss_landscapes.hessian import hessian_trace

   trace = hessian_trace(model, criterion, inputs, targets,
                         num_random_vectors=10)

Model Interface
---------------

Four wrapper classes adapt any PyTorch model to the landscape API.

``ModelWrapper``
   Base class; wraps any ``torch.nn.Module``.

``SimpleModelWrapper``
   Lightweight wrapper for single-input / single-output models.

``GeneralModelWrapper``
   Handles models with multiple inputs or complex forward signatures.

``TransformerModelWrapper``
   Specialised wrapper for HuggingFace-style transformer models with
   ``input_ids`` / ``attention_mask`` inputs.

Interactive Dashboard
---------------------

Generate a results directory with ``export=True``, then launch the Dash
application for interactive 3-D exploration of the loss surface:

.. code-block:: bash

   python -m oxford_loss_landscapes.dashboard.gui_loss_dash

Requires the ``advanced`` optional dependencies
(``pip install -e ".[advanced]"``).

Model Downloading
-----------------

Download pretrained models from Zenodo directly:

.. code-block:: python

   from oxford_loss_landscapes import download_zenodo_model

   download_zenodo_model(record_id="12345678", output_path="./models/")
