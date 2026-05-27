Core Functions
==============

The top-level ``oxford_loss_landscapes`` namespace exposes all primary
loss-landscape functions.

.. code-block:: python

   import oxford_loss_landscapes as oll

Loss Landscape Evaluation
--------------------------

.. autofunction:: oxford_loss_landscapes.point
.. autofunction:: oxford_loss_landscapes.linear_interpolation
.. autofunction:: oxford_loss_landscapes.random_line
.. autofunction:: oxford_loss_landscapes.planar_interpolation
.. autofunction:: oxford_loss_landscapes.random_plane
.. autofunction:: oxford_loss_landscapes.hessian_plane

Utilities
---------

.. autofunction:: oxford_loss_landscapes.rand_u_like
.. autofunction:: oxford_loss_landscapes.download_zenodo_model
.. autofunction:: oxford_loss_landscapes.download_zenodo_zip
.. autofunction:: oxford_loss_landscapes.extract_zip
