Hessian Module
==============

Tools for computing Hessian matrices and eigenvalues of neural network
loss functions. Import from ``oxford_loss_landscapes.hessian``.

.. code-block:: python

   from oxford_loss_landscapes.hessian import (
       min_max_hessian_eigs,
       get_hessian_eigenstuff,
       hessian_trace,
       get_hessian,
   )

Eigenvalue Solvers
------------------

.. autofunction:: oxford_loss_landscapes.hessian.min_max_hessian_eigs
.. autofunction:: oxford_loss_landscapes.hessian.get_hessian_eigenstuff
.. autofunction:: oxford_loss_landscapes.hessian.hessian_trace

Hessian Construction
--------------------

.. autofunction:: oxford_loss_landscapes.hessian.get_hessian
.. autofunction:: oxford_loss_landscapes.hessian.small_hessian
.. autofunction:: oxford_loss_landscapes.hessian.hessian_vector_product
.. autofunction:: oxford_loss_landscapes.hessian.create_hessian_vector_product
.. autofunction:: oxford_loss_landscapes.hessian.create_hessian_vector_product_from_loss
.. autofunction:: oxford_loss_landscapes.hessian.get_eigenstuff

Conversion Utilities
--------------------

.. autofunction:: oxford_loss_landscapes.hessian.npvec_to_tensorlist
.. autofunction:: oxford_loss_landscapes.hessian.gradtensor_to_npvec
.. autofunction:: oxford_loss_landscapes.hessian.eval_hess_vec_prod
