Model Interface
===============

Wrappers that give loss-landscape functions a uniform view of a PyTorch
model's parameters. Import from ``oxford_loss_landscapes`` or from
``oxford_loss_landscapes.model_interface.model_wrapper``.

Model Wrappers
--------------

.. autoclass:: oxford_loss_landscapes.ModelWrapper
   :members:
   :show-inheritance:

.. autoclass:: oxford_loss_landscapes.SimpleModelWrapper
   :members:
   :show-inheritance:

.. autoclass:: oxford_loss_landscapes.GeneralModelWrapper
   :members:
   :show-inheritance:

.. autoclass:: oxford_loss_landscapes.TransformerModelWrapper
   :members:
   :show-inheritance:

Model Parameters
----------------

.. autoclass:: oxford_loss_landscapes.model_interface.model_parameters.ModelParameters
   :members:

.. autofunction:: oxford_loss_landscapes.model_interface.model_parameters.rand_n_like
.. autofunction:: oxford_loss_landscapes.model_interface.model_parameters.orthogonal_to
