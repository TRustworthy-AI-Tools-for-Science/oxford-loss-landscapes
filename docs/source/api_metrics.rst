Metrics
=======

Metric classes define what quantity is evaluated at each point in
parameter space (e.g. loss, gradient norm). Import from
``oxford_loss_landscapes.metrics``.

.. code-block:: python

   from oxford_loss_landscapes.metrics import Loss, LossGradient

Base Class
----------

.. autoclass:: oxford_loss_landscapes.metrics.metric.Metric
   :members:

Supervised Learning Metrics
----------------------------

.. automodule:: oxford_loss_landscapes.metrics.sl_metrics
   :members:
   :undoc-members:
   :show-inheritance:

Reinforcement Learning Metrics
-------------------------------

.. automodule:: oxford_loss_landscapes.metrics.rl_metrics
   :members:
   :undoc-members:
   :show-inheritance:
