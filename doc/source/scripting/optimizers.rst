Optimizers (``recommendation.optimizers``)
******************************************

.. automodule:: orangecontrib.recommendation.optimizers

The classes presented in this section are optimizers to modify the SGD updates
during the training of a model.

The update functions control the learning rate during the SGD optimization

.. autosummary::
    :nosignatures:

    SGD
    Momentum
    NesterovMomentum
    AdaGrad
    RMSProp
    AdaDelta
    Adam


Stochastic Gradient Descent
============================

This is the optimizer by default in all models.

.. autoclass:: SGD
   :members:
   :special-members: __init__


Momentum
========

.. autoclass:: Momentum
   :members:
   :special-members: __init__


Nesterov's Accelerated Gradient
===============================

.. autoclass:: NesterovMomentum
   :members:
   :special-members: __init__


AdaGradient
===========

.. autoclass:: AdaGrad
   :members:
   :special-members: __init__


RMSProp
=======

.. autoclass:: RMSProp
   :members:
   :special-members: __init__


AdaDelta
========

.. autoclass:: AdaDelta
   :members:
   :special-members: __init__


Adam
=====

.. autoclass:: Adam
   :members:
   :special-members: __init__
