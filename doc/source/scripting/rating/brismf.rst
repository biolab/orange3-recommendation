BRISMF
======

BRISMF (Biased Regularized Incremental Simultaneous Matrix Factorization) is
factorization-based algorithm for large scale recommendation systems.

The basic idea is to factorize a very sparse matrix into two low-rank matrices
which represents user and item factors. This can be done by using an iterative
approach to minimize the loss function.

User's predictions are defined as follows:

.. math::
	\hat { r }_{ ui }=\mu +b_{ u }+b_{ i }+{ q }_{ i }^{ T }{ p }_{ u }


We learn the values of involved parameters by minimizing the regularized squared error function associated with:

.. math::
	\min _{ p*,q*,b* }{ \sum _{ (u,i\in k) }^{  }{ { ({ r }_{ ui }-\mu -b_{ u }-b_{ i }-{ q }_{ i }^{ T }{ p }_{ u }) }^{ 2 }+\lambda ({ { { b }_{ u } }^{ 2 }+{ { b }_{ i } }^{ 2 }+\left\| { p }_{ u } \right\|  }^{ 2 }+{ \left\| q_{ i } \right\|  }^{ 2 }) }  }


Example
-------

.. code-block:: python
   :linenos:

   import Orange
   from orangecontrib.recommendation import BRISMFLearner

   # Load data and train the model
   data = Orange.data.Table('movielens100k.tab')
   learner = BRISMFLearner(num_factors=15, num_iter=25, learning_rate=0.07, lmbda=0.1)
   recommender = learner(data)

   # Make predictions
   prediction = recommender(data[:3])
   print(prediction)
   >>>
   [ 3.79505151  3.75096513  1.293013 ]

.. autoclass:: BRISMFLearner
   :members:
   :special-members: __init__