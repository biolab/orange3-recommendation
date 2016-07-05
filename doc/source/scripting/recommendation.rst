##############################
Scripting (``recommendation``)
##############################

.. automodule:: orangecontrib.recommendation


.. index:: global_average
   pair: recommenders; global_average

==============
Global Average
==============

Global Average uses the average rating value of all ratings to make predictions.

.. math::
	r_{ ui } = \mu


Example
-------

.. code-block:: python
   :linenos:

    import Orange
    from orangecontrib.recommendation import GlobalAvgLearner

    # Load data and train the model
    data = Orange.data.Table('movielens100k.tab')
    learner = GlobalAvgLearner()
    recommender = learner(data)

    prediction = recommender(data[:3])
    print(prediction)
    >>>
    [ 3.52986  3.52986  3.52986]

.. autoclass:: GlobalAvgLearner
   :members:
   :special-members: __init__



.. index:: user_average
   pair: recommenders; user_average

============
User Average
============

User Average uses the average rating value of a user to make predictions.


.. math::
	r_{ ui } = \mu_{u}

Example
-------

.. code-block:: python
   :linenos:

   import Orange
   from orangecontrib.recommendation import UserAvgLearner

   # Load data and train the model
   data = Orange.data.Table('movielens100k.tab')
   learner = UserAvgLearner()
   recommender = learner(data)

   # Make predictions
   prediction = recommender(data[:3])
   print(prediction)
   >>>
   [ 3.61538462  3.41304348  3.3515625 ]

.. autoclass:: UserAvgLearner
   :members:
   :special-members: __init__



.. index:: item_average
   pair: recommenders; item_average

============
Item Average
============

Item Average uses the average rating value of an item to make predictions.

.. math::
	r_{ ui } = \mu_{i}

Example
-------

.. code-block:: python
   :linenos:

   import Orange
   from orangecontrib.recommendation import ItemAvgLearner

   # Load data and train the model
   data = Orange.data.Table('movielens100k.tab')
   learner = ItemAvgLearner()
   recommender = learner(data)

   # Make predictions
   prediction = recommender(data[:3])
   print(prediction)
   >>>
   [ 3.99145299  4.16161616  2.15384615]

.. autoclass:: ItemAvgLearner
   :members:
   :special-members: __init__



.. index:: user_item_baseline
   pair: recommenders; user_item_baseline

==================
User-Item Baseline
==================

User-Item Baseline takes the bias of users and items plus the global average to
make predictions.

.. math::
	r_{ ui } = \mu + b_{u} + b_{i}

Example
-------

.. code-block:: python
   :linenos:

    import Orange
    from orangecontrib.recommendation import UserItemBaselineLearner

    # Load data and train the model
    data = Orange.data.Table('movielens100k.tab')
    learner = UserItemBaselineLearner()
    recommender = learner(data)

    # Make predictions
    prediction = recommender(data[:3])
    print(prediction)
    >>>
    [ 4.07697761  4.04479964  1.97554865]


.. autoclass:: UserItemBaselineLearner
   :members:
   :special-members: __init__



.. index:: brismf
   pair: recommenders; brismf

======
BRISMF
======

BRISMF (Biased Regularized Incremental Simultaneous Matrix Factorization) is
factorization-based algorithm for large scale recommendation systems.

The basic idea is to factorize a very sparse matrix into two low-rank matrices
which represents user and item factors. This can be done by using an iterative
approach to minimize the loss function.

User's predictions are defined as follows:

.. math::
	\hat { r } _{ ui }=\mu +b_{ u }+b_{ i }+{ q }_{ i }^{ T }{ p }_{ u }


But in order to compute the two low-rank matrices, first these are randomly initialized and then optimized through a training loss like this:

.. math::
	\min _{ p*,q*,b* }{ \sum _{ (u,i\in k) }^{  }{ { ({ r }_{ ui }-\mu -b_{ u }-b_{ i }-{ q }_{ i }^{ T }{ p }_{ u }) }^{ 2 }+\lambda ({ \left\| { p }_{ u } \right\|  }^{ 2 }+{ \left\| q_{ i } \right\|  }^{ 2 }+{ { b }_{ u } }^{ 2 }+{ { b }_{ i } }^{ 2 }) }  }


Example
-------

.. code-block:: python
   :linenos:

   import Orange
   from orangecontrib.recommendation import BRISMFLearner

   # Load data and train the model
   data = Orange.data.Table('movielens100k.tab')
   learner = BRISMFLearner(K=15, steps=25, alpha=0.07, beta=0.1)
   recommender = learner(data)

   # Make predictions
   prediction = recommender(data[:3])
   print(prediction)
   >>>
   [ 3.79505151  3.75096513  1.293013 ]

.. autoclass:: BRISMFLearner
   :members:
   :special-members: __init__



.. index:: climf
   pair: recommenders; climf

=====
CLiMF
=====

CLiMF (Collaborative Less-is-More Filtering) is used in scenarios with binary
relevance data. Hence, it's focused on improving top-k recommendations through
ranking by directly maximizing the Mean Reciprocal Rank (MRR).

Following a similar technique as other iterative approaches, the two low-rank matrices
can be randomly initialize and then optimize through a training loss like this:

.. math::
	F(U,V) = \sum _{ i=1 }^{ M }{ \sum _{ j=1 }^{ N }{ { Y }_{ ij }[\ln{\quad g({ U }_{ i }^{ T }V_{ i })} +\sum _{ k=1 }^{ N }{ \ln { (1-{ Y }_{ ik }g({ U }_{ i }^{ T }V_{ k }-{ U }_{ i }^{ T }V_{ j })) }  } ] }  } -\frac { \lambda  }{ 2 } ({ \left\| U \right\|  }^{ 2 }+{ \left\| V \right\|  }^{ 2 })

Example
-------

.. code-block:: python
   :linenos:

    import Orange
    from orangecontrib.recommendation import CLiMFLearner

    # Load data and train the model
    data = Orange.data.Table('epinions_train.tab')
    learner = CLiMFLearner(K=10, steps=10, alpha=0.0001, beta=0.001)
    recommender = learner(data)

    # Load testing set
    data = Orange.data.Table('epinions_test.tab')

    # Compute predictions
    y_pred = recommender(data)

    # Get relevant items for the user[i]
    all_items_u = []
    for i in data.X[:, recommender.order[0]]:
       items_u = data.X[data.X[:, recommender.order[0]] == i][:, recommender.order[1]]
       all_items_u.append(items_u)

    # Compute Mean Reciprocal Rank (MRR)
    mrr = MeanReciprocalRank(results=y_pred, query=all_items_u)
    print('MRR: %.3f' % mrr)
    >>>
    MRR: 0.481

.. autoclass:: CLiMFLearner
   :members:
   :special-members: __init__

