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
	\hat { r }_{ ui } = \mu


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
	\hat { r }_{ ui } = \mu_{u}

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
	\hat { r }_{ ui } = \mu_{i}

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
	\hat { r }_{ ui } = \mu + b_{u} + b_{i}

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



.. index:: svdplusplus
   pair: recommenders; svdplusplus

=====
SVD++
=====

SVD++ is matrix factorization model which makes use of implicit feedback
information.

User's predictions are defined as follows:

.. math::
	\hat { r }_{ ui } = \mu + b_u + b_i + \left(p_u + \frac{1}{\sqrt{|N(u)|}}\sum_{j\in N(u)} y_j \right)^T q_i


We learn the values of involved parameters by minimizing the regularized squared error function associated with:

.. math::
    \begin{split}
    \min _{ p*,q*,y*,b* }&{\sum _{ (u,i\in k) }{ { ({ r }_{ ui }-\mu -b_{ u }-b_{ i }-{ q }_{ i }^{ T }\left( p_{ u }+\frac { 1 }{ \sqrt { |N(u)| }  } \sum _{ j\in N(u) } y_{ j } \right) ) }^{ 2 }}} \\
    &+\lambda ({ { { b }_{ u } }^{ 2 }+{ { b }_{ i } }^{ 2 }+\left\| { p }_{ u } \right\|  }^{ 2 }+{ \left\| q_{ i } \right\|  }^{ 2 }+\sum _{ j\in N(u) }{ \left\| y_{ j } \right\|  } ^{ 2 })
    \end{split}

Example
-------

.. code-block:: python
   :linenos:

   import Orange
   from orangecontrib.recommendation import SVDPlusPlusLearner

   # Load data and train the model
   data = Orange.data.Table('movielens100k.tab')
   learner = SVDPlusPlusLearner(num_factors=15, num_iter=25, learning_rate=0.07, lmbda=0.1)
   recommender = learner(data)

   # Make predictions
   prediction = recommender(data[:3])
   print(prediction)

.. autoclass:: SVDPlusPlusLearner
   :members:
   :special-members: __init__



.. index:: trustsvd
   pair: recommenders; trustsvd

========
TrustSVD
========

TrustSVD is a trust-based matrix factorization, which extends SVD++ with trust
information.

User's predictions are defined as follows:

.. math::
	\hat { r }_{ ui }=\mu +b_{ u }+b_{ i }+{ q_{ i } }^{ \top  }\left( p_{ u }+{ \left| { I }_{ u } \right|  }^{ -\frac { 1 }{ 2 }  }\sum _{ i\in { I }_{ u } } y_{ i }+{ \left| { T }_{ u } \right|  }^{ -\frac { 1 }{ 2 }  }\sum _{ v\in { T }_{ u } } w_{ v } \right)

We learn the values of involved parameters by minimizing the regularized squared error function associated with:

.. math::
    \begin{split}
    \mathcal{L} &=\frac { 1 }{ 2 } \sum _{ u }{ \sum _{ j\in { I }_{ u } }{ { ({ \hat { r } }_{ u,j } -{ r }_{ u,j }) }^{ 2 } }  } + \frac { { \lambda  }_{ t } }{ 2 } \sum _{ u }{ \sum _{ v\in { T }_{ u } }{ { ( { \hat { t } }_{ u,v } -{ t }_{ u,v }) }^{ 2 } }  } \\
    &+\frac { { \lambda  } }{ 2 } \sum _{ u }^{  }{ { \left| { I }_{ u } \right|  }^{ -\frac { 1 }{ 2 }  }{ b }_{ u }^{ 2 } } +\frac { { \lambda  } }{ 2 } \sum _{ j }{ { \left| { U }_{ j } \right|  }^{ -\frac { 1 }{ 2 }  }{ b }_{ j }^{ 2 } } \\
    &+\sum _{ u }^{  }{ (\frac { { \lambda  } }{ 2 } { \left| { I }_{ u } \right|  }^{ -\frac { 1 }{ 2 }  }+\frac { { \lambda  }_{ t } }{ 2 } { \left| { T }_{ u } \right|  }^{ -\frac { 1 }{ 2 }  }{ )\left\| { p }_{ u } \right\|  }_{ F }^{ 2 } } \\
    &+\frac { { \lambda  } }{ 2 } \sum _{ j }{ { \left| { U }_{ j } \right|  }^{ -\frac { 1 }{ 2 }  }{ \left\| { q }_{ j } \right\|  }_{ F }^{ 2 } } +\frac { { \lambda  } }{ 2 } \sum _{ i }{ { \left| { U }_{ i } \right|  }^{ -\frac { 1 }{ 2 }  }{ \left\| { y }_{ i } \right\|  }_{ F }^{ 2 } } \\
    &+\frac { { \lambda  } }{ 2 } { \left| { T }_{ v }^{ + } \right|  }^{ -\frac { 1 }{ 2 }  }{ \left\| { w }_{ v } \right\|  }_{ F }^{ 2 }
    \end{split}

Example
-------

.. code-block:: python
   :linenos:

    import Orange
    from orangecontrib.recommendation import TrustSVDLearner

    # Load data and train the model
    ratings = Orange.data.Table('filmtrust/ratings.tab')
    trust = Orange.data.Table('filmtrust/trust.tab')
    learner = TrustSVDLearner(num_factors=15, num_iter=25, learning_rate=0.07,
                              lmbda=0.1, social_lmbda=0.05, trust=trust)
    recommender = learner(data)

    # Make predictions
    prediction = recommender(data[:3])
    print(prediction)

.. autoclass:: TrustSVDLearner
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
    learner = CLiMFLearner(num_factors=10, num_iter=10, learning_rate=0.0001, lmbda=0.001)
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

