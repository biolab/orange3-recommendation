########################################
Recommender Systems (``recommendation``)
########################################

.. automodule:: orangecontrib.recommendation

Orange3 Recommendation is a Python module that extends Orange3 to include
support for recommender systems. All features can be combined with powerful data
mining techniques from the Orange data mining framework.


Overview
--------

**Rating pairs (user, item):**

Let's presume that we want to load a dataset, train it and predict its first three pairs of (id_user, id_item)

    >>> import Orange
    >>> from orangecontrib.recommendation import BRISMFLearner
    >>> data = Orange.data.Table('MovieLens100K.tab')
    >>> learner = BRISMFLearner(K=10, steps=5, alpha=0.05, beta=0.01)
    >>> recommender = learner(data)
    >>> prediction = recommender(data[:3])
    >>> print(prediction)
    [ 4.19462862  4.09710973  1.11653675]


**Recommend items for set of users:**

Now we want to get all the predictions (all items) for a set of users:

    >>> import Orange
    >>> from orangecontrib.recommendation import BRISMFLearner
    >>> import numpy as np
    >>> data = Orange.data.Table('ratings.tab')
    >>> learner = BRISMFLearner(K=10, steps=5, alpha=0.05, beta=0.01)
    >>> recommender = learner(data)
    >>> indices_users = np.array([0, 2, 4])
    >>> prediction = recommender.predict_items(indices_users)
    >>> print(prediction)
    [[ 1.22709168  2.82786491  4.00826241  4.8979855   2.67956549]
    [ 0.84144603  2.34508053  4.91226517  4.66622242  2.23030677]
    [ 4.0537457   4.94304479  1.14010409  1.31233216  3.3946432 ]]


**Evaluation:**

Finally, we want to known which of a list of recommender performs better on our dataset. Therefore,
we perform cross-validation over a list of learners:


    >>> import Orange
    >>> from Orange.evaluation.testing import CrossValidation
    >>> from orangecontrib.recommendation import BRISMFLearner
    >>> data = Orange.data.Table('MovieLens100K.tab')
    >>> global_avg = GlobalAvgLearner()
    >>> items_avg = ItemAvgLearner()
    >>> users_avg = UserAvgLearner()
    >>> useritem_baseline = UserItemBaselineLearner()
    >>> brismf = BRISMFLearner(K=15, steps=15, alpha=0.07, beta=0.1)
    >>> learners = [global_avg, items_avg, users_avg, useritem_baseline, brismf]
    >>> res = Orange.evaluation.CrossValidation(data, learners, k=5)
    >>> rmse = Orange.evaluation.RMSE(res)
    >>> r2 = Orange.evaluation.R2(res)
    >>> print("Learner  RMSE  R2")
    >>> for i in range(len(learners)):
    >>>     print("{:8s} {:.2f} {:5.2f}".format(learners[i].name, rmse[i], r2[i]))
    Learner                   RMSE  R2
      - Global average        1.13 -0.00
      - Item average          1.03  0.16
      - User average          1.04  0.14
      - User-Item Baseline    0.98  0.25
      - BRISMF                0.96  0.28



.. index:: .. index:: global_average
   pair: recommenders; global_average

Global Average
--------------

Global Average uses the average rating value of all ratings to make predictions.

Example
=======

    >>> import Orange
    >>> from orangecontrib.recommendation import GlobalAvgLearner
    >>> data = Orange.data.Table('MovieLens100K.tab')
    >>> learner = GlobalAvgLearner()
    >>> recommender = learner(data)
    >>> prediction = recommender(data[:3])
    >>> print(prediction)
    [ 3.52986  3.52986  3.52986]

.. autoclass:: GlobalAvgLearner
   :members:



.. index:: .. index:: user_average
   pair: recommenders; user_average

User Average
------------

User Average uses the average rating value of a user to make predictions.

Example
=======

    >>> import Orange
    >>> from orangecontrib.recommendation import UserAvgLearner
    >>> data = Orange.data.Table('MovieLens100K.tab')
    >>> learner = UserAvgLearner()
    >>> recommender = learner(data)
    >>> prediction = recommender(data[:3])
    >>> print(prediction)
    [ 3.61538462  3.41304348  3.3515625 ]

.. autoclass:: UserAvgLearner
   :members:



.. index:: .. index:: item_average
   pair: recommenders; item_average

Item Average
------------

Item Average uses the average rating value of an item to make predictions.

Example
=======

    >>> import Orange
    >>> from orangecontrib.recommendation import ItemAvgLearner
    >>> data = Orange.data.Table('MovieLens100K.tab')
    >>> learner = ItemAvgLearner()
    >>> recommender = learner(data)
    >>> prediction = recommender(data[:3])
    >>> print(prediction)
    [ 3.99145299  4.16161616  2.15384615]

.. autoclass:: ItemAvgLearner
   :members:



.. index:: .. index:: user_item_baseline
   pair: recommenders; user_item_baseline

User-Item Baseline
------------------

User-Item Baseline takes the bias of users and items plus the global average to
make predictions.

Example
=======

    >>> import Orange
    >>> from orangecontrib.recommendation import UserItemBaselineLearner
    >>> data = Orange.data.Table('MovieLens100K.tab')
    >>> learner = UserItemBaselineLearner()
    >>> recommender = learner(data)
    >>> prediction = recommender(data[:3])
    >>> print(prediction)
    [ 4.07697761  4.04479964  1.97554865]

.. autoclass:: UserItemBaselineLearner
   :members:



.. index:: .. index:: brismf
pair: recommenders; brismf

BRISMF
------

This model uses stochastic gradient descent to find the values of two low-rank
matrices which represents the user and item factors. This object can factorize
either dense or sparse matrices.

Example
=======

    >>> import Orange
    >>> from orangecontrib.recommendation import BRISMFLearner
    >>> data = Orange.data.Table('MovieLens100K.tab')
    >>> learner = BRISMFLearner(K=10, steps=5, alpha=0.05, beta=0.01)
    >>> recommender = learner(data)
    >>> prediction = recommender(data[:3])
    >>> print(prediction)
    [ 4.19462862  4.09710973  1.11653675]

.. autoclass:: BRISMFLearner
   :members:



.. index:: .. index:: climf
pair: recommenders; climf

CLiMF
-----

This model is focused on improving top-k recommendations through ranking by
directly maximizing the Mean Reciprocal Rank (MRR).

Example
=======

    >>> import Orange
    >>> from orangecontrib.recommendation import CLiMFLearner
    >>> data = Orange.data.Table('binary_data.tab')
    >>> learner = CLiMFLearner(K=10, steps=5, alpha=0.05, beta=0.01)
    >>> recommender = learner(data)
    >>> prediction = recommender(data[:3])
    >>> print(prediction)

.. autoclass:: CLiMFLearner
   :members:

