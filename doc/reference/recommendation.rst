########################################
Recommender Systems (``recommendation``)
########################################

.. automodule:: orangecontrib.recommendation



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
   
   
