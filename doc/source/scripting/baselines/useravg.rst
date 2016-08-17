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