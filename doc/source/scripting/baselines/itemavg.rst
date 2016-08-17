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