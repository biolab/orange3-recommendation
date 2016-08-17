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