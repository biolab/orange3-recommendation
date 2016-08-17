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