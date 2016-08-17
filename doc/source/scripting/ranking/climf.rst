CLiMF
=====

CLiMF (Collaborative Less-is-More Filtering) is used in scenarios with binary
relevance data. Hence, it's focused on improving top-k recommendations through
ranking by directly maximizing the Mean Reciprocal Rank (MRR).

Following a similar technique as other iterative approaches, the two low-rank matrices
can be randomly initialize and then optimize through a training loss like this:

.. math::
	F(U,V) = \sum _{ i=1 }^{ M }{ \sum _{ j=1 }^{ N }{ { Y }_{ ij }[\ln{\quad g({ U }_{ i }^{ T }V_{ i })} +\sum _{ k=1 }^{ N }{ \ln { (1-{ Y }_{ ik }g({ U }_{ i }^{ T }V_{ k }-{ U }_{ i }^{ T }V_{ j })) }  } ] }  } -\frac { \lambda  }{ 2 } ({ \left\| U \right\|  }^{ 2 }+{ \left\| V \right\|  }^{ 2 })

**Note:** *Orange3 currently does not support ranking operations. Therefore,
this model cannot be used neither in cross-validation nor in the prediction
module available in Orange3*

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