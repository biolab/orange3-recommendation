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
    import numpy as np
    from orangecontrib.recommendation import CLiMFLearner

    # Load data
    data = Orange.data.Table('epinions_train.tab')

    # Train recommender
    learner = CLiMFLearner(num_factors=10, num_iter=10, learning_rate=0.0001, lmbda=0.001)
    recommender = learner(data)

    # Load test dataset
    testdata = Orange.data.Table('epinions_test.tab')

    # Sample users
    num_users = len(recommender.U)
    num_samples = min(num_users, 1000)  # max. number to sample
    users_sampled = np.random.choice(np.arange(num_users), num_samples)

    # Compute Mean Reciprocal Rank (MRR)
    mrr, _ = recommender.compute_mrr(data=testdata, users=users_sampled)
    print('MRR: %.4f' % mrr)
    >>>
    MRR: 0.3975

.. autoclass:: CLiMFLearner
   :members:
   :special-members: __init__