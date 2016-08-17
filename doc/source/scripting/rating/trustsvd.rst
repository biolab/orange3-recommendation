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
