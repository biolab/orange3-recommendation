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