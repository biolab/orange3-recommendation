Definition of the API
=====================

The Sphinx documentation can be found on root->doc->_build->html->index.html


Installation
------------

To install the add-on, run

    python setup.py install

or

    pip install .

To register this add-on with Orange, but keep the code in the development directory (do not copy it to 
Python's site-packages directory), run

    python setup.py develop

or

    pip install -e .


Hierarchy
---------
```
orangecontrib
    |
    |- recsystems
       |- datasets
       |- models
           |- model_based
               |- brismf
               |- climf*
               |- SDV++*
               |- TrustSVD*
       |- tests
       
* Not added yet
```



Interface
---------

Every model inside **orangecontrib.recsystems.models.model_based.\*** can be imported with **orangecontrib.recsystems.models.\***

    from orangecontrib.recsystems.models import brismf
    
Recommend items for a user:

    import Orange

    data = Orange.data.Table('ratings.tab')

    learner = brismf.BRISMFLearner(K=2, steps=100, verbose=True)
    recommender = learner(data)

    prediction = recommender.predict_items(user=1, sort=False, top=None)
    print(prediction[:, 1].T)
    

    >   [0.46595242  0.40092525  0.23869106  0.43917838  0.40014543  0.47886861
         0.4958955   0.56123877  0.49768542  0.5279589   0.46288913  0.34375892
         0.42346417  0.39698852  0.58667468  0.49516489  0.33460744  0.71041034
         0.50779647  0.28443737]

Rating of (user, item):

    import Orange

    data = Orange.data.Table('ratings.tab')

    learner = brismf.BRISMFLearner(K=2, steps=100, verbose=True)
    recommender = learner(data)
    
    indices = np.array([1, 5])  # Pairs (user, item)
    prediction = recommender(indices)  # Equivalent to recommender.predict(...)
    print(prediction)
    
    > 0.48718586


Evaluation:

    from recsystems.models import brismf
    from Orange.evaluation.testing import CrossValidation
    
    data = Orange.data.Table('ratings.tab')
    
    learners = [brismf.BRISMFLearner(K=2, steps=100)]
    
    res = Orange.evaluation.CrossValidation(data, learners, k=5)
    rmse = Orange.evaluation.RMSE(res)
    r2 = Orange.evaluation.R2(res)
    
    print("Learner  RMSE  R2")
    for i in range(len(learners)):
        print("{:8s} {:.2f} {:5.2f}".format(learners[i].name, rmse[i], r2[i]))
    
BRISMF
------

This module has two classes: **BRISMFLearner** and **BRISMFModel**

**BRISMFLearner**
```
    Biased Regularized Incremental Simultaneous Matrix Factorization

    This model uses stochastic gradient descent to find the ratings of two
    low-rank matrices which represents user and item factors. This object can
    factorize either dense or sparse matrices.

    Attributes:
        K: int, optional
            The number of latent factors.

        steps: int, optional (default = 100)
            The number of epochs of stochastic gradient descent.

        alpha: float, optional (default = 0.005)
            The learning rate of stochastic gradient descent.

        beta: float, optional (default = 0.02)
            The regularization parameter.

        verbose: boolean, optional (default = False)
            Prints information about the process.
```

Methods:

* fit(self, X, Y=None, W=None)

        This function calls the factorization method.

        Args:
            X: Matrix
                Data to fit.

        Returns:
            Model object (BRISMFModel)

* prepare_data(self, X):

        Function to remove NaNs from the data (preprocessor)

        Args:
            X: Matrix (data to fit).

        Returns:
            X (matrix)

* matrix_factorization(self, R, K, steps, alpha, beta, verbose=False)`

        Factorize either a dense matrix or a sparse matrix into two low-rank
         matrices which represents user and item factors.

        Args:
            R: Matrix
                Matrix to factorize. (Zeros are equivalent to unknown data)

            K: int
                The number of latent factors.

            steps: int
                The number of epochs of stochastic gradient descent.

            alpha: float
                The learning rate of stochastic gradient descent.

            beta: float
                The regularization parameter.

            verbose: boolean, optional
                If true, it outputs information about the process.

        Returns:
            P (matrix, UxK), Q (matrix, KxI) and bias (dictionary, 'delta items'
            , 'delta users', 'global mean items' and 'global mean users')

<br>
<br>

**BRISMFModel**

```
This model receives a learner and provides and interface to make the
        predictions for a given user.

        Args:
            P: Matrix
            Q: Matrix
            bias: dictionary
                'delta items', 'delta users', 'global mean items' and
                'global mean users'
```

Methods:

* predict(self, user, sort=True, top=None):

        This function receives the index of a user and returns its
                recomendations.
                Args:
                    user: int
                        Index of the user to which make the predictions.
        
                    sort: boolean, optional
                        If True, the returned array with the ratings will be sorted in
                        descending order.
        
                    top: int, optional
                        Return just the first k recommendations.
        
                Returns:
                    Array with the recommendations for a given user.