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
    
**Recommend items for set of users:**

    import Orange

    data = Orange.data.Table('ratings.tab')

    learner = brismf.BRISMFLearner(K=2, steps=100, verbose=True)
    recommender = learner(data)

    indices_users = np.array([0, 1, 2, 3, 4])
    prediction = recommender.predict_items(indices_users)
    print(prediction)
    
    
    >   [[5 3 0 1]
         [4 0 0 1]
         [1 1 0 5]
         [1 0 0 4]
         [0 1 5 4]]
    >    - Time mean (dense): 0.001s
       
    >    - RMSE: 0.052
    >    [[ 5.0871884   3.0028073  -2.02109583  1.09584045]
         [ 4.0159966   1.02711254  0.77697579  1.0235315 ]
         [ 1.06961718  1.00407018  0.43729543  5.05828816]
         [ 1.00851939  0.38358066  1.23864854  4.01044684]
         [ 2.65224422  0.93723289  4.92663325  3.98136032]]

**Rating pairs (user, item):**

    import Orange

    data = Orange.data.Table('ratings.tab')

    learner = brismf.BRISMFLearner(K=2, steps=100, verbose=True)
    recommender = learner(data)
    
    prediction = recommender(data[:3])  # Recommend first 3 tuples X (U_id, I_id)
    print(prediction)
    
    > [ 5.20604847  3.66934749  2.14537758]
    
    
    


**Evaluation:**

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
        
    # Small bug -> Need to be solved
    > Learner  RMSE  R2
      BRISMF  2.80 -2.21
    
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