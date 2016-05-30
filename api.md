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

Functions...


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
Functions...
