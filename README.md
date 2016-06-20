Orange3 Recommendation
======================

[![Build Status](https://travis-ci.org/salvacarrion/orange3-recommendation.svg?branch=master)](https://travis-ci.org/salvacarrion/orange3-recommendation)
[![codecov](https://codecov.io/gh/salvacarrion/orange3-recommendation/branch/master/graph/badge.svg)](https://codecov.io/gh/salvacarrion/orange3-recommendation)
[![Documentation Status](https://readthedocs.org/projects/orange3-recommendation/badge/?version=latest)](http://orange3-recommendation.readthedocs.io/en/latest/?badge=latest)

Orange3 Recommendation is a Python module that extends [Orange3](http://orange.biolab.si) to include support for recommender systems.
All features can be combined with powerful data mining techniques from the Orange data mining framework.


Dependencies
============

Orange3-Recommendation is tested to work under Python 3.

The required dependencies to build the software are Numpy >= 1.9.0 and Scikit-Learn >= 0.16


Install
=======

This package uses distutils, which is the default way of installing
python modules. To install in your home directory, use::

    python setup.py install --user

To install for all users on Unix/Linux::

    python setup.py build
    sudo python setup.py install

For development mode use::

    python setup.py develop
    

Architecure
-----------

```
orangecontrib
    |
    |- recommendation
       |- datasets
       |- models
           |- global_avg
           |- item_avg
           |- user_avg
           |- user_item_baseline
           |- brismf
           |- climf
           |- SDV++*
           |- TrustSVD*
       |- tests
       
* Not added yet
```


Usage
-----

All modules can be found inside **orangecontrib.recommendation.***. Thus, to import all modules we can type:

    from orangecontrib.recommendation import *
    
    
**Rating pairs (user, item):**

Let's presume that we want to load a dataset, train it and predict its first three pairs of (id_user, id_item)

    >>> import Orange
    >>> from orangecontrib.recommendation import BRISMFLearner
    
    >>> data = Orange.data.Table('MovieLens100K.tab')
    
    >>> learner = BRISMFLearner(K=10, steps=5, alpha=0.05, beta=0.01)
    >>> recommender = learner(data)
    
    >>> prediction = recommender(data[:3])
    >>> print(prediction)
    [ 4.19462862  4.09710973  1.11653675]
    
    
**Recommend items for set of users:**

Now we want to get all the predictions (all items) for a set of users:

    >>> import Orange
    >>> from orangecontrib.recommendation import BRISMFLearner
    >>> import numpy as np
    
    >>> data = Orange.data.Table('ratings.tab')

    >>> learner = BRISMFLearner(K=10, steps=5, alpha=0.05, beta=0.01)
    >>> recommender = learner(data)

    >>> indices_users = np.array([0, 2, 4])
    >>> prediction = recommender.predict_items(indices_users)
    >>> print(prediction)
    [[ 1.22709168  2.82786491  4.00826241  4.8979855   2.67956549]
    [ 0.84144603  2.34508053  4.91226517  4.66622242  2.23030677]
    [ 4.0537457   4.94304479  1.14010409  1.31233216  3.3946432 ]]

    
**Evaluation:**

Finally, we want to known which of a list of recommender performs better on our dataset. Therefore,
we perform cross-validation over a list of learners:

    
    >>> import Orange
    >>> from Orange.evaluation.testing import CrossValidation
    >>> from orangecontrib.recommendation import BRISMFLearner
    
    >>> data = Orange.data.Table('MovieLens100K.tab')
    
    >>> global_avg = GlobalAvgLearner()
    >>> items_avg = ItemAvgLearner()
    >>> users_avg = UserAvgLearner()
    >>> useritem_baseline = UserItemBaselineLearner()
    >>> brismf = BRISMFLearner(K=15, steps=15, alpha=0.07, beta=0.1)
    >>> learners = [global_avg, items_avg, users_avg, useritem_baseline, brismf]
    
    >>> res = Orange.evaluation.CrossValidation(data, learners, k=5)
    >>> rmse = Orange.evaluation.RMSE(res)
    >>> r2 = Orange.evaluation.R2(res)
    
    >>> print("Learner  RMSE  R2")
    >>> for i in range(len(learners)):
    >>>     print("{:8s} {:.2f} {:5.2f}".format(learners[i].name, rmse[i], r2[i]))
        
    Learner                   RMSE  R2
      - Global average        1.13 -0.00
      - Item average          1.03  0.16
      - User average          1.04  0.14
      - User-Item Baseline    0.98  0.25
      - BRISMF                0.96  0.28
      
      
Performance
-----------

    Times on MovieLens100K:
        - Loading time: 0.428s
        - Time (GlobalAvgLearner): 0.001s
        - Time (ItemAvgLearner): 0.001s
        - Time (UserAvgLearner): 0.001s
        - Time (UserItemBaselineLearner): 0.001s
        - Time (BRISMFLearner): 1.453s/iter; k=15; alpha=0.07; beta=0.1
    
    RMSE on MovieLens100K:
        - RMSE (GlobalAvgLearner): 1.126
        - RMSE (ItemAvgLearner): 1.000
        - RMSE (UserAvgLearner): 1.031
        - RMSE (UserItemBaselineLearner): 0.938
        - RMSE (BRISMFLearner): 0.823
    ----------------------------------------------------
    
    Times on MovieLens1M:
        - Loading time: 4.535s
        - Time (GlobalAvgLearner): 0.010s
        - Time (ItemAvgLearner): 0.018s
        - Time (UserAvgLearner): 0.021s
        - Time (UserItemBaselineLearner): 0.027s
        - Time (BRISMFLearner): 14.347s/iter; k=15; alpha=0.07; beta=0.1
        
    RMSE on MovieLens1M:
        - RMSE (GlobalAvgLearner): 1.117
        - RMSE (ItemAvgLearner): 0.975
        - RMSE (UserAvgLearner): 1.028
        - RMSE (UserItemBaselineLearner): 0.924
        - RMSE (BRISMFLearner): 0.872
    ----------------------------------------------------
    
    Times on MovieLens10M:
        - Loading time: 49.804s
        - Time (GlobalAvgLearner): 0.129s
        - Time (ItemAvgLearner): 0.256s
        - Time (UserAvgLearner): 0.256s
        - Time (UserItemBaselineLearner): 0.361s
        - Time (BRISMFLearner): 138.309s/iter; k=15; alpha=0.07; beta=0.1
        
    RMSE on MovieLens10M:
        - RMSE (GlobalAvgLearner): 1.060
        - RMSE (ItemAvgLearner): 0.942
        - RMSE (UserAvgLearner): 0.970
        - RMSE (UserItemBaselineLearner): 0.877
        - RMSE (BRISMFLearner): 0.841
        
        
Relevant links
==============

- Official source code repo: [https://github.com/salvacarrion/orange3-recommendation](https://github.com/salvacarrion/orange3-recommendation)
- HTML documentation: [http://orange3-recommendation.readthedocs.io](http://orange3-recommendation.readthedocs.io)
- Download releases: [https://github.com/salvacarrion/orange3-recommendation/releases](https://github.com/salvacarrion/orange3-recommendation/releases)
- Issue tracker: [https://github.com/salvacarrion/orange3-recommendation/issues](https://github.com/salvacarrion/orange3-recommendation/issues)

