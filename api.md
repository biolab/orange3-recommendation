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
           |- global_avg
           |- item_avg
           |- user_avg
           |- user_item_baseline
           |- brismf
           |- climf*
           |- SDV++*
           |- TrustSVD*
       |- tests
       
* Not added yet
```



Interface
---------

All modules can be found inside **orangecontrib.recommendation.***

    from orangecontrib.recommendation import *
    
**Recommend items for set of users:**

    import Orange
    from orangecontrib.recommendation import BRISMFLearner
    
    data = Orange.data.Table('ratings.tab')

    learner = BRISMFLearner(K=2, steps=100, verbose=True)
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
    from orangecontrib.recommendation import BRISMFLearner
    
    data = Orange.data.Table('ratings.tab')

    learner = BRISMFLearner(K=5, steps=20, verbose=False)
    recommender = learner(data)
    
    prediction = recommender(data[:3])  # Recommend first 3 tuples X (U_id, I_id)
    print(prediction)
    
    > [ 5.20604847  3.66934749  2.14537758]
    
    
**Evaluation:**

    from orangecontrib.recommendation import BRISMFLearner
    from Orange.evaluation.testing import CrossValidation
    
    data = Orange.data.Table('MovieLens100K.tab')
    
    global_avg = GlobalAvgLearner()
    items_avg = ItemAvgLearner()
    users_avg = UserAvgLearner()
    useritem_baseline = UserItemBaselineLearner()
    brismf = BRISMFLearner(K=15, steps=15, alpha=0.07, beta=0.1)
    learners = [global_avg, items_avg, users_avg, useritem_baseline, brismf]
    
    res = Orange.evaluation.CrossValidation(data, learners, k=5)
    rmse = Orange.evaluation.RMSE(res)
    r2 = Orange.evaluation.R2(res)
    
    print("Learner  RMSE  R2")
    for i in range(len(learners)):
        print("{:8s} {:.2f} {:5.2f}".format(learners[i].name, rmse[i], r2[i]))
        
    > Learner                   RMSE  R2
        - Global average        1.13 -0.00
        - Item average          1.03  0.16
        - User average          1.04  0.14
        - User-Item Baseline    0.98  0.25
        - BRISMF                0.96  0.28
    
