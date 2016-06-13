Orange3 RecommenderSystems Add-on
=================================

[![Build Status](https://travis-ci.org/salvacarrion/orange3-recommendersystems.svg?branch=master)](https://travis-ci.org/salvacarrion/orange3-recommendersystems)
[![Documentation Status](https://readthedocs.org/projects/orange3-recommendersystems/badge/?version=latest)](http://orange3-recommendersystems.readthedocs.io/en/latest/?badge=latest)
                
Orange3 RecommenderSystems extends [Orange3](http://orange.biolab.si), a data mining software
package, with common functionality for make recommendations. It provides access
to publicly available data, like MovieLens, Yahoo! Music, Flixster,... All features can be combined with powerful data mining techniques
from the Orange data mining framework.

Last results
------------

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
        
        
RECENT CHANGES
--------------

    1. I've temporarily removed the CLiMF module because it had no sense to have several modules with the same problems.
       Therefore, when the problems of BRISMF are solved, I will commit CLiMF module again.


PROBLEMS 
--------
    None

DETAILED
--------

More information in [API.md](https://github.com/salvacarrion/orange3-recommendersystems/blob/master/api.md)

