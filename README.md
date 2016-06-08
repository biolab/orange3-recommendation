Orange3 RecommenderSystems Add-on
======================

Orange3 RecommenderSystems extends [Orange3](http://orange.biolab.si), a data mining software
package, with common functionality for make recommendations. It provides access
to publicly available data, like MovieLens, Yahoo! Music, Flixster,... All features can be combined with powerful data mining techniques
from the Orange data mining framework.

Last results
------------

    Times on MovieLens100K:
        - Loading time: 2.848s
        - Time (GlobalAvgLearner): 0.001s
        - Time (ItemAvgLearner): 0.001s
        - Time (UserAvgLearner): 0.001s
        - Time (UserItemBaselineLearner): 0.001s
        - Time (BRISMFLearner): 1.999s/iter; k=5

    RMSE on MovieLens100K:
        - RMSE (GlobalAvgLearner): 1.126
        - RMSE (ItemAvgLearner): 1.000
        - RMSE (UserAvgLearner): 1.031
        - RMSE (UserItemBaselineLearner): 0.938
        - RMSE (BRISMFLearner): —Overflow—
    

        
RECENT CHANGES
--------------

    1. I've temporarily removed the CLiMF module because it had no sense to have several modules with the same problems.
       Therefore, when the problems of BRISMF are solved, I will commit CLiMF module again.


PROBLEMS 
--------
    1. BRISMF overflows from time to time.

DETAILED
--------

More information in [API.md](https://github.com/salvacarrion/orange3-recommendersystems/blob/master/api.md)

