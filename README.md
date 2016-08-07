Orange3 Recommendation
======================

[![Build Status](https://travis-ci.org/salvacarrion/orange3-recommendation.svg?branch=master)](https://travis-ci.org/salvacarrion/orange3-recommendation)
[![codecov](https://codecov.io/gh/salvacarrion/orange3-recommendation/branch/master/graph/badge.svg)](https://codecov.io/gh/salvacarrion/orange3-recommendation)
[![Documentation Status](https://readthedocs.org/projects/orange3-recommendation/badge/?version=latest)](http://orange3-recommendation.readthedocs.io/en/latest/?badge=latest)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/9488fe350c4446f395d0d260f8be49f5)](https://www.codacy.com/app/salva-carrion/orange3-recommendation?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=salvacarrion/orange3-recommendation&amp;utm_campaign=Badge_Grade)

Orange3 Recommendation is a Python module that extends [Orange3](http://orange.biolab.si) to include support for recommender systems.
All features can be combined with powerful data mining techniques from the Orange data mining framework.

For more information, see our [documentation](http://orange3-recommendation.readthedocs.io)

**DISCLAIMER: THIS LIBRARY IS STILL IN A VERY EARLY DEVELOPMENT STAGE (ALPHA)**
_Currently the algorithms are too slow to use them in big datasets, so right now this 
library is oriented towards educational purposes.
Further modifications have to be done in the main architecture of the library. Moreover, 
extra parameters must be added for latent factor models, along with a function to check the
convergece rate of SGD/ALS._
 
Dependencies
============

Orange3-Recommendation is tested to work under Python 3.

The required dependencies to build the software are Numpy >= 1.9.0 and Scikit-Learn >= 0.16


Install
=======

This package uses distutils, which is the default way of installing
python modules. To install in your home directory, use:

    python setup.py install --user

To install for all users on Unix/Linux::

    python setup.py build
    sudo python setup.py install

For development mode use:

    python setup.py develop
    


Widget usage
------------

After the installation, the widgets from this add-on are registered with Orange. To run Orange from the terminal,
use

    python3 -m Orange.canvas

new widgets are in the toolbox bar under *Recommendation* section.
      

Scripting
---------
All modules can be found inside **orangecontrib.recommendation.***. Thus, to import all modules we can type:

    from orangecontrib.recommendation import *
    
    
**Rating pairs (user, item):**

Let's presume that we want to load a dataset, train it and predict its first three pairs of (id_user, id_item)

    >>> import Orange
    >>> from orangecontrib.recommendation import BRISMFLearner
    >>> data = Orange.data.Table('movielens100k.tab')
    >>> learner = BRISMFLearner(num_factors=15, num_iter=25, learning_rate=0.07, lmbda=0.1)
    >>> recommender = learner(data)
    >>> prediction = recommender(data[:3])
    >>> print(prediction)
    [ 3.79505151  3.75096513  1.293013 ]
    
    
**Recommend items for set of users:**

Now we want to get all the predictions (all items) for a set of users:

    >>> import numpy as np
    >>> indices_users = np.array([4, 12, 36])
    >>> prediction = recommender.predict_items(indices_users)
    >>> print(prediction)
    [[ 1.34743879  4.61513578  3.90757263 ...,  3.03535099  4.08221699 4.26139511]
     [ 1.16652757  4.5516808   3.9867497  ...,  2.94690548  3.67274108 4.1868596 ]
     [ 2.74395768  4.04859096  4.04553826 ...,  3.22923456  3.69682699 4.95043435]]

Performance
-----------


**MovieLens100K:** (Loading time: 0.428s)

|      Algorithm     |  RMSE | MAE |  Train time  |          Settings          |
|:------------------:|:-----:|:---:|:------------:|:--------------------------:|
| Global Average     | 1.126 |  -  |    0.001s    |              -             |
| Item Average       | 1.000 |  -  |    0.001s    |              -             |
| User Average       | 1.031 |  -  |    0.001s    |              -             |
| User-Item Baseline | 0.938 |  -  |    0.001s    |              -             |
| BRISMF             | - |  -  |  2.067s/iter | num_factors=15; learning_rate=0.07; lmbda=0.1 |

**MovieLens1M:** (Loading time: 4.535s)

|      Algorithm     |  RMSE | MAE |  Train time   |          Settings          |
|:------------------:|:-----:|:---:|:-------------:|:--------------------------:|
| Global Average     | 1.117 |  -  |    0.010s     |              -             |
| Item Average       | 0.975 |  -  |    0.018s     |              -             |
| User Average       | 1.028 |  -  |    0.021s     |              -             |
| User-Item Baseline | 0.924 |  -  |    0.027s     |              -             |


**MovieLens10M:** (Loading time: 49.804s)

|      Algorithm     |  RMSE | MAE |   Train time  |          Settings          |
|:------------------:|:-----:|:---:|:-------------:|:--------------------------:|
| Global Average     | 1.060 |  -  |     0.129s    |              -             |
| Item Average       | 0.942 |  -  |     0.256s    |              -             |
| User Average       | 0.970 |  -  |     0.256s    |              -             |
| User-Item Baseline | 0.877 |  -  |     0.361s    |              -             |


        
Relevant links
==============

- Official source code repo: [https://github.com/salvacarrion/orange3-recommendation](https://github.com/salvacarrion/orange3-recommendation)
- HTML documentation: [http://orange3-recommendation.readthedocs.io](http://orange3-recommendation.readthedocs.io)
- Download releases: [https://github.com/salvacarrion/orange3-recommendation/releases](https://github.com/salvacarrion/orange3-recommendation/releases)
- Issue tracker: [https://github.com/salvacarrion/orange3-recommendation/issues](https://github.com/salvacarrion/orange3-recommendation/issues)

