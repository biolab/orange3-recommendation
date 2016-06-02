Orange3 RecommenderSystems Add-on
======================

Orange3 RecommenderSystems extends [Orange3](http://orange.biolab.si), a data mining software
package, with common functionality for make recommendations. It provides access
to publicly available data, like MovieLens, Yahoo! Music, Flixster,... All features can be combined with powerful data mining techniques
from the Orange data mining framework.

Last results
------------

    BRISMF -> 150 ratings, 100 users, 25 items.
    ------------------------------------------
    
    DENSE:
        - RMSE: 0.390
        - Time mean: 0.000s
        - Time: 0.277s
        
        * Mac OSX has a feature to compress the memory, so dense matrix are automatically manage as a kind of dense matrices.
        
    SPARSE:
        - RMSE: 0.379
        - Time mean: 0.002s
        - Time: 1.237s
        
        * Too slow (x4-5), I don't know why. (CSR<CSC<COO)
    

        
RECENT CHANGES
--------------

    1. I've temporarily removed the CLiMF module because it had no sense to have several modules with the same problems.
       Therefore, when the problems of BRISMF are solved, I will commit CLiMF module again.



Installation
------------

To install the add-on, run

    python setup.py install

To register this add-on with Orange, but keep the code in the development directory (do not copy it to 
Python's site-packages directory), run

    python setup.py develop

Usage
-----

After the installation, the widget from this add-on is registered with Orange. To run Orange from the terminal,
use

    python -m Orange.canvas

The new widget appears in the toolbox bar under the section Example.

![screenshot](https://github.com/biolab/orange3-example-addon/blob/master/screenshot.png)
