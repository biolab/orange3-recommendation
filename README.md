Orange3 RecommenderSystems Add-on
======================

Orange3 RecommenderSystems extends [Orange3](http://orange.biolab.si), a data mining software
package, with common functionality for make recommendations. It provides access
to publicly available data, like MovieLens, Yahoo! Music, Flixster,... Further,
it **supports GPU acceleration**. All features can be combined with powerful data mining techniques
from the Orange data mining framework.

    Using gpu device 0: GeForce GTX TITAN Black (CNMeM is disabled)
    ---
    Settings:
        - Size: 5000x5000
        - Iterations: 10

    Times:
        - GPU (Theano): 1.798s
        - CPU: 50.011s

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
