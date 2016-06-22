======
BRISMF
======

.. figure:: icons/brismf.png

Matrix factorization with explicit ratings, learning is performed by stochastic
gradient descent.


Signals
-------

**Inputs**:

-  **Data**

Data set.

-  **Preprocessor**

Preprocessed data.

**Outputs**:

-  **Learner**

The learning algorithm with the supplied parameters

-  **Predictor**

Trained recommender. Signal *Predictor* sends the output signal only if
input *Data* is present.


Description
-----------

**BRISMF** widget uses a biased regularized algorithm to factorize a matrix into
two low rank matrices as it's explained in *Y. Koren, R. Bell, C. Volinsky,
Matrix Factorization Techniques for Recommender Sys- tems. IEE Computer
Society, 2009.*


Example
-------

No example yet for this widget.
