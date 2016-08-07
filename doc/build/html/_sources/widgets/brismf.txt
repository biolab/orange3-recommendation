======
BRISMF
======

.. figure:: icons/brismf.svg
    :width: 64pt

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

-  **P**

Latent features of the users

-  **Q**

Latent features of the items


Description
-----------

**BRISMF** widget uses a biased regularized algorithm to factorize a matrix into
two low rank matrices as it's explained in *Y. Koren, R. Bell, C. Volinsky,
Matrix Factorization Techniques for Recommender Systems. IEE Computer
Society, 2009.*


Example
-------

Below is a simple workflow showing how to use both the *Predictor* and
the *Learner* output. For the *Predictor* we input the prediction model
into `Predictions <http://docs.orange.biolab.si/3/visual-programming/widgets/evaluation/predictions.html>`_
widget and view the results in `Data Table <http://docs.orange.biolab.si/3/visual-programming/widgets/data/datatable.html>`_.
For *Learner* we can compare different learners in `Test&Score <http://docs.orange.biolab.si/3/visual-programming/widgets/evaluation/testlearners.html>`_ widget.

.. figure:: images/example_latent_factor_models.png

