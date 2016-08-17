SVD++
=====

.. figure:: ../resources/icons/svdplusplus.svg
    :width: 64pt

Matrix factorization model which makes use of implicit feedback information.


Signals
-------

**Inputs**:

-  **Data**

   Data set.

-  **Preprocessor**

   Preprocessed data.

-  **Feedback information**

   Implicit feedback information.
   Optional, if None (default), it will be inferred from the ratings.

**Outputs**:

-  **Learner**

   The learning algorithm with the supplied parameters.

-  **Predictor**

   Trained recommender. Signal *Predictor* sends the output signal only if
   input *Data* is present.

-  **P**

   Latent features of the users.

-  **Q**

   Latent features of the items.

-  **Y**

   Latent features of the implicit information.


Description
-----------

**SVD++** widget uses a biased regularized algorithm which makes use of implicit
feedback information to factorize a matrix into three low rank matrices as it's
explained in *Y. Koren, Factorization Meets the Neighborhood: a Multifaceted
Collaborative Filtering Model*


Example
-------

Below is a simple workflow showing how to use both the *Predictor* and
the *Learner* output. For the *Predictor* we input the prediction model
into `Predictions <http://docs.orange.biolab.si/3/visual-programming/widgets/evaluation/predictions.html>`_
widget and view the results in `Data Table <http://docs.orange.biolab.si/3/visual-programming/widgets/data/datatable.html>`_.
For *Learner* we can compare different learners in `Test&Score <http://docs.orange.biolab.si/3/visual-programming/widgets/evaluation/testlearners.html>`_ widget.

.. figure:: ../resources/images/example_latent_factor_models.png

