TrustSVD
========

.. figure:: ../resources/icons/trustsvd.svg
    :width: 64pt

Trust-based matrix factorization, which extends SVD++ with trust information.


Signals
-------

**Inputs**:

-  **Data**

   Data set.

-  **Preprocessor**

   Preprocessed data.

-  **Trust information**

   Trust information. The weights of the connections can be integer or float
   (binary relations can represented by 0 or 1).


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

-  **W**

   Latent features of the trust information.

Description
-----------

**TrustSVD** widget uses a biased regularized algorithm which makes use of
implicit feedback information and trust information to factorize a matrix into
four low rank matrices as it's explained in *Guibing Guo, Jie Zhang, Neil
Yorke-Smith, TrustSVD: Collaborative Filtering with Both the Explicit and
Implicit Influence of User Trust and of Item Ratings*


Example
-------

Below is a simple workflow showing how to use both the *Predictor* and
the *Learner* output. For the *Predictor* we input the prediction model
into `Predictions <http://docs.orange.biolab.si/3/visual-programming/widgets/evaluation/predictions.html>`_
widget and view the results in `Data Table <http://docs.orange.biolab.si/3/visual-programming/widgets/data/datatable.html>`_.
For *Learner* we can compare different learners in `Test&Score <http://docs.orange.biolab.si/3/visual-programming/widgets/evaluation/testlearners.html>`_ widget.

.. figure:: ../resources/images/example_latent_factor_models.png

