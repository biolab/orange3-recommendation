=====
CLiMF
=====

.. figure:: icons/climf.svg
    :width: 64pt

Matrix factorization for scenarios with binary relevance data when only a few
(k) items are recommended to individual users. It improves top-k recommendations
through ranking by directly maximizing the Mean Reciprocal Rank (MRR)


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

-  **U**

Latent features of the users

-  **V**

Latent features of the items



Description
-----------

**CLiMF** widget uses the algorithm described in *CLiMF: Learning to Maximize
Reciprocal Rank with Collaborative Less-is-More Filtering Yue Shi, Martha
Larson, Alexandros Karatzoglou, Nuria Oliver, Linas Baltrunas, Alan Hanjalic
ACM RecSys 2012*
