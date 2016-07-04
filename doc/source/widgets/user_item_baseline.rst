==================
User-Item baseline
==================

.. figure:: icons/user-item-baseline.svg
    :width: 64pt

This model takes the bias of users and items plus the global average to make
predictions.


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

**User-Item baseline** widget uses computes the bias of users and items plus the
global average to make predictions.


Example
-------


Below is a simple workflow showing how to use both the *Predictor* and
the *Learner* output. For the *Predictor* we input the prediction model
into :doc:`Predictions<../evaluation/predictions>` widget and view the results in :doc:`Data Table<../data/datatable>`. For
*Learner* we can compare different learners in :doc:`Test&Score<../evaluation/testlearners>` widget.

.. figure:: images/example.png
