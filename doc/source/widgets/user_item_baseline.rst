==================
User-Item baseline
==================

.. figure:: icons/user-item-baseline.svg

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

**Global average** widget uses computes the bias of users and items plus the
global average to make predictions.


Example
-------

No example yet for this widget.
