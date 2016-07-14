==============
Global average
==============

.. figure:: icons/average.svg
    :width: 64pt

Uses the average rating value of all ratings to make predictions.


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

**Global average** widget uses computes the average of all ratings to make
predictions.


Example
-------

Below is a simple workflow showing how to use both the *Predictor* and
the *Learner* output. For the *Predictor* we input the prediction model
into `Predictions <http://docs.orange.biolab.si/3/visual-programming/widgets/evaluation/predictions.html>`_
widget and view the results in `Data Table <http://docs.orange.biolab.si/3/visual-programming/widgets/data/datatable.html>`_.
For *Learner* we can compare different learners in `Test&Score <http://docs.orange.biolab.si/3/visual-programming/widgets/evaluation/testlearners.html>`_ widget.

.. figure:: images/example.png

