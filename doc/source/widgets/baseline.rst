==================
Baseline
==================

.. figure:: icons/user-item-baseline.svg
    :width: 64pt

This widget includes four basic baseline models: Global average, User average,
Item average and User-Item baseline.

Signals
-------

**Inputs**:

-  **Data**

Data set.

-  **Preprocessor**

Preprocessed data.

**Outputs**:

-  **Learner**

The selected learner in the widget.

-  **Predictor**

Trained recommender. Signal *Predictor* sends the output signal only if
input *Data* is present.


Description
-----------

-  **Global average:**
Computes the average of all ratings and use it to make predictions.

-  **User average:**
Takes the average rating value of a user to make predictions.

-  **Item average:**
Takes the average rating value of an item to make predictions.

-  **User-Item baseline:**
Takes the bias of users and items plus the global average to make
predictions.


Example
-------

Below is a simple workflow showing how to use both the *Predictor* and
the *Learner* output. For the *Predictor* we input the prediction model
into `Predictions <http://docs.orange.biolab.si/3/visual-programming/widgets/evaluation/predictions.html>`_
widget and view the results in `Data Table <http://docs.orange.biolab.si/3/visual-programming/widgets/data/datatable.html>`_.
For *Learner* we can compare different learners in `Test&Score <http://docs.orange.biolab.si/3/visual-programming/widgets/evaluation/testlearners.html>`_ widget.

.. figure:: images/example.png
