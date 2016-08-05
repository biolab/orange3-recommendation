########
Tutorial
########


==========
Input data
==========

.. index: data

This section describes how to load the data in Orange3-Recommendation.

Data format
-----------

..  index::
    single: data; input

Orange can read files in native tab-delimited format, or can load data from any
of the major standard spreadsheet file type, like CSV and Excel. Native format
starts with a header row with feature (column) names. Second header row gives
the attribute type, which can be continuous, discrete, string or time. The third
header line contains meta information to identify dependent features (class),
irrelevant features (ignore) or meta features (meta). Here are the first few
lines from a data set :download:`ratings.tab <code/ratings.tab>`::

    tid      user        movie       score
    string   discrete    discrete    continuous
    meta     row=1       col=1       class
    1        Breza       HarrySally  2
    2        Dana        Cvetje      5
    3        Cene        Prometheus  5
    4        Ksenija     HarrySally  4
    5        Albert      Matrix      4
    ...


**The third row is mandatory in this kind of datasets**, in order to know which
attributes correspond to the users (row=1) and which ones to the items (col=1).
For the case of big datasets, users and items must be specified as a continuous
attributes due to efficiency issues. Here are the first few lines from a data
set :download:`MovieLens100K.tab <code/movielens100k.tab>`::

    user            movie         score         tid
    continuous      continuous    continuous    time
    row=1           col=1         class         meta
    196             242           3             881250949
    186             302           3             891717742
    22              377           1             878887116
    244             51            2             880606923
    166             346           1             886397596
    298             474           4             884182806
    ...


Loading data
------------

Datasets can be loaded as follow::

    import Orange
    data = Orange.data.Table("ratings.tab")

In the add-on, several toy datasets are included: *ratings.tab,
movielens100k.tab, binary_data.tab, epinions_train.tab, epinions_test.tab,...*
and a few more.


===============
Getting started
===============


Rating pairs (user, item)
-------------------------

Let's presume that we want to load a dataset, train it and predict its first three pairs of (id_user, id_item)

.. code-block:: python
   :linenos:

    import Orange
    from orangecontrib.recommendation import BRISMFLearner

    # Load data and train the model
    data = Orange.data.Table('movielens100k.tab')
    learner = BRISMFLearner(num_factors=15, num_iter=25, learning_rate=0.07, lmbda=0.1)
    recommender = learner(data)

    # Make predictions
    prediction = recommender(data[:3])
    print(prediction)
    >>>
    [ 3.79505151  3.75096513  1.293013 ]


The first three lines of code, import the Orange module, the BRISMF factorization model
and loads the MovieLens100K dataset. In the next lines we instantiate the model
(*learner = BRISMFLearner(...)*) and we fit the model with the loaded data.

Finally, we predict the ratings for the first three pairs (user, item) in the loaded dataset.

Recommend items for set of users
--------------------------------

Now we want to get all the predictions (all items) for a set of users:

.. code-block:: python
   :linenos:

   import numpy as np
   indices_users = np.array([4, 12, 36])
   prediction = recommender.predict_items(indices_users)
   print(prediction)
   >>>
   [[ 1.34743879  4.61513578  3.90757263 ...,  3.03535099  4.08221699 4.26139511]
    [ 1.16652757  4.5516808   3.9867497  ...,  2.94690548  3.67274108 4.1868596 ]
    [ 2.74395768  4.04859096  4.04553826 ...,  3.22923456  3.69682699 4.95043435]]

This time, we've fill an array with the indices of the users to which make the predictions
for all the items.

If we want as an output just the first *k* elements (do not confuse with *top best* items),
we have to add the parameter *top=INTEGER* to the function

.. code-block:: python

   prediction = recommender.predict_items(indices_users, top=2)
   print(prediction)
   >>>
   [[ 1.34743879  4.61513578]
    [ 1.16652757  4.5516808]
    [ 2.74395768  4.04859096]]

Evaluation
----------

Finally, we want to known which of a list of recommender performs better on our dataset. Therefore,
we perform cross-validation over a list of learners.

The first thing we need to do is to make a list of all the learners that we want to cross-validate.

.. code-block:: python

    from orangecontrib.recommendation import GlobalAvgLearner,
                                                 ItemAvgLearner,
                                                 UserAvgLearner,
                                                 UserItemBaselineLearner
    global_avg = GlobalAvgLearner()
    items_avg = ItemAvgLearner()
    users_avg = UserAvgLearner()
    useritem_baseline = UserItemBaselineLearner()
    brismf = BRISMFLearner(num_factors=15, num_iter=25, learning_rate=0.07, lmbda=0.1)
    learners = [global_avg, items_avg, users_avg, useritem_baseline, brismf]


Once, we have the list of learners and the data loaded, we score the methods.
For the case, we have scored the recommendation two measures for goodnes of fit,
which they're later printed. To measure the error of the scoring, you can use
all the functions defined in ``Orange.evaluation``.

.. code-block:: python

    res = Orange.evaluation.CrossValidation(data, learners, k=5)
    rmse = Orange.evaluation.RMSE(res)
    r2 = Orange.evaluation.R2(res)

    print("Learner  RMSE  R2")
    for i in range(len(learners)):
        print("{:8s} {:.2f} {:5.2f}".format(learners[i].name, rmse[i], r2[i]))
    >>>
    Learner                   RMSE  R2
      - Global average        1.13 -0.00
      - Item average          1.03  0.16
      - User average          1.04  0.14
      - User-Item Baseline    0.98  0.25
      - BRISMF                0.96  0.28