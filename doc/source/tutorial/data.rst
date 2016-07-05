The Data
========

.. index: data

This section describes how to load the data in Orange3-Recommendation.

Data Input
----------

..  index::
    single: data; input

Orange can read files in native tab-delimited format, or can load data from any of the major standard spreadsheet file type, like CSV and Excel. Native format starts with a header row with feature (column) names. Second header row gives the attribute type, which can be continuous, discrete, string or time. The third header line contains meta information to identify dependent features (class), irrelevant features (ignore) or meta features (meta). Here are the first few lines from a data set :download:`lenses.tab <code/lenses.tab>`::

   tid      user        movie       score
   string   discrete    discrete    continuous
   meta     row=1       col=1       class
   1        Breza       HarrySally  2
   2        Dana        Cvetje      5
   3        Cene        Prometheus  5
   4        Ksenija     HarrySally  4
   5        Albert      Matrix      4
   ...


**The third row is mandatory in this kind of datasets**, in order to know which attributes correspond to the users (row=1) and which ones to the items (col=1).
For the case of big datasets, users and items must be specified as a continuous attributes due to efficiency issues.

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


