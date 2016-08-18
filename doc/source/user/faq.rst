Frequently Asked Questions
**************************

Do I need to know to program?
=============================

Not at all. This library can be installed in Orange3 in such a way that you only
need to *drag and drop* widgets to build your pipeline.

.. figure:: ../resources/images/example_latent_factor_models.png


Why is there no widget for the ranking models?
==============================================

**Short answer:** Currently Orange3 does not support ranking.

**Long answer:** This problem is related with how Orange3 works internally. For
a given sample X, it expects to return a single value Y. The reason behind this
is related with "safety", as most of the regression and classification models
return just one single value.

In ranking problems, multiple results are returned. Therefore, Orange3 treats
the output as the output of a classification, returning the maximun value in the
sequence.


Is the library prepared for big data?
=====================================

Not really. From its very beginnings we were focused on building something easy
to use, mostly oriented towards educational purposes and research.

This doesn't mean that you cannot run big datasets. For instance, you can train
*BRISMF* with the Netflix dataset in 30-40min. But if you plan to do so, we
recommend you to use other alternatives highly optimized for those purposes.


Why are the algorithms not implemented in C/C++?
================================================

I refer back to the answer above. We want to speed-up the code as much as we
can but keeping its readability and flexibility at its maximun levels, as well
as having the less possible amount of dependecies.

Therefore, in order to achieve so, we try to cache as much accessings and
operations as we can (keeping in mind the spacial cost), and also we try to
vectorized everything we can.


Why don't you use Cython or Numba?
==================================

As it is been said before, readability and flexibility are a top priority.
*Cython* is not as simple to read as *Numpy* vectorized operations and *Numba*
can present problems with dependencies in some computers.


Can I contribute to the library?
================================

Yes, please! Indeed, if you don't want, you don't have to worry neither about
the widgets nor the documentation if you don't want. The only requirement is you
add a new model is that it passes through all the tests.

Fork and contribute all as you want!
https://github.com/biolab/orange3-recommendation