.. -*- mode: rst -*-

=============
RecommenderSystems
=============


*RecommenderSystems* is a Python module for data fusion based on recent collective latent
factor models.

Dependencies
============

RecommenderSystems is tested to work under Python 3.

The required dependencies to build the software are Numpy >= 1.7, SciPy >= 0.12

Install
=======

This package uses distutils, which is the default way of installing
python modules. To install in your home directory, use::

    python setup.py install --user

To install for all users on Unix/Linux::

    python setup.py build
    sudo python setup.py install

For development mode use::

    python setup.py develop

Usage
=====

Let's import a dataset of ratings::

     >>> import Orange
     >>> data = Orange.data.Table("orange movie ratings.tab")

Next, we fit our model::

     >>> learner = Orange.recommenders.model based.BRISMFLearner()
     >>> recommender = learner(data)

and finally we make recommendations for a specific user::

     >>> prediction = recommender.predict(user=1, sort=False, top=None)

****

scikit-fusion is distributed with a few working data fusion scenarios::

    >>> from skfusion import datasets
    >>> dicty = datasets.load_dicty()
    >>> print(dicty)
    FusionGraph(Object types: 3, Relations: 3)
    >>> print(dicty.object_types)
    {ObjectType(GO term), ObjectType(Experimental condition), ObjectType(Gene)}
    >>> print(dicty.relations)
    {Relation(ObjectType(Gene), ObjectType(GO term)),
     Relation(ObjectType(Gene), ObjectType(Gene)),
     Relation(ObjectType(Gene), ObjectType(Experimental condition))}

Relevant links
==============

- Official source code repo: https://github.com/salvacarrion/orange3-recommendersystems
- HTML documentation: TBA
- Download releases: https://github.com/salvacarrion/orange3-recommendersystems/releases
- Issue tracker: https://github.com/salvacarrion/orange3-recommendersystems/issues

****

- Matrix factorization-based (BRISMF): http://www.netflixprize.com/assets/GrandPrize2009_BPC_BellKor.pdf
- CLiMF: http://dl.acm.org/citation.cfm?id=2365981