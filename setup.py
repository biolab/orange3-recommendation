#!/usr/bin/env python

import sys
from setuptools import setup

ENTRY_POINTS = {
    # Entry point used to specify packages containing tutorials accessible
    # from welcome screen. Tutorials are saved Orange Workflows (.ows files).
    'orange.widgets.tutorials': (
        # Syntax: any_text = path.to.package.containing.tutorials
        #'exampletutorials = orangecontrib.recsystems.tutorials',
    ),

    # Entry point used to specify packages containing widgets.
    'orange.widgets': (
        # Syntax: category name = path.to.package.containing.widgets
        # Widget category specification can be seen in
        #    orangecontrib/example/widgets/__init__.py
        #'My Category = orangecontrib.recsystems.widgets',
    ),
}

if 'test' in sys.argv:
    extra_setuptools_args = dict(
        test_suite='orangecontrib.recommendation.tests',
    )
else:
    extra_setuptools_args = dict()


if __name__ == '__main__':
    setup(
        name="Orange3-Recommendation",
        packages=['orangecontrib',
                  'orangecontrib.recommendation'],
                  # 'orangecontrib.recsystems.tutorials',
                  # 'orangecontrib.recsystems.widgets'],
        package_data={
            'orangecontrib.recommendation': ['tutorials/*.ows'],
            'orangecontrib.recommendation.widgets': ['icons/*'],
        },
        install_requires=['Orange'],
        entry_points=ENTRY_POINTS,
        namespace_packages=['orangecontrib'],
        include_package_data=True,
        zip_safe=False,
    )
