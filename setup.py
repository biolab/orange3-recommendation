#!/usr/bin/env python

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

if __name__ == '__main__':
    setup(
        name="Orange3 Recommender Systems",
        packages=['orangecontrib',
                  'orangecontrib.recsystems'],
                  # 'orangecontrib.recsystems.tutorials',
                  # 'orangecontrib.recsystems.widgets'],
        package_data={
            'orangecontrib.recsystems': ['tutorials/*.ows'],
            'orangecontrib.recsystems.widgets': ['icons/*'],
        },
        install_requires=['Orange'],
        entry_points=ENTRY_POINTS,
        namespace_packages=['orangecontrib'],
        include_package_data=True,
        zip_safe=False,
    )
