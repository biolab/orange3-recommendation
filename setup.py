#!/usr/bin/env python

import os
import sys
import pkg_resources
from setuptools import setup, find_packages
from setuptools.command.install import install

NAME = 'Orange3-Recommendation'

VERSION = '0.1.0'

DESCRIPTION = 'Orange3 Recommendation add-on.'
README_FILE = os.path.join(os.path.dirname(__file__), 'README.pypi')
LONG_DESCRIPTION = open(README_FILE).read()
AUTHOR = 'Bioinformatics Laboratory, FRI UL'
AUTHOR_EMAIL = 'contact@orange.biolab.si'
URL = "https://github.com/salvacarrion/orange3-recommendation"
DOWNLOAD_URL = "https://github.com/salvacarrion/orange3-recommendation/tarball/{}".format(VERSION)


ENTRY_POINTS = {
    'orange3.addon': (
        'recommendation = orangecontrib.recommendation',
    ),

    # Entry point used to specify packages containing tutorials accessible
    # from welcome screen. Tutorials are saved Orange Workflows (.ows files).
    'orange.widgets.tutorials': (
        # Syntax: any_text = path.to.package.containing.tutorials
        'recommendationtutorials = orangecontrib.recommendation.tutorials',
    ),

    # Entry point used to specify packages containing widgets.
    'orange.widgets': (
        # Syntax: category name = path.to.package.containing.widgets
        # Widget category specification can be seen in
        #    orangecontrib/example/widgets/__init__.py
        'Recommendation = orangecontrib.recommendation.widgets',
    ),

    # Register widget help
    "orange.canvas.help": (
        'html-index = orangecontrib.recommendation.widgets:WIDGET_HELP_PATH',),
}

KEYWORDS = [
    # [PyPi](https://pypi.python.org) packages with keyword "orange3 add-on"
    # can be installed using the Orange Add-on Manager
    'orange3-recommendation',
    'data mining',
    'orange3 add-on',
]

INSTALL_REQUIRES = sorted(set(
    line.partition('#')[0].strip()
    for line in open(os.path.join(os.path.dirname(__file__), 'requirements.txt'))
) - {''})

if 'test' in sys.argv:
    extra_setuptools_args = dict(
        test_suite='orangecontrib.recommendation.tests.coverage',
    )
else:
    extra_setuptools_args = dict()


class StoreDatasets(install):
    def run(self):
        super().run()

        old_cwd = os.getcwd()
        os.chdir(os.path.abspath(os.path.sep))

        src = pkg_resources.resource_filename('orangecontrib.recommendation',
                                              'datasets')
        dst = os.path.join(pkg_resources.resource_filename('Orange', 'datasets'),
                           'recommendation')

        try:
            os.remove(dst)
        except OSError:
            pass
        try:
            os.symlink(src, dst, target_is_directory=True)
        except OSError:
            pass
        finally:
            os.chdir(old_cwd)


if __name__ == '__main__':
    setup(
        name=NAME,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        version=VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        download_url=DOWNLOAD_URL,
        cmdclass={'install': StoreDatasets},
        packages=find_packages(),
        package_data={
            "orangecontrib.recommendation": ["datasets/*.tab",
                                             "datasets/*.csv"],
            "orangecontrib.recommendation.widgets": ["icons/*.svg"],
            "orangecontrib.recommendation.tutorials": ["*.ows"],
        },
        include_package_data=True,
        install_requires=INSTALL_REQUIRES,
        entry_points=ENTRY_POINTS,
        keywords=KEYWORDS,
        namespace_packages=['orangecontrib'],
        zip_safe=False,
        **extra_setuptools_args
    )
