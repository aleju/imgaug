# pylint: disable=missing-module-docstring
import re

from pkg_resources import get_distribution, DistributionNotFound
from setuptools import setup, find_packages

INSTALL_REQUIRES = [
    "six",
    "numpy>=1.15",
    "scipy",
    "Pillow",
    "matplotlib",
    "scikit-image>=0.14.2",
    "opencv-python-headless",
    "imageio<=2.6.1; python_version<'3.5'",
    "imageio; python_version>='3.5'",
    "Shapely"
]

ALT_INSTALL_REQUIRES = {
    "opencv-python-headless": ["opencv-python", "opencv-contrib-python", "opencv-contrib-python-headless"],
}


def check_alternative_installation(install_require, alternative_install_requires):
    """If some version version of alternative requirement installed, return alternative,
    else return main.
    """
    for alternative_install_require in alternative_install_requires:
        try:
            alternative_pkg_name = re.split(r"[!<>=]", alternative_install_require)[0]
            get_distribution(alternative_pkg_name)
            return str(alternative_install_require)
        except DistributionNotFound:
            continue

    return str(install_require)


def get_install_requirements(main_requires, alternative_requires):
    """Iterates over all install requires
    If an install require has an alternative option, check if this option is installed
    If that is the case, replace the install require by the alternative to not install dual package"""
    install_requires = []
    for main_require in main_requires:
        if main_require in alternative_requires:
            main_require = check_alternative_installation(main_require, alternative_requires.get(main_require))
        install_requires.append(main_require)

    return install_requires


INSTALL_REQUIRES = get_install_requirements(INSTALL_REQUIRES, ALT_INSTALL_REQUIRES)

setup(install_requires=INSTALL_REQUIRES)
