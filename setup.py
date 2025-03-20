# pylint: disable=missing-module-docstring
import re

from pkg_resources import get_distribution, DistributionNotFound
from setuptools import setup, find_packages

long_description = """A library for image augmentation in machine learning experiments, particularly convolutional
neural networks. Supports the augmentation of images, keypoints/landmarks, bounding boxes, heatmaps and segmentation
maps in a variety of different ways."""

INSTALL_REQUIRES = [
    "six",
    "numpy>=1.21",
    "scipy",
    "Pillow",
    "matplotlib",
    "scikit-image>=0.17",
    "opencv-python-headless",
    "opencv-python",
    "imageio",
    "Shapely",
    "imagecorruptions-imaug>=1.1.3",
]

ALT_INSTALL_REQUIRES = {
    "opencv-python-headless": ["opencv-python", "opencv-contrib-python", "opencv-contrib-python-headless"],
}

DEV_REQUIRES = [
    "pytest-subtests",
    "xdoctest >= 0.7.2",
    "coverage",
    "pytest-cov",
    "flake8",
]


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

setup(
    name="imgaug",
    version="0.4.0", # TODO(erjel)
    author="imaug",
    author_email="ej_foss@mailbox.org",
    url="https://github.com/imaug/imaug",
    # TODO(erjel) download_url="https://github.com/aleju/imgaug/archive/0.4.0.tar.gz",
    python_requires='>3.6,<3.13', # imagecorruptions require numba -> numba 0.60 requires python <3.13
    install_requires=INSTALL_REQUIRES,
    extras_require={
        'dev': DEV_REQUIRES,
    },
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "": ["LICENSE", "README.md", "requirements.txt"],
        "imgaug": ["DejaVuSans.ttf", "quokka.jpg", "quokka_annotations.json", "quokka_depth_map_halfres.png"],
        "imgaug.checks": ["README.md"]
    },
    license="MIT",
    description="Image augmentation library for deep neural networks",
    long_description=long_description,
    keywords=["augmentation", "image", "deep learning", "neural network", "CNN", "machine learning",
              "computer vision", "overfitting"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ]
)
