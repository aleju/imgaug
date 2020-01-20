import re

from pkg_resources import get_distribution, DistributionNotFound
from setuptools import setup, find_packages

long_description = """A library for image augmentation in machine learning experiments, particularly convolutional
neural networks. Supports the augmentation of images, keypoints/landmarks, bounding boxes, heatmaps and segmentation
maps in a variety of different ways."""

INSTALL_REQUIRES = [
    "six",
    "numpy>=1.15",
    "scipy",
    "Pillow",
    "matplotlib",
    "scikit-image>=0.14.2",
    "opencv-python-headless",
    "imageio",
    "Shapely",
]

ALT_INSTALL_REQUIRES = {"opencv-python-headless": "opencv-python"}


def check_alternative_installation(install_require, alternative_install_require):
    """If some version version of alternative requirement installed, return alternative,
    else return main.
    """
    try:
        alternative_pkg_name = re.split(r"[!<>=]", alternative_install_require)[0]
        get_distribution(alternative_pkg_name)
    except DistributionNotFound:
        return install_require

    return str(alternative_install_require)


def get_install_requirements(main_requires, alternative_requires):
    """Iterates over all install requires
    If an install require has an alternative option, check if this option is installed
    If that is the case, replace the install require by the alternative to not install dual package"""
    new_install_requires = []
    for main_require in main_requires:
        if main_require in alternative_requires.keys():
            main_require = check_alternative_installation(main_require, alternative_requires.get(main_require))
        new_install_requires.append(main_require)

    return new_install_requires


INSTALL_REQUIRES = get_install_requirements(INSTALL_REQUIRES, ALT_INSTALL_REQUIRES)

setup(
    name="imgaug",
    version="0.3.0",
    author="Alexander Jung",
    author_email="kontakt@ajung.name",
    url="https://github.com/aleju/imgaug",
    download_url="https://github.com/aleju/imgaug/archive/0.3.0.tar.gz",
    install_requires=INSTALL_REQUIRES,
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
              "computer vision", "overfitting"]
)
