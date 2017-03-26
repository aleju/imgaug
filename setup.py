from setuptools import setup, find_packages

try:
    import cv2
except ImportError as e:
    raise Exception("Could not find package 'cv2' (OpenCV). It cannot be automatically installed, so you will have to manually install it.")

long_description = """A library for image augmentation in machine learning experiments, particularly convolutional neural networks.
Supports augmentation of images and keypoints/landmarks in a variety of different ways."""

setup(
    name="imgaug",
    version="0.2.0",
    author="Alexander Jung",
    author_email="kontakt@ajung.name",
    url="https://github.com/aleju/imgaug",
    download_url="https://github.com/aleju/imgaug/archive/0.2.0.tar.gz",
    install_requires=["scipy", "scikit-image>=0.11.0", "numpy", "six"],
    packages=find_packages(),
    license="MIT",
    description="Image augmentation library for machine learning",
    long_description=long_description,
    keywords=["augmentation", "image", "deep learning", "neural network", "machine learning"]
)
