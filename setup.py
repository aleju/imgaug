from setuptools import setup, find_packages
import os

# Check if OpenCV is installed and raise an error if it is not
# but don't do this if the ReadTheDocs systems tries to install
# the library, as that is configured to mock cv2 anyways
READ_THE_DOCS = (os.environ.get("READTHEDOCS", "False").lower()
                 in ["true", "1", "on", "yes"])
NO_CV2_INSTALLED_CHECK = (os.environ.get("IMGAUG_NO_CV2_INSTALLED_CHECK", "False").lower()
                          in ["true", "1", "on", "yes"])
if not READ_THE_DOCS and not NO_CV2_INSTALLED_CHECK:
    try:
        import cv2  # pylint: disable=locally-disabled, unused-import, line-too-long
    except ImportError as e:
        raise Exception(
            "Could not find package 'cv2' (OpenCV). Please install it manually, e.g. via: pip install opencv-python"
        )

long_description = """A library for image augmentation in machine learning experiments, particularly convolutional
neural networks. Supports the augmentation of images, keypoints/landmarks, bounding boxes, heatmaps and segmentation
maps in a variety of different ways."""

setup(
    name="imgaug",
    version="0.2.7",
    author="Alexander Jung",
    author_email="kontakt@ajung.name",
    url="https://github.com/aleju/imgaug",
    download_url="https://github.com/aleju/imgaug/archive/0.2.7.tar.gz",
    install_requires=["scipy", "scikit-image>=0.11.0", "numpy>=1.7.0", "six", "imageio", "Pillow", "matplotlib",
                      "Shapely", "opencv-python"],
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
