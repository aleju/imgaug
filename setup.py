from setuptools import setup, find_packages

long_description = """A library for image augmentation in machine learning experiments, particularly convolutional
neural networks. Supports the augmentation of images, keypoints/landmarks, bounding boxes, heatmaps and segmentation
maps in a variety of different ways."""

setup(
    name="imgaug",
    version="0.2.8",
    author="Alexander Jung",
    author_email="kontakt@ajung.name",
    url="https://github.com/aleju/imgaug",
    download_url="https://github.com/aleju/imgaug/archive/0.2.8.tar.gz",
    install_requires=["scipy", "scikit-image>=0.11.0", "numpy>=1.15.0", "six", "imageio", "Pillow", "matplotlib",
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
