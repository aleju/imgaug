============
Installation
============

The library uses python, which must be installed. Most development and testing
is done in python 2.7. Python 3.x is occasionally tested and seems to also work.

The following packages must be installed:

  * numpy
  * scipy
  * scikit-image (``pip install -U scikit-image``)
  * six (``pip install -U six``)
  * OpenCV (i.e. `cv2` must be available in python). The library is mainly tested in OpenCV 2, but seems to also work in OpenCV 3.

When executing the installer, these packages will be automatically installed/upgraded
where needed. This is not the case for OpenCV, which must be installed manually.

Once the packages are available, `imgaug` can be installed using the following
command::

    pip install git+https://github.com/aleju/imgaug

This installs the latest version directly from github. If any error pops up,
try adding ``sudo`` in front (i.e. ``sudo pip install git+https://github.com/aleju/imgaug``).

Alternatively, you can install the latest version which was added to pypi via
``pip install imgaug``. That version can sometimes be behind the version on github.

In rare cases, one might prefer to install from the locally cloned repository.
This is possible using ``python setup.py sdist && sudo pip install dist/imgaug-VERSION.tar.gz``,
where `VERSION` must be replaced by the current version of the library (e.g. ``imgaug-0.2.5.tar.gz``).
The current version can be derived from `setup.py`'s content.

To deinstall the library use ``pip uninstall imgaug``.
