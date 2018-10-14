============
Installation
============

The library uses python, which must be installed. Python 2.7, 3.4, 3.5, 3.6 and 3.7 are supported.

Install Requirements
--------------------

To install all requirements, use ::

    pip install six numpy scipy matplotlib scikit-image opencv-python imageio

This should work in both pip and anaconda.
Note that if you already have OpenCV, you might not need ``opencv-python``.
If you get any "permission denied" errors, try adding ``sudo`` in front of the command.

Install Library
---------------

Once the packages are installed, ``imgaug`` can be installed using the following
command::

    pip install git+https://github.com/aleju/imgaug

This installs the latest version directly from github.

Alternatively, you can install the latest version which was added to pypi via ::

    pip install imgaug

That version can sometimes be behind the version on github.

In rare cases, one might prefer to install from the locally cloned repository.
This is possible using ::

    git clone https://github.com/aleju/imgaug.git && cd imgaug && python setup.py install


Uninstall
---------

To deinstall the library use ::

    pip uninstall imgaug
