"""
Automatically run tests for this library.
Run tests via
    python run_all.py

or alternatively:
    python -m pytest ./ --verbose --doctest-modules --ignore="test/run_all.py" -s

or use
    python -m pytest ./test/augmenters/test_size.py --verbose --doctest-modules --ignore="test/run_all.py" -s
to run the tests in a specific file.

or use
    python -m pytest ./test/augmenters/test_size.py::test_Crop --verbose --doctest-modules --ignore="test/run_all.py" -s
to run only the test function "test_Crop" in test_size.py.

"""
from __future__ import print_function, division, absolute_import

import time

import test_imgaug
import test_parameters
from augmenters import test_arithmetic
from augmenters import test_blend
from augmenters import test_blur
from augmenters import test_color
from augmenters import test_contrast
from augmenters import test_convolutional
from augmenters import test_flip
from augmenters import test_geometric
from augmenters import test_meta
from augmenters import test_mixed_files
from augmenters import test_segmentation
from augmenters import test_size
from augmenters import test_weather


def main():
    time_start = time.time()

    test_imgaug.main()
    test_parameters.main()
    test_arithmetic.main()
    test_blend.main()
    test_blur.main()
    test_color.main()
    test_contrast.main()
    test_convolutional.main()
    test_flip.main()
    test_geometric.main()
    test_meta.main()
    test_mixed_files.main()
    test_segmentation.main()
    test_size.main()
    test_weather.main()

    time_end = time.time()
    print("Finished all tests without errors in %.4fs." % (time_end - time_start,))


if __name__ == "__main__":
    main()
