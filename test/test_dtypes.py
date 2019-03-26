from __future__ import print_function, division, absolute_import

import time

import matplotlib
matplotlib.use('Agg')  # fix execution of tests involving matplotlib on travis
import numpy as np

from imgaug import dtypes as iadt


def main():
    time_start = time.time()

    test_copy_dtypes_for_restore()

    time_end = time.time()
    print("<%s> Finished without errors in %.4fs." % (__file__, time_end - time_start,))


# TODO extend tests towards all dtypes and actual minima/maxima of value ranges
def test_clip():
    arr = np.int32([0, 1, 2, 3, 4, 5])
    observed = iadt.clip_(arr, 0, 10)
    assert np.array_equal(observed, np.int32([0, 1, 2, 3, 4, 5]))

    arr = np.int32([0, 1, 2, 3, 4, 5])
    observed = iadt.clip_(arr, 0, 5)
    assert np.array_equal(observed, np.int32([0, 1, 2, 3, 4, 5]))

    arr = np.int32([0, 1, 2, 3, 4, 5])
    observed = iadt.clip_(arr, 0, 4)
    assert np.array_equal(observed, np.int32([0, 1, 2, 3, 4, 4]))

    arr = np.float32([-1.0])
    observed = iadt.clip_(arr, 0, 1)
    assert np.allclose(observed, np.float32([0.0]))

    arr = np.float32([-1.0])
    observed = iadt.clip_(arr, -1.0, 1)
    assert np.allclose(observed, np.float32([-1.0]))

    arr = np.uint32([0])
    observed = iadt.clip_(arr, 0, 1)
    assert np.array_equal(observed, np.uint32([0]))

    arr = np.uint32([1])
    observed = iadt.clip_(arr, 0, 1)
    assert np.array_equal(observed, np.uint32([1]))

    arr = np.uint32([2])
    observed = iadt.clip_(arr, 0, 1)
    assert np.array_equal(observed, np.uint32([1]))

    arr = np.uint32([1])
    observed = iadt.clip_(arr, -1, 1)
    assert np.array_equal(observed, np.uint32([1]))

    arr = np.uint32([10])
    observed = iadt.clip_(arr, -1, 1)
    assert np.array_equal(observed, np.uint32([1]))

    arr = np.int8([127])
    observed = iadt.clip_(arr, 0, 127)
    assert np.array_equal(observed, np.int8([127]))

    arr = np.int8([127])
    observed = iadt.clip_(arr, 0, 128)
    assert np.array_equal(observed, np.int8([127]))

    arr = np.int8([127])
    observed = iadt.clip_(arr, -1, 127)
    assert np.array_equal(observed, np.int8([127]))

    arr = np.int8([1])
    observed = iadt.clip_(arr, None, None)
    assert np.array_equal(observed, np.int8([1]))

    arr = np.int8([1])
    observed = iadt.clip_(arr, None, 10)
    assert np.array_equal(observed, np.int8([1]))

    arr = np.int8([1])
    observed = iadt.clip_(arr, -10, None)
    assert np.array_equal(observed, np.int8([1]))

    arr = np.int8([10])
    observed = iadt.clip_(arr, None, 1)
    assert np.array_equal(observed, np.int8([1]))

    arr = np.int8([-10])
    observed = iadt.clip_(arr, -1, None)
    assert np.array_equal(observed, np.int8([-1]))

    # single value arrays, shape == tuple()
    arr = np.int8(-10)
    observed = iadt.clip_(arr, -10, 10)
    assert np.array_equal(observed, np.int8(-10))

    arr = np.int8(-10)
    observed = iadt.clip_(arr, -1, 10)
    assert np.array_equal(observed, np.int8(-1))

    arr = np.int8(10)
    observed = iadt.clip_(arr, -10, 1)
    assert np.array_equal(observed, np.int8(1))


def test_copy_dtypes_for_restore():
    # TODO using dtype=np.bool is causing this to fail as it ends up being <type bool> instead of
    # <type 'numpy.bool_'>. Any problems from that for the library?
    images = [
        np.zeros((1, 1, 3), dtype=np.uint8),
        np.zeros((10, 16, 3), dtype=np.float32),
        np.zeros((20, 10, 6), dtype=np.int32)
    ]

    dtypes_copy = iadt.copy_dtypes_for_restore(images, force_list=False)
    assert all([dtype_i.type == dtype_j for dtype_i, dtype_j in zip(dtypes_copy, [np.uint8, np.float32, np.int32])])

    dts = [np.uint8, np.float32, np.int32]
    for dt in dts:
        images = np.zeros((10, 16, 32, 3), dtype=dt)
        dtypes_copy = iadt.copy_dtypes_for_restore(images)
        assert isinstance(dtypes_copy, np.dtype)
        assert dtypes_copy.type == dt

        dtypes_copy = iadt.copy_dtypes_for_restore(images, force_list=True)
        assert isinstance(dtypes_copy, list)
        assert all([dtype_i.type == dt for dtype_i in dtypes_copy])


# TODO remove these tests once a similar test for restore_dtypes_() was added
"""
def test_restore_augmented_image_dtype_():
    image = np.zeros((16, 32, 3), dtype=np.uint8)
    image_result = iaa.restore_augmented_image_dtype_(image, np.int32)
    assert image_result.dtype.type == np.int32


def test_restore_augmented_image_dtype():
    image = np.zeros((16, 32, 3), dtype=np.uint8)
    image_result = iaa.restore_augmented_image_dtype(image, np.int32)
    assert image_result.dtype.type == np.int32


def test_restore_augmented_images_dtypes_():
    images = np.zeros((10, 16, 32, 3), dtype=np.int32)
    dtypes = iaa.copy_dtypes_for_restore(images)
    images = images.astype(np.uint8)
    assert images.dtype.type == np.uint8
    images_result = iaa.restore_augmented_images_dtypes_(images, dtypes)
    assert images_result.dtype.type == np.int32

    images = [np.zeros((16, 32, 3), dtype=np.int32) for _ in sm.xrange(10)]
    dtypes = iaa.copy_dtypes_for_restore(images)
    images = [image.astype(np.uint8) for image in images]
    assert all([image.dtype.type == np.uint8 for image in images])
    images_result = iaa.restore_augmented_images_dtypes_(images, dtypes)
    assert all([image_result.dtype.type == np.int32 for image_result in images_result])


def test_restore_augmented_images_dtypes():
    images = np.zeros((10, 16, 32, 3), dtype=np.int32)
    dtypes = iaa.copy_dtypes_for_restore(images)
    images = images.astype(np.uint8)
    assert images.dtype.type == np.uint8
    images_restored = iaa.restore_augmented_images_dtypes(images, dtypes)
    assert images_restored.dtype.type == np.int32

    images = [np.zeros((16, 32, 3), dtype=np.int32) for _ in sm.xrange(10)]
    dtypes = iaa.copy_dtypes_for_restore(images)
    images = [image.astype(np.uint8) for image in images]
    assert all([image.dtype.type == np.uint8 for image in images])
    images_restored = iaa.restore_augmented_images_dtypes(images, dtypes)
    assert all([image_restored.dtype.type == np.int32 for image_restored in images_restored])
"""


if __name__ == "__main__":
    main()
