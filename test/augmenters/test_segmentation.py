from __future__ import print_function, division, absolute_import

import time

import matplotlib
matplotlib.use('Agg')  # fix execution of tests involving matplotlib on travis
import numpy as np
import six.moves as sm

from imgaug import augmenters as iaa
from imgaug import parameters as iap
from imgaug import dtypes as iadt
from imgaug.testutils import reseed


def main():
    time_start = time.time()

    test_Superpixels()

    time_end = time.time()
    print("<%s> Finished without errors in %.4fs." % (__file__, time_end - time_start,))


def test_Superpixels():
    reseed()

    def _array_equals_tolerant(a, b, tolerance):
        diff = np.abs(a.astype(np.int32) - b.astype(np.int32))
        return np.all(diff <= tolerance)

    base_img = [
        [255, 255, 255, 0, 0, 0],
        [255, 235, 255, 0, 20, 0],
        [250, 250, 250, 5, 5, 5]
    ]
    base_img = np.tile(np.array(base_img, dtype=np.uint8)[..., np.newaxis], (1, 1, 3))

    base_img_superpixels = [
        [251, 251, 251, 4, 4, 4],
        [251, 251, 251, 4, 4, 4],
        [251, 251, 251, 4, 4, 4]
    ]
    base_img_superpixels = np.tile(np.array(base_img_superpixels, dtype=np.uint8)[..., np.newaxis], (1, 1, 3))

    base_img_superpixels_left = np.copy(base_img_superpixels)
    base_img_superpixels_left[:, 3:, :] = base_img[:, 3:, :]

    base_img_superpixels_right = np.copy(base_img_superpixels)
    base_img_superpixels_right[:, :3, :] = base_img[:, :3, :]

    aug = iaa.Superpixels(p_replace=0, n_segments=2)
    observed = aug.augment_image(base_img)
    expected = base_img
    assert np.allclose(observed, expected)

    aug = iaa.Superpixels(p_replace=1.0, n_segments=2)
    observed = aug.augment_image(base_img)
    expected = base_img_superpixels
    assert _array_equals_tolerant(observed, expected, 2)

    aug = iaa.Superpixels(p_replace=1.0, n_segments=iap.Deterministic(2))
    observed = aug.augment_image(base_img)
    expected = base_img_superpixels
    assert _array_equals_tolerant(observed, expected, 2)

    aug = iaa.Superpixels(p_replace=iap.Binomial(iap.Choice([0.0, 1.0])), n_segments=2)
    observed = aug.augment_image(base_img)
    assert np.allclose(observed, base_img) or _array_equals_tolerant(observed, base_img_superpixels, 2)

    aug = iaa.Superpixels(p_replace=0.5, n_segments=2)
    seen = {"none": False, "left": False, "right": False, "both": False}
    for _ in sm.xrange(100):
        observed = aug.augment_image(base_img)
        if _array_equals_tolerant(observed, base_img, 2):
            seen["none"] = True
        elif _array_equals_tolerant(observed, base_img_superpixels_left, 2):
            seen["left"] = True
        elif _array_equals_tolerant(observed, base_img_superpixels_right, 2):
            seen["right"] = True
        elif _array_equals_tolerant(observed, base_img_superpixels, 2):
            seen["both"] = True
        else:
            raise Exception("Generated superpixels image does not match any expected image.")
        if all(seen.values()):
            break
    assert all(seen.values())

    # test exceptions for wrong parameter types
    got_exception = False
    try:
        _ = iaa.Superpixels(p_replace="test", n_segments=100)
    except Exception:
        got_exception = True
    assert got_exception

    got_exception = False
    try:
        _ = iaa.Superpixels(p_replace=1, n_segments="test")
    except Exception:
        got_exception = True
    assert got_exception

    # test get_parameters()
    aug = iaa.Superpixels(p_replace=0.5, n_segments=2, max_size=100, interpolation="nearest")
    params = aug.get_parameters()
    assert isinstance(params[0], iap.Binomial)
    assert isinstance(params[0].p, iap.Deterministic)
    assert isinstance(params[1], iap.Deterministic)
    assert 0.5 - 1e-4 < params[0].p.value < 0.5 + 1e-4
    assert params[1].value == 2
    assert params[2] == 100
    assert params[3] == "nearest"

    ###################
    # test other dtypes
    ###################
    # bool
    aug = iaa.Superpixels(p_replace=1.0, n_segments=2)
    img = np.array([
        [False, False, True, True],
        [False, False, True, True]
    ], dtype=bool)
    img_aug = aug.augment_image(img)
    assert img_aug.dtype == img.dtype
    assert np.all(img_aug == img)

    aug = iaa.Superpixels(p_replace=1.0, n_segments=1)
    img = np.array([
        [True, True, True, True],
        [False, True, True, True]
    ], dtype=bool)
    img_aug = aug.augment_image(img)
    assert img_aug.dtype == img.dtype
    assert np.all(img_aug)

    for dtype in [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32]:
        min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)

        if np.dtype(dtype).kind == "i":
            values = [int(center_value), int(0.1 * max_value), int(0.2 * max_value),
                      int(0.5 * max_value), max_value-100]
            values = [((-1)*value, value) for value in values]
        else:
            values = [(0, int(center_value)),
                      (10, int(0.1 * max_value)), (10, int(0.2 * max_value)), (10, int(0.5 * max_value)),
                      (0, max_value), (int(center_value), max_value)]

        for v1, v2 in values:
            aug = iaa.Superpixels(p_replace=1.0, n_segments=2)
            img = np.array([
                [v1, v1, v2, v2],
                [v1, v1, v2, v2]
            ], dtype=dtype)
            img_aug = aug.augment_image(img)
            assert img_aug.dtype == np.dtype(dtype)
            assert np.array_equal(img_aug, img)

            aug = iaa.Superpixels(p_replace=1.0, n_segments=1)
            img = np.array([
                [v2, v2, v2, v2],
                [v1, v2, v2, v2]
            ], dtype=dtype)
            img_aug = aug.augment_image(img)
            assert img_aug.dtype == np.dtype(dtype)
            assert np.all(img_aug == int(np.round((7/8)*v2 + (1/8)*v1)))

    for dtype in []:
        def _allclose(a, b):
            atol = 1e-4 if dtype == np.float16 else 1e-8
            return np.allclose(a, b, atol=atol, rtol=0)

        isize = np.dtype(dtype).itemsize
        for value in [0, 1.0, 10.0, 1000 ** (isize - 1)]:
            v1 = (-1) * value
            v2 = value

            aug = iaa.Superpixels(p_replace=1.0, n_segments=2)
            img = np.array([
                [v1, v1, v2, v2],
                [v1, v1, v2, v2]
            ], dtype=dtype)
            img_aug = aug.augment_image(img)
            assert img_aug.dtype == np.dtype(dtype)
            assert _allclose(img_aug, img)

            aug = iaa.Superpixels(p_replace=1.0, n_segments=1)
            img = np.array([
                [v2, v2, v2, v2],
                [v1, v2, v2, v2]
            ], dtype=dtype)
            img_aug = aug.augment_image(img)
            assert img_aug.dtype == np.dtype(dtype)
            assert _allclose(img_aug, (7/8)*v2 + (1/8)*v1)


if __name__ == "__main__":
    main()
