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

    test_Convolve()
    test_Sharpen()
    test_Emboss()
    # TODO EdgeDetect
    # TODO DirectedEdgeDetect

    time_end = time.time()
    print("<%s> Finished without errors in %.4fs." % (__file__, time_end - time_start,))


def test_Convolve():
    reseed()

    img = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    img = np.uint8(img)

    # matrix is None
    aug = iaa.Convolve(matrix=None)
    observed = aug.augment_image(img)
    assert np.array_equal(observed, img)

    aug = iaa.Convolve(matrix=lambda _img, nb_channels, random_state: [None])
    observed = aug.augment_image(img)
    assert np.array_equal(observed, img)

    # matrix is [[1]]
    aug = iaa.Convolve(matrix=np.float32([[1]]))
    observed = aug.augment_image(img)
    assert np.array_equal(observed, img)

    aug = iaa.Convolve(matrix=lambda _img, nb_channels, random_state: np.float32([[1]]))
    observed = aug.augment_image(img)
    assert np.array_equal(observed, img)

    # matrix is [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    m = np.float32([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])
    aug = iaa.Convolve(matrix=m)
    observed = aug.augment_image(img)
    assert np.array_equal(observed, img)

    aug = iaa.Convolve(matrix=lambda _img, nb_channels, random_state: m)
    observed = aug.augment_image(img)
    assert np.array_equal(observed, img)

    # matrix is [[0, 0, 0], [0, 2, 0], [0, 0, 0]]
    m = np.float32([
        [0, 0, 0],
        [0, 2, 0],
        [0, 0, 0]
    ])
    aug = iaa.Convolve(matrix=m)
    observed = aug.augment_image(img)
    assert np.array_equal(observed, 2*img)

    aug = iaa.Convolve(matrix=lambda _img, nb_channels, random_state: m)
    observed = aug.augment_image(img)
    assert np.array_equal(observed, 2*img)

    # matrix is [[0, 0, 0], [0, 2, 0], [0, 0, 0]]
    # with 3 channels
    m = np.float32([
        [0, 0, 0],
        [0, 2, 0],
        [0, 0, 0]
    ])
    img3 = np.tile(img[..., np.newaxis], (1, 1, 3))
    aug = iaa.Convolve(matrix=m)
    observed = aug.augment_image(img3)
    assert np.array_equal(observed, 2*img3)

    aug = iaa.Convolve(matrix=lambda _img, nb_channels, random_state: m)
    observed = aug.augment_image(img3)
    assert np.array_equal(observed, 2*img3)

    # matrix is [[0, -1, 0], [0, 10, 0], [0, 0, 0]]
    m = np.float32([
        [0, -1, 0],
        [0, 10, 0],
        [0, 0, 0]
    ])
    expected = np.uint8([
        [10*1+(-1)*4, 10*2+(-1)*5, 10*3+(-1)*6],
        [10*4+(-1)*1, 10*5+(-1)*2, 10*6+(-1)*3],
        [10*7+(-1)*4, 10*8+(-1)*5, 10*9+(-1)*6]
    ])

    aug = iaa.Convolve(matrix=m)
    observed = aug.augment_image(img)
    assert np.array_equal(observed, expected)

    aug = iaa.Convolve(matrix=lambda _img, nb_channels, random_state: m)
    observed = aug.augment_image(img)
    assert np.array_equal(observed, expected)

    # changing matrices when using callable
    expected = []
    for i in sm.xrange(5):
        expected.append(img * i)

    aug = iaa.Convolve(matrix=lambda _img, nb_channels, random_state: np.float32([[random_state.randint(0, 5)]]))
    seen = [False] * 5
    for _ in sm.xrange(200):
        observed = aug.augment_image(img)
        found = False
        for i, expected_i in enumerate(expected):
            if np.array_equal(observed, expected_i):
                seen[i] = True
                found = True
                break
        assert found
        if all(seen):
            break
    assert all(seen)

    # bad datatype for matrix
    got_exception = False
    try:
        aug = iaa.Convolve(matrix=False)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # get_parameters()
    matrix = np.int32([[1]])
    aug = iaa.Convolve(matrix=matrix)
    params = aug.get_parameters()
    assert np.array_equal(params[0], matrix)
    assert params[1] == "constant"

    # TODO add test for keypoints once their handling was improved in Convolve

    ###################
    # test other dtypes
    ###################
    identity_matrix = np.int64([[1]])
    aug = iaa.Convolve(matrix=identity_matrix)

    image = np.zeros((3, 3), dtype=bool)
    image[1, 1] = True
    image_aug = aug.augment_image(image)
    assert image.dtype.type == np.bool_
    assert np.all(image_aug == image)

    for dtype in [np.uint8, np.uint16, np.int8, np.int16]:
        image = np.zeros((3, 3), dtype=dtype)
        image[1, 1] = 100
        image_aug = aug.augment_image(image)
        assert image.dtype.type == dtype
        assert np.all(image_aug == image)

    for dtype in [np.float16, np.float32, np.float64]:
        image = np.zeros((3, 3), dtype=dtype)
        image[1, 1] = 100.0
        image_aug = aug.augment_image(image)
        assert image.dtype.type == dtype
        assert np.allclose(image_aug, image)

    # ----
    # non-identity matrix
    # ----
    matrix = np.float64([
        [0, 0.6, 0],
        [0, 0.4, 0],
        [0,   0, 0]
    ])
    aug = iaa.Convolve(matrix=matrix)

    image = np.zeros((3, 3), dtype=bool)
    image[1, 1] = True
    image[2, 1] = True
    expected = np.zeros((3, 3), dtype=bool)
    expected[0, 1] = True
    expected[2, 1] = True
    image_aug = aug.augment_image(image)
    assert image.dtype.type == np.bool_
    assert np.all(image_aug == expected)

    matrix = np.float64([
        [0, 0.5, 0],
        [0, 0.5, 0],
        [0,   0, 0]
    ])
    aug = iaa.Convolve(matrix=matrix)

    # uint, int
    for dtype in [np.uint8, np.uint16, np.int8, np.int16]:
        image = np.zeros((3, 3), dtype=dtype)
        image[1, 1] = 100
        image[2, 1] = 100
        image_aug = aug.augment_image(image)

        expected = np.zeros((3, 3), dtype=dtype)
        expected[0, 1] = int(np.round(100 * 0.5))
        expected[1, 1] = int(np.round(100 * 0.5))
        expected[2, 1] = int(np.round(100 * 0.5 + 100 * 0.5))

        diff = np.abs(image_aug.astype(np.int64) - expected.astype(np.int64))
        assert image_aug.dtype.type == dtype
        assert np.max(diff) <= 2

    # float
    for dtype in [np.float16, np.float32, np.float64]:
        image = np.zeros((3, 3), dtype=dtype)
        image[1, 1] = 100.0
        image[2, 1] = 100.0
        image_aug = aug.augment_image(image)

        expected = np.zeros((3, 3), dtype=dtype)
        expected[0, 1] = 100 * 0.5
        expected[1, 1] = 100 * 0.5
        expected[2, 1] = 100 * 0.5 + 100 * 0.5

        diff = np.abs(image_aug.astype(np.float128) - expected.astype(np.float128))
        assert image_aug.dtype.type == dtype
        assert np.max(diff) < 1.0

    # ----
    # non-identity matrix, higher values
    # ----
    matrix = np.float64([
        [0, 0.5, 0],
        [0, 0.5, 0],
        [0,   0, 0]
    ])
    aug = iaa.Convolve(matrix=matrix)

    # uint, int
    for dtype in [np.uint8, np.uint16, np.int8, np.int16]:
        _min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)
        value = int(center_value + 0.4 * max_value)

        image = np.zeros((3, 3), dtype=dtype)
        image[1, 1] = value
        image[2, 1] = value
        image_aug = aug.augment_image(image)

        expected = np.zeros((3, 3), dtype=dtype)
        expected[0, 1] = int(np.round(value * 0.5))
        expected[1, 1] = int(np.round(value * 0.5))
        expected[2, 1] = int(np.round(value * 0.5 + value * 0.5))

        diff = np.abs(image_aug.astype(np.int64) - expected.astype(np.int64))
        assert image_aug.dtype.type == dtype
        assert np.max(diff) <= 2

    # float
    for dtype, value in zip([np.float16, np.float32, np.float64], [5000, 1000*1000, 1000*1000*1000]):
        image = np.zeros((3, 3), dtype=dtype)
        image[1, 1] = value
        image[2, 1] = value
        image_aug = aug.augment_image(image)

        expected = np.zeros((3, 3), dtype=dtype)
        expected[0, 1] = value * 0.5
        expected[1, 1] = value * 0.5
        expected[2, 1] = value * 0.5 + value * 0.5

        diff = np.abs(image_aug.astype(np.float128) - expected.astype(np.float128))
        assert image_aug.dtype.type == dtype
        assert np.max(diff) < 1.0

    # assert failure on invalid dtypes
    aug = iaa.Convolve(matrix=identity_matrix)
    for dt in [np.uint32, np.uint64, np.int32, np.int64]:
        got_exception = False
        try:
            _ = aug.augment_image(np.zeros((1, 1), dtype=dt))
        except Exception as exc:
            assert "forbidden dtype" in str(exc)
            got_exception = True
        assert got_exception


def test_Sharpen():
    reseed()

    def _compute_sharpened_base_img(lightness, m):
        base_img_sharpened = np.zeros((3, 3), dtype=np.float32)
        k = 1
        # note that cv2 uses reflection padding by default
        base_img_sharpened[0, 0] = (m[1, 1] + lightness)/k * 10 + 4 * (m[0, 0]/k) * 10 + 4 * (m[2, 2]/k) * 20
        base_img_sharpened[0, 2] = base_img_sharpened[0, 0]
        base_img_sharpened[2, 0] = base_img_sharpened[0, 0]
        base_img_sharpened[2, 2] = base_img_sharpened[0, 0]
        base_img_sharpened[0, 1] = (m[1, 1] + lightness)/k * 10 + 6 * (m[0, 1]/k) * 10 + 2 * (m[2, 2]/k) * 20
        base_img_sharpened[1, 0] = base_img_sharpened[0, 1]
        base_img_sharpened[1, 2] = base_img_sharpened[0, 1]
        base_img_sharpened[2, 1] = base_img_sharpened[0, 1]
        base_img_sharpened[1, 1] = (m[1, 1] + lightness)/k * 20 + 8 * (m[0, 1]/k) * 10

        base_img_sharpened = np.clip(base_img_sharpened, 0, 255).astype(np.uint8)

        return base_img_sharpened

    base_img = [[10, 10, 10],
                [10, 20, 10],
                [10, 10, 10]]
    base_img = np.uint8(base_img)
    m = np.float32([[-1, -1, -1],
                    [-1, 8, -1],
                    [-1, -1, -1]])
    m_noop = np.float32([[0, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0]])
    base_img_sharpened = _compute_sharpened_base_img(1, m)

    aug = iaa.Sharpen(alpha=0, lightness=1)
    observed = aug.augment_image(base_img)
    expected = base_img
    assert np.allclose(observed, expected)

    aug = iaa.Sharpen(alpha=1.0, lightness=1)
    observed = aug.augment_image(base_img)
    expected = base_img_sharpened
    assert np.allclose(observed, expected)

    aug = iaa.Sharpen(alpha=0.5, lightness=1)
    observed = aug.augment_image(base_img)
    expected = _compute_sharpened_base_img(0.5*1, 0.5 * m_noop + 0.5 * m)
    assert np.allclose(observed, expected.astype(np.uint8))

    aug = iaa.Sharpen(alpha=0.75, lightness=1)
    observed = aug.augment_image(base_img)
    expected = _compute_sharpened_base_img(0.75*1, 0.25 * m_noop + 0.75 * m)
    assert np.allclose(observed, expected)

    aug = iaa.Sharpen(alpha=iap.Choice([0.5, 1.0]), lightness=1)
    observed = aug.augment_image(base_img)
    expected1 = _compute_sharpened_base_img(0.5*1, m)
    expected2 = _compute_sharpened_base_img(1.0*1, m)
    assert np.allclose(observed, expected1) or np.allclose(observed, expected2)

    got_exception = False
    try:
        _ = iaa.Sharpen(alpha="test", lightness=1)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    aug = iaa.Sharpen(alpha=1.0, lightness=2)
    observed = aug.augment_image(base_img)
    expected = _compute_sharpened_base_img(1.0*2, m)
    assert np.allclose(observed, expected)

    aug = iaa.Sharpen(alpha=1.0, lightness=3)
    observed = aug.augment_image(base_img)
    expected = _compute_sharpened_base_img(1.0*3, m)
    assert np.allclose(observed, expected)

    aug = iaa.Sharpen(alpha=1.0, lightness=iap.Choice([1.0, 1.5]))
    observed = aug.augment_image(base_img)
    expected1 = _compute_sharpened_base_img(1.0*1.0, m)
    expected2 = _compute_sharpened_base_img(1.0*1.5, m)
    assert np.allclose(observed, expected1) or np.allclose(observed, expected2)

    got_exception = False
    try:
        _ = iaa.Sharpen(alpha=1.0, lightness="test")
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # this part doesnt really work so far due to nonlinearities resulting from clipping to uint8
    """
    # alpha range
    aug = iaa.Sharpen(alpha=(0.0, 1.0), lightness=1)
    base_img = np.copy(base_img)
    base_img_sharpened_min = _compute_sharpened_base_img(0.0*1, 1.0 * m_noop + 0.0 * m)
    base_img_sharpened_max = _compute_sharpened_base_img(1.0*1, 0.0 * m_noop + 1.0 * m)
    #distance_max = np.average(np.abs(base_img_sharpened.astype(np.float32) - base_img.astype(np.float32)))
    distance_max = np.average(np.abs(base_img_sharpened_max - base_img_sharpened_min))
    nb_iterations = 250
    distances = []
    for _ in sm.xrange(nb_iterations):
        observed = aug.augment_image(base_img)
        distance = np.average(np.abs(observed.astype(np.float32) - base_img_sharpened_max.astype(np.float32))) / distance_max
        distances.append(distance)

    print(distances)
    print(min(distances), np.average(distances), max(distances))
    assert 0 - 1e-4 < min(distances) < 0.1
    assert 0.4 < np.average(distances) < 0.6
    assert 0.9 < max(distances) < 1.0 + 1e-4

    nb_bins = 5
    hist, _ = np.histogram(distances, bins=nb_bins, range=(0.0, 1.0), density=False)
    density_expected = 1.0/nb_bins
    density_tolerance = 0.05
    for nb_samples in hist:
        density = nb_samples / nb_iterations
        assert density_expected - density_tolerance < density < density_expected + density_tolerance

    # lightness range
    aug = iaa.Sharpen(alpha=1.0, lightness=(0.5, 2.0))
    base_img = np.copy(base_img)
    base_img_sharpened = _compute_sharpened_base_img(1.0*2.0, m)
    distance_max = np.average(np.abs(base_img_sharpened.astype(np.int32) - base_img.astype(np.int32)))
    nb_iterations = 250
    distances = []
    for _ in sm.xrange(nb_iterations):
        observed = aug.augment_image(base_img)
        distance = np.average(np.abs(observed.astype(np.int32) - base_img.astype(np.int32))) / distance_max
        distances.append(distance)

    assert 0 - 1e-4 < min(distances) < 0.1
    assert 0.4 < np.average(distances) < 0.6
    assert 0.9 < max(distances) < 1.0 + 1e-4

    nb_bins = 5
    hist, _ = np.histogram(distances, bins=nb_bins, range=(0.0, 1.0), density=False)
    density_expected = 1.0/nb_bins
    density_tolerance = 0.05
    for nb_samples in hist:
        density = nb_samples / nb_iterations
        assert density_expected - density_tolerance < density < density_expected + density_tolerance
    """


def test_Emboss():
    reseed()

    base_img = [[10, 10, 10],
                [10, 20, 10],
                [10, 10, 15]]
    base_img = np.uint8(base_img)

    def _compute_embossed_base_img(img, alpha, strength):
        img = np.copy(img)
        base_img_embossed = np.zeros((3, 3), dtype=np.float32)

        m = np.float32([[-1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])
        strength_matrix = strength * np.float32([
            [-1, -1, 0],
            [-1, 0, 1],
            [0, 1, 1]
        ])
        ms = m + strength_matrix

        for i in range(base_img_embossed.shape[0]):
            for j in range(base_img_embossed.shape[1]):
                for u in range(ms.shape[0]):
                    for v in range(ms.shape[1]):
                        weight = ms[u, v]
                        inputs_i = abs(i + (u - (ms.shape[0]-1)//2))
                        inputs_j = abs(j + (v - (ms.shape[1]-1)//2))
                        if inputs_i >= img.shape[0]:
                            diff = inputs_i - (img.shape[0]-1)
                            inputs_i = img.shape[0] - 1 - diff
                        if inputs_j >= img.shape[1]:
                            diff = inputs_j - (img.shape[1]-1)
                            inputs_j = img.shape[1] - 1 - diff
                        inputs = img[inputs_i, inputs_j]
                        base_img_embossed[i, j] += inputs * weight

        return np.clip((1-alpha) * img + alpha * base_img_embossed, 0, 255).astype(np.uint8)

    def _allclose(a, b):
        return np.max(a.astype(np.float32) - b.astype(np.float32)) <= 2.1

    aug = iaa.Emboss(alpha=0, strength=1)
    observed = aug.augment_image(base_img)
    expected = base_img
    assert _allclose(observed, expected)

    aug = iaa.Emboss(alpha=1.0, strength=1)
    observed = aug.augment_image(base_img)
    expected = _compute_embossed_base_img(base_img, alpha=1.0, strength=1)
    assert _allclose(observed, expected)

    aug = iaa.Emboss(alpha=0.5, strength=1)
    observed = aug.augment_image(base_img)
    expected = _compute_embossed_base_img(base_img, alpha=0.5, strength=1)
    assert _allclose(observed, expected.astype(np.uint8))

    aug = iaa.Emboss(alpha=0.75, strength=1)
    observed = aug.augment_image(base_img)
    expected = _compute_embossed_base_img(base_img, alpha=0.75, strength=1)
    assert _allclose(observed, expected)

    aug = iaa.Emboss(alpha=iap.Choice([0.5, 1.0]), strength=1)
    observed = aug.augment_image(base_img)
    expected1 = _compute_embossed_base_img(base_img, alpha=0.5, strength=1)
    expected2 = _compute_embossed_base_img(base_img, alpha=1.0, strength=1)
    assert _allclose(observed, expected1) or np.allclose(observed, expected2)

    got_exception = False
    try:
        _ = iaa.Emboss(alpha="test", strength=1)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    aug = iaa.Emboss(alpha=1.0, strength=2)
    observed = aug.augment_image(base_img)
    expected = _compute_embossed_base_img(base_img, alpha=1.0, strength=2)
    assert _allclose(observed, expected)

    aug = iaa.Emboss(alpha=1.0, strength=3)
    observed = aug.augment_image(base_img)
    expected = _compute_embossed_base_img(base_img, alpha=1.0, strength=3)
    assert _allclose(observed, expected)

    aug = iaa.Emboss(alpha=1.0, strength=6)
    observed = aug.augment_image(base_img)
    expected = _compute_embossed_base_img(base_img, alpha=1.0, strength=6)
    assert _allclose(observed, expected)

    aug = iaa.Emboss(alpha=1.0, strength=iap.Choice([1.0, 1.5]))
    observed = aug.augment_image(base_img)
    expected1 = _compute_embossed_base_img(base_img, alpha=1.0, strength=1.0)
    expected2 = _compute_embossed_base_img(base_img, alpha=1.0, strength=1.5)
    assert _allclose(observed, expected1) or np.allclose(observed, expected2)

    got_exception = False
    try:
        _ = iaa.Emboss(alpha=1.0, strength="test")
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception


if __name__ == "__main__":
    main()
