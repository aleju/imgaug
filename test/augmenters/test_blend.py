from __future__ import print_function, division, absolute_import

import time

import matplotlib
matplotlib.use('Agg')  # fix execution of tests involving matplotlib on travis
import numpy as np
import six.moves as sm

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from imgaug import dtypes as iadt
from imgaug.augmenters import blend
from imgaug.testutils import keypoints_equal, reseed
from imgaug.augmentables.heatmaps import HeatmapsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


def main():
    time_start = time.time()

    test_blend_alpha()
    test_Alpha()
    test_AlphaElementwise()
    # TODO SimplexNoiseAlpha
    # TODO FrequencyNoiseAlpha

    time_end = time.time()
    print("<%s> Finished without errors in %.4fs." % (__file__, time_end - time_start,))


def test_blend_alpha():
    img_fg = np.full((3, 3, 1), 0, dtype=bool)
    img_bg = np.full((3, 3, 1), 1, dtype=bool)
    img_blend = blend.blend_alpha(img_fg, img_bg, 1.0, eps=0)
    assert img_blend.dtype.name == np.dtype(np.bool_)
    assert img_blend.shape == (3, 3, 1)
    assert np.all(img_blend == 0)

    img_fg = np.full((3, 3, 1), 0, dtype=bool)
    img_bg = np.full((3, 3, 1), 1, dtype=bool)
    img_blend = blend.blend_alpha(img_fg, img_bg, 0.0, eps=0)
    assert img_blend.dtype.name == np.dtype(np.bool_)
    assert img_blend.shape == (3, 3, 1)
    assert np.all(img_blend == 1)

    img_fg = np.full((3, 3, 1), 0, dtype=bool)
    img_bg = np.full((3, 3, 1), 1, dtype=bool)
    img_blend = blend.blend_alpha(img_fg, img_bg, 0.3, eps=0)
    assert img_blend.dtype.name == np.dtype(np.bool_)
    assert img_blend.shape == (3, 3, 1)
    assert np.all(img_blend == 1)

    img_fg = np.full((3, 3, 2), 0, dtype=bool)
    img_bg = np.full((3, 3, 2), 1, dtype=bool)
    img_blend = blend.blend_alpha(img_fg, img_bg, [1.0, 0.0], eps=0)
    assert img_blend.dtype.name == np.dtype(np.bool_)
    assert img_blend.shape == (3, 3, 2)
    assert np.all(img_blend[:, :, 0] == 0)
    assert np.all(img_blend[:, :, 1] == 1)

    for dtype in [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64]:
        min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)
        values = [
            (0, 0),
            (0, 10),
            (10, 20),
            (min_value, min_value),
            (max_value, max_value),
            (min_value, max_value),
            (min_value, int(center_value)),
            (int(center_value), max_value),
            (int(center_value + 0.20 * max_value), max_value),
            (int(center_value + 0.27 * max_value), max_value),
            (int(center_value + 0.40 * max_value), max_value),
            (min_value, 0),
            (0, max_value)
        ]
        values = values + [(v2, v1) for v1, v2 in values]

        for v1, v2 in values:
            img_fg = np.full((3, 3, 1), v1, dtype=dtype)
            img_bg = np.full((3, 3, 1), v2, dtype=dtype)
            img_blend = blend.blend_alpha(img_fg, img_bg, 1.0, eps=0)
            assert img_blend.dtype.name == np.dtype(dtype)
            assert img_blend.shape == (3, 3, 1)
            assert np.all(img_blend == dtype(v1))

            img_fg = np.full((3, 3, 1), v1, dtype=dtype)
            img_bg = np.full((3, 3, 1), v2, dtype=dtype)
            img_blend = blend.blend_alpha(img_fg, img_bg, 0.99, eps=0.1)
            assert img_blend.dtype.name == np.dtype(dtype)
            assert img_blend.shape == (3, 3, 1)
            assert np.all(img_blend == dtype(v1))

            img_fg = np.full((3, 3, 1), v1, dtype=dtype)
            img_bg = np.full((3, 3, 1), v2, dtype=dtype)
            img_blend = blend.blend_alpha(img_fg, img_bg, 0.0, eps=0)
            assert img_blend.dtype.name == np.dtype(dtype)
            assert img_blend.shape == (3, 3, 1)
            assert np.all(img_blend == dtype(v2))

            # TODO this test breaks for numpy <1.15 -- why?
            for c in sm.xrange(3):
                img_fg = np.full((3, 3, c), v1, dtype=dtype)
                img_bg = np.full((3, 3, c), v2, dtype=dtype)
                img_blend = blend.blend_alpha(img_fg, img_bg, 0.75, eps=0)
                assert img_blend.dtype.name == np.dtype(dtype)
                assert img_blend.shape == (3, 3, c)
                for ci in sm.xrange(c):
                    v_blend = min(max(int(0.75*np.float128(v1) + 0.25*np.float128(v2)), min_value), max_value)
                    diff = v_blend - img_blend if v_blend > img_blend[0, 0, ci] else img_blend - v_blend
                    assert np.all(diff < 1.01)

            img_fg = np.full((3, 3, 2), v1, dtype=dtype)
            img_bg = np.full((3, 3, 2), v2, dtype=dtype)
            img_blend = blend.blend_alpha(img_fg, img_bg, 0.75, eps=0)
            assert img_blend.dtype.name == np.dtype(dtype)
            assert img_blend.shape == (3, 3, 2)
            v_blend = min(max(int(0.75 * np.float128(v1) + 0.25 * np.float128(v2)), min_value), max_value)
            diff = v_blend - img_blend if v_blend > img_blend[0, 0, 0] else img_blend - v_blend
            assert np.all(diff < 1.01)

            img_fg = np.full((3, 3, 2), v1, dtype=dtype)
            img_bg = np.full((3, 3, 2), v2, dtype=dtype)
            img_blend = blend.blend_alpha(img_fg, img_bg, [1.0, 0.0], eps=0.1)
            assert img_blend.dtype.name == np.dtype(dtype)
            assert img_blend.shape == (3, 3, 2)
            assert np.all(img_blend[:, :, 0] == dtype(v1))
            assert np.all(img_blend[:, :, 1] == dtype(v2))

            # elementwise, alphas.shape = (1, 2)
            img_fg = np.full((1, 2, 3), v1, dtype=dtype)
            img_bg = np.full((1, 2, 3), v2, dtype=dtype)
            alphas = np.zeros((1, 2), dtype=np.float64)
            alphas[:, :] = [1.0, 0.0]
            img_blend = blend.blend_alpha(img_fg, img_bg, alphas, eps=0)
            assert img_blend.dtype.name == np.dtype(dtype)
            assert img_blend.shape == (1, 2, 3)
            assert np.all(img_blend[0, 0, :] == dtype(v1))
            assert np.all(img_blend[0, 1, :] == dtype(v2))

            # elementwise, alphas.shape = (1, 2, 1)
            img_fg = np.full((1, 2, 3), v1, dtype=dtype)
            img_bg = np.full((1, 2, 3), v2, dtype=dtype)
            alphas = np.zeros((1, 2, 1), dtype=np.float64)
            alphas[:, :, 0] = [1.0, 0.0]
            img_blend = blend.blend_alpha(img_fg, img_bg, alphas, eps=0)
            assert img_blend.dtype.name == np.dtype(dtype)
            assert img_blend.shape == (1, 2, 3)
            assert np.all(img_blend[0, 0, :] == dtype(v1))
            assert np.all(img_blend[0, 1, :] == dtype(v2))

            # elementwise, alphas.shape = (1, 2, 3)
            img_fg = np.full((1, 2, 3), v1, dtype=dtype)
            img_bg = np.full((1, 2, 3), v2, dtype=dtype)
            alphas = np.zeros((1, 2, 3), dtype=np.float64)
            alphas[:, :, 0] = [1.0, 0.0]
            alphas[:, :, 1] = [0.0, 1.0]
            alphas[:, :, 2] = [1.0, 0.0]
            img_blend = blend.blend_alpha(img_fg, img_bg, alphas, eps=0)
            assert img_blend.dtype.name == np.dtype(dtype)
            assert img_blend.shape == (1, 2, 3)
            assert np.all(img_blend[0, 0, [0, 2]] == dtype(v1))
            assert np.all(img_blend[0, 1, [0, 2]] == dtype(v2))
            assert np.all(img_blend[0, 0, 1] == dtype(v2))
            assert np.all(img_blend[0, 1, 1] == dtype(v1))

    for dtype in [np.float16, np.float32, np.float64]:
        def _allclose(a, b):
            atol = 1e-4 if dtype == np.float16 else 1e-8
            return np.allclose(a, b, atol=atol, rtol=0)

        isize = np.dtype(dtype).itemsize
        max_value = 1000 ** (isize - 1)
        min_value = -max_value
        center_value = 0
        values = [
            (0, 0),
            (0, 10),
            (10, 20),
            (min_value, min_value),
            (max_value, max_value),
            (min_value, max_value),
            (min_value, center_value),
            (center_value, max_value),
            (center_value + 0.20 * max_value, max_value),
            (center_value + 0.27 * max_value, max_value),
            (center_value + 0.40 * max_value, max_value),
            (min_value, 0),
            (0, max_value)
        ]
        values = values + [(v2, v1) for v1, v2 in values]

        max_float_dt = np.float128

        for v1, v2 in values:
            img_fg = np.full((3, 3, 1), v1, dtype=dtype)
            img_bg = np.full((3, 3, 1), v2, dtype=dtype)
            img_blend = blend.blend_alpha(img_fg, img_bg, 1.0, eps=0)
            assert img_blend.dtype.name == np.dtype(dtype)
            assert img_blend.shape == (3, 3, 1)
            assert _allclose(img_blend, max_float_dt(v1))

            img_fg = np.full((3, 3, 1), v1, dtype=dtype)
            img_bg = np.full((3, 3, 1), v2, dtype=dtype)
            img_blend = blend.blend_alpha(img_fg, img_bg, 0.99, eps=0.1)
            assert img_blend.dtype.name == np.dtype(dtype)
            assert img_blend.shape == (3, 3, 1)
            assert _allclose(img_blend, max_float_dt(v1))

            img_fg = np.full((3, 3, 1), v1, dtype=dtype)
            img_bg = np.full((3, 3, 1), v2, dtype=dtype)
            img_blend = blend.blend_alpha(img_fg, img_bg, 0.0, eps=0)
            assert img_blend.dtype.name == np.dtype(dtype)
            assert img_blend.shape == (3, 3, 1)
            assert _allclose(img_blend, max_float_dt(v2))

            for c in sm.xrange(3):
                img_fg = np.full((3, 3, c), v1, dtype=dtype)
                img_bg = np.full((3, 3, c), v2, dtype=dtype)
                img_blend = blend.blend_alpha(img_fg, img_bg, 0.75, eps=0)
                assert img_blend.dtype.name == np.dtype(dtype)
                assert img_blend.shape == (3, 3, c)
                assert _allclose(img_blend, 0.75*max_float_dt(v1) + 0.25*max_float_dt(v2))

            img_fg = np.full((3, 3, 2), v1, dtype=dtype)
            img_bg = np.full((3, 3, 2), v2, dtype=dtype)
            img_blend = blend.blend_alpha(img_fg, img_bg, [1.0, 0.0], eps=0.1)
            assert img_blend.dtype.name == np.dtype(dtype)
            assert img_blend.shape == (3, 3, 2)
            assert _allclose(img_blend[:, :, 0], max_float_dt(v1))
            assert _allclose(img_blend[:, :, 1], max_float_dt(v2))

            # elementwise, alphas.shape = (1, 2)
            img_fg = np.full((1, 2, 3), v1, dtype=dtype)
            img_bg = np.full((1, 2, 3), v2, dtype=dtype)
            alphas = np.zeros((1, 2), dtype=np.float64)
            alphas[:, :] = [1.0, 0.0]
            img_blend = blend.blend_alpha(img_fg, img_bg, alphas, eps=0)
            assert img_blend.dtype.name == np.dtype(dtype)
            assert img_blend.shape == (1, 2, 3)
            assert _allclose(img_blend[0, 0, :], dtype(v1))
            assert _allclose(img_blend[0, 1, :], dtype(v2))

            # elementwise, alphas.shape = (1, 2, 1)
            img_fg = np.full((1, 2, 3), v1, dtype=dtype)
            img_bg = np.full((1, 2, 3), v2, dtype=dtype)
            alphas = np.zeros((1, 2, 1), dtype=np.float64)
            alphas[:, :, 0] = [1.0, 0.0]
            img_blend = blend.blend_alpha(img_fg, img_bg, alphas, eps=0)
            assert img_blend.dtype.name == np.dtype(dtype)
            assert img_blend.shape == (1, 2, 3)
            assert _allclose(img_blend[0, 0, :], dtype(v1))
            assert _allclose(img_blend[0, 1, :], dtype(v2))

            # elementwise, alphas.shape = (1, 2, 3)
            img_fg = np.full((1, 2, 3), v1, dtype=dtype)
            img_bg = np.full((1, 2, 3), v2, dtype=dtype)
            alphas = np.zeros((1, 2, 3), dtype=np.float64)
            alphas[:, :, 0] = [1.0, 0.0]
            alphas[:, :, 1] = [0.0, 1.0]
            alphas[:, :, 2] = [1.0, 0.0]
            img_blend = blend.blend_alpha(img_fg, img_bg, alphas, eps=0)
            assert img_blend.dtype.name == np.dtype(dtype)
            assert img_blend.shape == (1, 2, 3)
            assert _allclose(img_blend[0, 0, [0, 2]], dtype(v1))
            assert _allclose(img_blend[0, 1, [0, 2]], dtype(v2))
            assert _allclose(img_blend[0, 0, 1], dtype(v2))
            assert _allclose(img_blend[0, 1, 1], dtype(v1))


def test_Alpha():
    reseed()

    base_img = np.zeros((3, 3, 1), dtype=np.uint8)

    heatmaps_arr = np.float32([[0.0, 0.0, 1.0],
                               [0.0, 0.0, 1.0],
                               [0.0, 1.0, 1.0]])
    heatmaps_arr_r1 = np.float32([[0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0]])
    heatmaps_arr_l1 = np.float32([[0.0, 1.0, 0.0],
                                  [0.0, 1.0, 0.0],
                                  [1.0, 1.0, 0.0]])
    heatmaps = HeatmapsOnImage(heatmaps_arr, shape=(3, 3, 3))

    segmaps_arr = np.int32([[0, 0, 1],
                            [0, 0, 1],
                            [0, 1, 1]])
    segmaps_arr_r1 = np.int32([[0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 1]])
    segmaps_arr_l1 = np.int32([[0, 1, 0],
                               [0, 1, 0],
                               [1, 1, 0]])
    segmaps = SegmentationMapsOnImage(segmaps_arr, shape=(3, 3, 3))

    aug = iaa.Alpha(1, iaa.Add(10), iaa.Add(20))
    observed = aug.augment_image(base_img)
    expected = np.round(base_img + 10).astype(np.uint8)
    assert np.allclose(observed, expected)

    for per_channel in [False, True]:
        aug = iaa.Alpha(1, iaa.Affine(translate_px={"x": 1}), iaa.Affine(translate_px={"x": -1}),
                        per_channel=per_channel)
        observed = aug.augment_heatmaps([heatmaps])[0]
        assert observed.shape == heatmaps.shape
        assert 0 - 1e-6 < heatmaps.min_value < 0 + 1e-6
        assert 1 - 1e-6 < heatmaps.max_value < 1 + 1e-6
        assert np.allclose(observed.get_arr(), heatmaps_arr_r1)

    for per_channel in [False, True]:
        aug = iaa.Alpha(1,
                        iaa.Affine(translate_px={"x": 1}),
                        iaa.Affine(translate_px={"x": -1}),
                        per_channel=per_channel)
        observed = aug.augment_segmentation_maps([segmaps])[0]
        assert observed.shape == segmaps.shape
        assert np.array_equal(observed.get_arr(), segmaps_arr_r1)

    aug = iaa.Alpha(0, iaa.Add(10), iaa.Add(20))
    observed = aug.augment_image(base_img)
    expected = np.round(base_img + 20).astype(np.uint8)
    assert np.allclose(observed, expected)

    for per_channel in [False, True]:
        aug = iaa.Alpha(0,
                        iaa.Affine(translate_px={"x": 1}),
                        iaa.Affine(translate_px={"x": -1}),
                        per_channel=per_channel)
        observed = aug.augment_heatmaps([heatmaps])[0]
        assert observed.shape == heatmaps.shape
        assert 0 - 1e-6 < heatmaps.min_value < 0 + 1e-6
        assert 1 - 1e-6 < heatmaps.max_value < 1 + 1e-6
        assert np.allclose(observed.get_arr(), heatmaps_arr_l1)

    for per_channel in [False, True]:
        aug = iaa.Alpha(0,
                        iaa.Affine(translate_px={"x": 1}),
                        iaa.Affine(translate_px={"x": -1}),
                        per_channel=per_channel)
        observed = aug.augment_segmentation_maps([segmaps])[0]
        assert observed.shape == segmaps.shape
        assert np.array_equal(observed.get_arr(), segmaps_arr_l1)

    aug = iaa.Alpha(0.75, iaa.Add(10), iaa.Add(20))
    observed = aug.augment_image(base_img)
    expected = np.round(base_img + 0.75 * 10 + 0.25 * 20).astype(np.uint8)
    assert np.allclose(observed, expected)

    aug = iaa.Alpha(0.75, None, iaa.Add(20))
    observed = aug.augment_image(base_img + 10)
    expected = np.round(base_img + 0.75 * 10 + 0.25 * (10 + 20)).astype(np.uint8)
    assert np.allclose(observed, expected)

    aug = iaa.Alpha(0.75, iaa.Add(10), None)
    observed = aug.augment_image(base_img + 10)
    expected = np.round(base_img + 0.75 * (10 + 10) + 0.25 * 10).astype(np.uint8)
    assert np.allclose(observed, expected)

    base_img = np.zeros((1, 2, 1), dtype=np.uint8)
    nb_iterations = 1000
    aug = iaa.Alpha((0.0, 1.0), iaa.Add(10), iaa.Add(110))
    values = []
    for _ in sm.xrange(nb_iterations):
        observed = aug.augment_image(base_img)
        observed_val = np.round(np.average(observed)) - 10
        values.append(observed_val / 100)

    nb_bins = 5
    hist, _ = np.histogram(values, bins=nb_bins, range=(0.0, 1.0), density=False)
    density_expected = 1.0/nb_bins
    density_tolerance = 0.05
    for nb_samples in hist:
        density = nb_samples / nb_iterations
        assert density_expected - density_tolerance < density < density_expected + density_tolerance

    # bad datatype for factor
    got_exception = False
    try:
        _ = iaa.Alpha(False, iaa.Add(10), None)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # per_channel
    aug = iaa.Alpha(1.0, iaa.Add((0, 100), per_channel=True), None, per_channel=True)
    observed = aug.augment_image(np.zeros((1, 1, 1000), dtype=np.uint8))
    uq = np.unique(observed)
    assert len(uq) > 1
    assert np.max(observed) > 80
    assert np.min(observed) < 20

    aug = iaa.Alpha((0.0, 1.0), iaa.Add(100), None, per_channel=True)
    observed = aug.augment_image(np.zeros((1, 1, 1000), dtype=np.uint8))
    uq = np.unique(observed)
    assert len(uq) > 1
    assert np.max(observed) > 80
    assert np.min(observed) < 20

    aug = iaa.Alpha((0.0, 1.0), iaa.Add(100), iaa.Add(0), per_channel=0.5)
    seen = [0, 0]
    for _ in sm.xrange(200):
        observed = aug.augment_image(np.zeros((1, 1, 100), dtype=np.uint8))
        uq = np.unique(observed)
        if len(uq) == 1:
            seen[0] += 1
        elif len(uq) > 1:
            seen[1] += 1
        else:
            assert False
    assert 100 - 50 < seen[0] < 100 + 50
    assert 100 - 50 < seen[1] < 100 + 50

    # bad datatype for per_channel
    got_exception = False
    try:
        _ = iaa.Alpha(0.5, iaa.Add(10), None, per_channel="test")
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # propagating
    aug = iaa.Alpha(0.5, iaa.Add(100), iaa.Add(50), name="AlphaTest")

    def propagator(images, augmenter, parents, default):
        if "Alpha" in augmenter.name:
            return False
        else:
            return default

    hooks = ia.HooksImages(propagator=propagator)
    image = np.zeros((10, 10, 3), dtype=np.uint8) + 1
    observed = aug.augment_image(image, hooks=hooks)
    assert np.array_equal(observed, image)

    # -----
    # keypoints
    # -----
    kps = [ia.Keypoint(x=5, y=10), ia.Keypoint(x=6, y=11)]
    kpsoi = ia.KeypointsOnImage(kps, shape=(20, 20, 3))

    aug = iaa.Alpha(1.0, iaa.Noop(), iaa.Affine(translate_px={"x": 1}))
    observed = aug.augment_keypoints([kpsoi])[0]
    expected = kpsoi.deepcopy()
    assert keypoints_equal([observed], [expected])

    aug = iaa.Alpha(0.501, iaa.Noop(), iaa.Affine(translate_px={"x": 1}))
    observed = aug.augment_keypoints([kpsoi])[0]
    expected = kpsoi.deepcopy()
    assert keypoints_equal([observed], [expected])

    aug = iaa.Alpha(0.0, iaa.Noop(), iaa.Affine(translate_px={"x": 1}))
    observed = aug.augment_keypoints([kpsoi])[0]
    expected = kpsoi.shift(x=1)
    assert keypoints_equal([observed], [expected])

    aug = iaa.Alpha(0.499, iaa.Noop(), iaa.Affine(translate_px={"x": 1}))
    observed = aug.augment_keypoints([kpsoi])[0]
    expected = kpsoi.shift(x=1)
    assert keypoints_equal([observed], [expected])

    # per_channel
    aug = iaa.Alpha(1.0, iaa.Noop(), iaa.Affine(translate_px={"x": 1}), per_channel=True)
    observed = aug.augment_keypoints([kpsoi])[0]
    expected = kpsoi.deepcopy()
    assert keypoints_equal([observed], [expected])

    aug = iaa.Alpha(0.0, iaa.Noop(), iaa.Affine(translate_px={"x": 1}), per_channel=True)
    observed = aug.augment_keypoints([kpsoi])[0]
    expected = kpsoi.shift(x=1)
    assert keypoints_equal([observed], [expected])

    aug = iaa.Alpha(iap.Choice([0.49, 0.51]), iaa.Noop(), iaa.Affine(translate_px={"x": 1}), per_channel=True)
    expected_same = kpsoi.deepcopy()
    expected_shifted = kpsoi.shift(x=1)
    seen = [0, 0]
    for _ in sm.xrange(200):
        observed = aug.augment_keypoints([kpsoi])[0]
        if keypoints_equal([observed], [expected_same]):
            seen[0] += 1
        elif keypoints_equal([observed], [expected_shifted]):
            seen[1] += 1
        else:
            assert False
    assert 100 - 50 < seen[0] < 100 + 50
    assert 100 - 50 < seen[1] < 100 + 50

    # empty keypoints
    aug = iaa.Alpha(0.501, iaa.Noop(), iaa.Affine(translate_px={"x": 1}))
    observed = aug.augment_keypoints(ia.KeypointsOnImage([], shape=(1, 2, 3)))
    assert len(observed.keypoints) == 0
    assert observed.shape == (1, 2, 3)

    # propagating
    aug = iaa.Alpha(0.0, iaa.Affine(translate_px={"x": 1}), iaa.Affine(translate_px={"y": 1}), name="AlphaTest")

    def propagator(kpsoi_to_aug, augmenter, parents, default):
        if "Alpha" in augmenter.name:
            return False
        else:
            return default

    hooks = ia.HooksKeypoints(propagator=propagator)
    observed = aug.augment_keypoints([kpsoi], hooks=hooks)[0]
    assert keypoints_equal([observed], [kpsoi])

    # -----
    # polygons
    # -----
    ps = [ia.Polygon([(5, 5), (10, 5), (10, 10)])]
    psoi = ia.PolygonsOnImage(ps, shape=(20, 20, 3))

    aug = iaa.Alpha(1.0, iaa.Noop(), iaa.Affine(translate_px={"x": 1}))
    observed = aug.augment_polygons([psoi])
    assert len(observed) == 1
    assert len(observed[0].polygons) == 1
    assert observed[0].shape == psoi.shape
    assert observed[0].polygons[0].exterior_almost_equals(psoi.polygons[0])
    assert observed[0].polygons[0].is_valid

    aug = iaa.Alpha(0.501, iaa.Noop(), iaa.Affine(translate_px={"x": 1}))
    observed = aug.augment_polygons([psoi])
    assert len(observed) == 1
    assert len(observed[0].polygons) == 1
    assert observed[0].shape == psoi.shape
    assert observed[0].polygons[0].exterior_almost_equals(psoi.polygons[0])
    assert observed[0].polygons[0].is_valid

    aug = iaa.Alpha(0.0, iaa.Noop(), iaa.Affine(translate_px={"x": 1}))
    observed = aug.augment_polygons([psoi])
    expected = psoi.shift(left=1)
    assert len(observed) == 1
    assert len(observed[0].polygons) == 1
    assert observed[0].shape == psoi.shape
    assert observed[0].polygons[0].exterior_almost_equals(expected.polygons[0])
    assert observed[0].polygons[0].is_valid

    aug = iaa.Alpha(0.499, iaa.Noop(), iaa.Affine(translate_px={"x": 1}))
    observed = aug.augment_polygons([psoi])
    expected = psoi.shift(left=1)
    assert len(observed) == 1
    assert len(observed[0].polygons) == 1
    assert observed[0].shape == psoi.shape
    assert observed[0].polygons[0].exterior_almost_equals(expected.polygons[0])
    assert observed[0].polygons[0].is_valid

    # per_channel
    aug = iaa.Alpha(1.0, iaa.Noop(), iaa.Affine(translate_px={"x": 1}), per_channel=True)
    observed = aug.augment_polygons([psoi])
    assert len(observed) == 1
    assert len(observed[0].polygons) == 1
    assert observed[0].shape == psoi.shape
    assert observed[0].polygons[0].exterior_almost_equals(psoi.polygons[0])
    assert observed[0].polygons[0].is_valid

    aug = iaa.Alpha(0.0, iaa.Noop(), iaa.Affine(translate_px={"x": 1}), per_channel=True)
    observed = aug.augment_polygons([psoi])
    expected = psoi.shift(left=1)
    assert len(observed) == 1
    assert len(observed[0].polygons) == 1
    assert observed[0].shape == psoi.shape
    assert observed[0].polygons[0].exterior_almost_equals(expected.polygons[0])
    assert observed[0].polygons[0].is_valid

    aug = iaa.Alpha(iap.Choice([0.49, 0.51]), iaa.Noop(), iaa.Affine(translate_px={"x": 1}), per_channel=True)
    expected_same = psoi.deepcopy()
    expected_shifted = psoi.shift(left=1)
    seen = [0, 0]
    for _ in sm.xrange(200):
        observed = aug.augment_polygons([psoi])[0]
        if observed.polygons[0].exterior_almost_equals(expected_same.polygons[0]):
            seen[0] += 1
        elif observed.polygons[0].exterior_almost_equals(expected_shifted.polygons[0]):
            seen[1] += 1
        else:
            assert False
    assert 100 - 50 < seen[0] < 100 + 50
    assert 100 - 50 < seen[1] < 100 + 50

    # empty polygons
    aug = iaa.Alpha(0.501, iaa.Noop(), iaa.Affine(translate_px={"x": 1}))
    observed = aug.augment_polygons(ia.PolygonsOnImage([], shape=(1, 2, 3)))
    assert len(observed.polygons) == 0
    assert observed.shape == (1, 2, 3)

    # propagating
    aug = iaa.Alpha(0.0, iaa.Affine(translate_px={"x": 1}), iaa.Affine(translate_px={"y": 1}), name="AlphaTest")

    def propagator(psoi_to_aug, augmenter, parents, default):
        if "Alpha" in augmenter.name:
            return False
        else:
            return default

    hooks = ia.HooksKeypoints(propagator=propagator)  # no hooks for polygons yet, so we use HooksKeypoints
    observed = aug.augment_polygons([psoi], hooks=hooks)[0]
    assert observed.polygons[0].exterior_almost_equals(psoi.polygons[0])

    # -----
    # get_parameters()
    # -----
    first = iaa.Noop()
    second = iaa.Sequential([iaa.Add(1)])
    aug = iaa.Alpha(0.65, first, second, per_channel=1)
    params = aug.get_parameters()
    assert isinstance(params[0], iap.Deterministic)
    assert isinstance(params[1], iap.Deterministic)
    assert 0.65 - 1e-6 < params[0].value < 0.65 + 1e-6
    assert params[1].value == 1

    # -----
    # get_children_lists()
    # -----
    first = iaa.Noop()
    second = iaa.Sequential([iaa.Add(1)])
    aug = iaa.Alpha(0.65, first, second, per_channel=1)
    children_lsts = aug.get_children_lists()
    assert len(children_lsts) == 2
    assert ia.is_iterable([lst for lst in children_lsts])
    assert first in children_lsts[0]
    assert second == children_lsts[1]


def test_AlphaElementwise():
    reseed()

    base_img = np.zeros((3, 3, 1), dtype=np.uint8)

    heatmaps_arr = np.float32([[0.0, 0.0, 1.0],
                               [0.0, 0.0, 1.0],
                               [0.0, 1.0, 1.0]])
    heatmaps_arr_r1 = np.float32([[0.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0]])
    heatmaps_arr_l1 = np.float32([[0.0, 1.0, 0.0],
                                  [0.0, 1.0, 0.0],
                                  [1.0, 1.0, 0.0]])
    heatmaps = HeatmapsOnImage(heatmaps_arr, shape=(3, 3, 3))

    segmaps_arr = np.int32([[0, 0, 1],
                            [0, 0, 1],
                            [0, 1, 1]])
    segmaps_arr_r1 = np.int32([[0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 1]])
    segmaps_arr_l1 = np.int32([[0, 1, 0],
                               [0, 1, 0],
                               [1, 1, 0]])
    segmaps = SegmentationMapsOnImage(segmaps_arr, shape=(3, 3, 3))

    aug = iaa.AlphaElementwise(1, iaa.Add(10), iaa.Add(20))
    observed = aug.augment_image(base_img)
    expected = base_img + 10
    assert np.allclose(observed, expected)

    aug = iaa.AlphaElementwise(1,
                               iaa.Affine(translate_px={"x": 1}),
                               iaa.Affine(translate_px={"x": -1}))
    observed = aug.augment_heatmaps([heatmaps])[0]
    assert observed.shape == (3, 3, 3)
    assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
    assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
    assert np.allclose(observed.get_arr(), heatmaps_arr_r1)

    aug = iaa.AlphaElementwise(1,
                               iaa.Affine(translate_px={"x": 1}),
                               iaa.Affine(translate_px={"x": -1}))
    observed = aug.augment_segmentation_maps([segmaps])[0]
    assert observed.shape == (3, 3, 3)
    assert np.array_equal(observed.get_arr(), segmaps_arr_r1)

    aug = iaa.AlphaElementwise(0, iaa.Add(10), iaa.Add(20))
    observed = aug.augment_image(base_img)
    expected = base_img + 20
    assert np.allclose(observed, expected)

    aug = iaa.AlphaElementwise(0,
                               iaa.Affine(translate_px={"x": 1}),
                               iaa.Affine(translate_px={"x": -1}))
    observed = aug.augment_heatmaps([heatmaps])[0]
    assert observed.shape == (3, 3, 3)
    assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
    assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
    assert np.allclose(observed.get_arr(), heatmaps_arr_l1)

    aug = iaa.AlphaElementwise(0,
                               iaa.Affine(translate_px={"x": 1}),
                               iaa.Affine(translate_px={"x": -1}))
    observed = aug.augment_segmentation_maps([segmaps])[0]
    assert observed.shape == (3, 3, 3)
    assert np.array_equal(observed.get_arr(), segmaps_arr_l1)

    aug = iaa.AlphaElementwise(0.75, iaa.Add(10), iaa.Add(20))
    observed = aug.augment_image(base_img)
    expected = np.round(base_img + 0.75 * 10 + 0.25 * 20).astype(np.uint8)
    assert np.allclose(observed, expected)

    aug = iaa.AlphaElementwise(0.75, None, iaa.Add(20))
    observed = aug.augment_image(base_img + 10)
    expected = np.round(base_img + 0.75 * 10 + 0.25 * (10 + 20)).astype(np.uint8)
    assert np.allclose(observed, expected)

    aug = iaa.AlphaElementwise(0.75, iaa.Add(10), None)
    observed = aug.augment_image(base_img + 10)
    expected = np.round(base_img + 0.75 * (10 + 10) + 0.25 * 10).astype(np.uint8)
    assert np.allclose(observed, expected)

    base_img = np.zeros((100, 100), dtype=np.uint8)
    aug = iaa.AlphaElementwise((0.0, 1.0), iaa.Add(10), iaa.Add(110))
    observed = (aug.augment_image(base_img) - 10) / 100
    nb_bins = 10
    hist, _ = np.histogram(observed.flatten(),  bins=nb_bins, range=(0.0, 1.0), density=False)
    density_expected = 1.0/nb_bins
    density_tolerance = 0.05
    for nb_samples in hist:
        density = nb_samples / observed.size
        assert density_expected - density_tolerance < density < density_expected + density_tolerance

    base_img = np.zeros((1, 1, 100), dtype=np.uint8)
    aug = iaa.AlphaElementwise((0.0, 1.0), iaa.Add(10), iaa.Add(110), per_channel=True)
    observed = aug.augment_image(base_img)
    assert len(set(observed.flatten())) > 1

    # propagating
    aug = iaa.AlphaElementwise(0.5, iaa.Add(100), iaa.Add(50), name="AlphaElementwiseTest")

    def propagator(images, augmenter, parents, default):
        if "AlphaElementwise" in augmenter.name:
            return False
        else:
            return default

    hooks = ia.HooksImages(propagator=propagator)
    image = np.zeros((10, 10, 3), dtype=np.uint8) + 1
    observed = aug.augment_image(image, hooks=hooks)
    assert np.array_equal(observed, image)

    # -----
    # heatmaps and per_channel
    # -----
    class _DummyMaskParameter(iap.StochasticParameter):
        def __init__(self, inverted=False):
            super(_DummyMaskParameter, self).__init__()
            self.nb_calls = 0
            self.inverted = inverted

        def _draw_samples(self, size, random_state):
            self.nb_calls += 1
            h, w = size
            ones = np.ones((h, w), dtype=np.float32)
            zeros = np.zeros((h, w), dtype=np.float32)
            if self.nb_calls == 1:
                return zeros if not self.inverted else ones
            elif self.nb_calls in [2, 3]:
                return ones if not self.inverted else zeros
            else:
                assert False

    aug = iaa.AlphaElementwise(
        _DummyMaskParameter(inverted=False),
        iaa.Affine(translate_px={"x": 1}),
        iaa.Affine(translate_px={"x": -1}),
        per_channel=True
    )
    observed = aug.augment_heatmaps([heatmaps])[0]
    assert observed.shape == (3, 3, 3)
    assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
    assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
    assert np.allclose(observed.get_arr(), heatmaps_arr_r1)

    aug = iaa.AlphaElementwise(
        _DummyMaskParameter(inverted=True),
        iaa.Affine(translate_px={"x": 1}),
        iaa.Affine(translate_px={"x": -1}),
        per_channel=True
    )
    observed = aug.augment_heatmaps([heatmaps])[0]
    assert observed.shape == (3, 3, 3)
    assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
    assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
    assert np.allclose(observed.get_arr(), heatmaps_arr_l1)

    # -----
    # segmaps and per_channel
    # -----
    aug = iaa.AlphaElementwise(
        _DummyMaskParameter(inverted=False),
        iaa.Affine(translate_px={"x": 1}),
        iaa.Affine(translate_px={"x": -1}),
        per_channel=True
    )
    observed = aug.augment_segmentation_maps([segmaps])[0]
    assert observed.shape == (3, 3, 3)
    assert np.array_equal(observed.get_arr(), segmaps_arr_r1)

    aug = iaa.AlphaElementwise(
        _DummyMaskParameter(inverted=True),
        iaa.Affine(translate_px={"x": 1}),
        iaa.Affine(translate_px={"x": -1}),
        per_channel=True
    )
    observed = aug.augment_segmentation_maps([segmaps])[0]
    assert observed.shape == (3, 3, 3)
    assert np.array_equal(observed.get_arr(), segmaps_arr_l1)

    # -----
    # keypoints
    # -----
    kps = [ia.Keypoint(x=5, y=10), ia.Keypoint(x=6, y=11)]
    kpsoi = ia.KeypointsOnImage(kps, shape=(20, 20, 3))

    aug = iaa.AlphaElementwise(1.0, iaa.Noop(), iaa.Affine(translate_px={"x": 1}))
    observed = aug.augment_keypoints([kpsoi])[0]
    expected = kpsoi.deepcopy()
    assert keypoints_equal([observed], [expected])

    aug = iaa.AlphaElementwise(0.501, iaa.Noop(), iaa.Affine(translate_px={"x": 1}))
    observed = aug.augment_keypoints([kpsoi])[0]
    expected = kpsoi.deepcopy()
    assert keypoints_equal([observed], [expected])

    aug = iaa.AlphaElementwise(0.0, iaa.Noop(), iaa.Affine(translate_px={"x": 1}))
    observed = aug.augment_keypoints([kpsoi])[0]
    expected = kpsoi.shift(x=1)
    assert keypoints_equal([observed], [expected])

    aug = iaa.AlphaElementwise(0.499, iaa.Noop(), iaa.Affine(translate_px={"x": 1}))
    observed = aug.augment_keypoints([kpsoi])[0]
    expected = kpsoi.shift(x=1)
    assert keypoints_equal([observed], [expected])

    # per_channel
    aug = iaa.AlphaElementwise(1.0, iaa.Noop(), iaa.Affine(translate_px={"x": 1}), per_channel=True)
    observed = aug.augment_keypoints([kpsoi])[0]
    expected = kpsoi.deepcopy()
    assert keypoints_equal([observed], [expected])

    aug = iaa.AlphaElementwise(0.0, iaa.Noop(), iaa.Affine(translate_px={"x": 1}), per_channel=True)
    observed = aug.augment_keypoints([kpsoi])[0]
    expected = kpsoi.shift(x=1)
    assert keypoints_equal([observed], [expected])

    """
    TODO this test currently doesn't work as AlphaElementwise augments keypoints without sampling
    overlay factors per (x, y) location. (i.e. similar behaviour to Alpha)

    aug = iaa.Alpha(iap.Choice([0.49, 0.51]), iaa.Noop(), iaa.Affine(translate_px={"x": 1}), per_channel=True)
    expected_same = kpsoi.deepcopy()
    expected_both_shifted = kpsoi.shift(x=1)
    expected_first_shifted = KeypointsOnImage([kps[0].shift(x=1), kps[1]], shape=kpsoi.shape)
    expected_second_shifted = KeypointsOnImage([kps[0], kps[1].shift(x=1)], shape=kpsoi.shape)
    seen = [0, 0]
    for _ in sm.xrange(200):
        observed = aug.augment_keypoints([kpsoi])[0]
        if keypoints_equal([observed], [expected_same]):
            seen[0] += 1
        elif keypoints_equal([observed], [expected_both_shifted]):
            seen[1] += 1
        elif keypoints_equal([observed], [expected_first_shifted]):
            seen[2] += 1
        elif keypoints_equal([observed], [expected_second_shifted]):
            seen[3] += 1
        else:
            assert False
    assert 100 - 50 < seen[0] < 100 + 50
    assert 100 - 50 < seen[1] < 100 + 50
    """

    # propagating
    aug = iaa.AlphaElementwise(0.0, iaa.Affine(translate_px={"x": 1}), iaa.Affine(translate_px={"y": 1}),
                               name="AlphaElementwiseTest")

    def propagator(kpsoi_to_aug, augmenter, parents, default):
        if "AlphaElementwise" in augmenter.name:
            return False
        else:
            return default

    hooks = ia.HooksKeypoints(propagator=propagator)
    observed = aug.augment_keypoints([kpsoi], hooks=hooks)[0]
    assert keypoints_equal([observed], [kpsoi])

    # -----
    # polygons
    # -----
    ps = [ia.Polygon([(5, 5), (10, 5), (10, 10)])]
    psoi = ia.PolygonsOnImage(ps, shape=(20, 20, 3))

    aug = iaa.AlphaElementwise(1.0, iaa.Noop(), iaa.Affine(translate_px={"x": 1}))
    observed = aug.augment_polygons([psoi])
    assert len(observed) == 1
    assert len(observed[0].polygons) == 1
    assert observed[0].shape == psoi.shape
    assert observed[0].polygons[0].exterior_almost_equals(psoi.polygons[0])
    assert observed[0].polygons[0].is_valid

    aug = iaa.AlphaElementwise(0.501, iaa.Noop(), iaa.Affine(translate_px={"x": 1}))
    observed = aug.augment_polygons([psoi])
    assert len(observed) == 1
    assert len(observed[0].polygons) == 1
    assert observed[0].shape == psoi.shape
    assert observed[0].polygons[0].exterior_almost_equals(psoi.polygons[0])
    assert observed[0].polygons[0].is_valid

    aug = iaa.AlphaElementwise(0.0, iaa.Noop(), iaa.Affine(translate_px={"x": 1}))
    observed = aug.augment_polygons([psoi])
    expected = psoi.shift(left=1)
    assert len(observed) == 1
    assert len(observed[0].polygons) == 1
    assert observed[0].shape == psoi.shape
    assert observed[0].polygons[0].exterior_almost_equals(expected.polygons[0])
    assert observed[0].polygons[0].is_valid

    aug = iaa.AlphaElementwise(0.499, iaa.Noop(), iaa.Affine(translate_px={"x": 1}))
    observed = aug.augment_polygons([psoi])
    expected = psoi.shift(left=1)
    assert len(observed) == 1
    assert len(observed[0].polygons) == 1
    assert observed[0].shape == psoi.shape
    assert observed[0].polygons[0].exterior_almost_equals(expected.polygons[0])
    assert observed[0].polygons[0].is_valid

    # per_channel
    aug = iaa.AlphaElementwise(1.0, iaa.Noop(), iaa.Affine(translate_px={"x": 1}), per_channel=True)
    observed = aug.augment_polygons([psoi])
    assert len(observed) == 1
    assert len(observed[0].polygons) == 1
    assert observed[0].shape == psoi.shape
    assert observed[0].polygons[0].exterior_almost_equals(psoi.polygons[0])
    assert observed[0].polygons[0].is_valid

    aug = iaa.AlphaElementwise(0.0, iaa.Noop(), iaa.Affine(translate_px={"x": 1}), per_channel=True)
    observed = aug.augment_polygons([psoi])
    expected = psoi.shift(left=1)
    assert len(observed) == 1
    assert len(observed[0].polygons) == 1
    assert observed[0].shape == psoi.shape
    assert observed[0].polygons[0].exterior_almost_equals(expected.polygons[0])
    assert observed[0].polygons[0].is_valid

    aug = iaa.AlphaElementwise(iap.Choice([0.49, 0.51]), iaa.Noop(), iaa.Affine(translate_px={"x": 1}), per_channel=True)
    expected_same = psoi.deepcopy()
    expected_shifted = psoi.shift(left=1)
    seen = [0, 0]
    for _ in sm.xrange(200):
        observed = aug.augment_polygons([psoi])[0]
        if observed.polygons[0].exterior_almost_equals(expected_same.polygons[0]):
            seen[0] += 1
        elif observed.polygons[0].exterior_almost_equals(expected_shifted.polygons[0]):
            seen[1] += 1
        else:
            assert False
    assert 100 - 50 < seen[0] < 100 + 50
    assert 100 - 50 < seen[1] < 100 + 50

    # empty polygons
    aug = iaa.AlphaElementwise(0.501, iaa.Noop(), iaa.Affine(translate_px={"x": 1}))
    observed = aug.augment_polygons(ia.PolygonsOnImage([], shape=(1, 2, 3)))
    assert len(observed.polygons) == 0
    assert observed.shape == (1, 2, 3)

    # propagating
    aug = iaa.AlphaElementwise(0.0, iaa.Affine(translate_px={"x": 1}), iaa.Affine(translate_px={"y": 1}), name="AlphaTest")

    def propagator(psoi_to_aug, augmenter, parents, default):
        if "Alpha" in augmenter.name:
            return False
        else:
            return default

    hooks = ia.HooksKeypoints(propagator=propagator)  # no hooks for polygons yet, so we use HooksKeypoints
    observed = aug.augment_polygons([psoi], hooks=hooks)[0]
    assert observed.polygons[0].exterior_almost_equals(psoi.polygons[0])


if __name__ == "__main__":
    main()
