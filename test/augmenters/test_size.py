from __future__ import print_function, division, absolute_import

import time

import matplotlib
matplotlib.use('Agg')  # fix execution of tests involving matplotlib on travis
import numpy as np
import six.moves as sm
import cv2

import imgaug as ia
from imgaug import augmenters as iaa
import imgaug.augmenters.size as iaa_size
from imgaug import parameters as iap
from imgaug import dtypes as iadt
from imgaug.testutils import array_equal_lists, keypoints_equal, reseed


def main():
    time_start = time.time()

    test__handle_position_parameter()

    test_Resize()
    # TODO test_CropAndPad()
    test_Pad()
    test_Crop()
    test_PadToFixedSize()
    test_CropToFixedSize()
    test_KeepSizeByResize()

    time_end = time.time()
    print("<%s> Finished without errors in %.4fs." % (__file__, time_end - time_start,))


def test__handle_position_parameter():
    observed = iaa_size._handle_position_parameter("uniform")
    assert isinstance(observed, tuple)
    assert len(observed) == 2
    for i in range(2):
        assert isinstance(observed[i], iap.Uniform)
        assert isinstance(observed[i].a, iap.Deterministic)
        assert isinstance(observed[i].b, iap.Deterministic)
        assert 0.0 - 1e-4 < observed[i].a.value < 0.0 + 1e-4
        assert 1.0 - 1e-4 < observed[i].b.value < 1.0 + 1e-4

    observed = iaa_size._handle_position_parameter("center")
    assert isinstance(observed, tuple)
    assert len(observed) == 2
    for i in range(2):
        assert isinstance(observed[i], iap.Deterministic)
        assert 0.5 - 1e-4 < observed[i].value < 0.5 + 1e-4

    observed = iaa_size._handle_position_parameter("normal")
    assert isinstance(observed, tuple)
    assert len(observed) == 2
    for i in range(2):
        assert isinstance(observed[i], iap.Clip)
        assert isinstance(observed[i].other_param, iap.Normal)
        assert isinstance(observed[i].other_param.loc, iap.Deterministic)
        assert isinstance(observed[i].other_param.scale, iap.Deterministic)
        assert 0.5 - 1e-4 < observed[i].other_param.loc.value < 0.5 + 1e-4
        assert 0.35/2 - 1e-4 < observed[i].other_param.scale.value < 0.35/2 + 1e-4

    pos_x = [
        ("left", 0.0),
        ("center", 0.5),
        ("right", 1.0)
    ]
    pos_y = [
        ("top", 0.0),
        ("center", 0.5),
        ("bottom", 1.0)
    ]
    for x_str, x_val in pos_x:
        for y_str, y_val in pos_y:
            observed = iaa_size._handle_position_parameter("%s-%s" % (x_str, y_str))
            assert isinstance(observed[0], iap.Deterministic)
            assert x_val - 1e-4 < observed[0].value < x_val + 1e-4
            assert isinstance(observed[1], iap.Deterministic)
            assert y_val - 1e-4 < observed[1].value < y_val + 1e-4

    observed = iaa_size._handle_position_parameter(iap.Poisson(2))
    assert isinstance(observed, iap.Poisson)

    observed = iaa_size._handle_position_parameter((0.4, 0.6))
    assert isinstance(observed, tuple)
    assert len(observed) == 2
    assert isinstance(observed[0], iap.Deterministic)
    assert 0.4 - 1e-4 < observed[0].value < 0.4 + 1e-4
    assert isinstance(observed[1], iap.Deterministic)
    assert 0.6 - 1e-4 < observed[1].value < 0.6 + 1e-4

    got_exception = False
    try:
        _ = iaa_size._handle_position_parameter((1.2, 0.6))
    except Exception as e:
        assert "must be within the value range" in str(e)
        got_exception = True
    assert got_exception

    observed = iaa_size._handle_position_parameter((iap.Poisson(2), iap.Poisson(3)))
    assert isinstance(observed[0], iap.Poisson)
    assert isinstance(observed[0].lam, iap.Deterministic)
    assert 2 - 1e-4 < observed[0].lam.value < 2 + 1e-4
    assert isinstance(observed[1], iap.Poisson)
    assert isinstance(observed[1].lam, iap.Deterministic)
    assert 3 - 1e-4 < observed[1].lam.value < 3 + 1e-4

    observed = iaa_size._handle_position_parameter((0.4, iap.Poisson(3)))
    assert isinstance(observed, tuple)
    assert len(observed) == 2
    assert isinstance(observed[0], iap.Deterministic)
    assert 0.4 - 1e-4 < observed[0].value < 0.4 + 1e-4
    assert isinstance(observed[1], iap.Poisson)
    assert isinstance(observed[1].lam, iap.Deterministic)
    assert 3 - 1e-4 < observed[1].lam.value < 3 + 1e-4

    got_exception = False
    try:
        _ = iaa_size._handle_position_parameter(False)
    except Exception as e:
        assert "Expected one of the following as position parameter" in str(e)
        got_exception = True
    assert got_exception


def test_Resize():
    reseed()

    base_img2d = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 255, 255, 255, 255, 255, 255, 0],
        [0, 255, 255, 255, 255, 255, 255, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ]
    base_img2d = np.array(base_img2d, dtype=np.uint8)
    base_img3d = np.tile(base_img2d[..., np.newaxis], (1, 1, 3))

    intensity_avg = np.average(base_img2d)
    intensity_low = intensity_avg - 0.2 * np.abs(intensity_avg - 128)
    intensity_high = intensity_avg + 0.2 * np.abs(intensity_avg - 128)

    aspect_ratio2d = base_img2d.shape[1] / base_img2d.shape[0]
    aspect_ratio3d = base_img3d.shape[1] / base_img3d.shape[0]

    aug = iaa.Resize(12)
    observed2d = aug.augment_image(base_img2d)
    observed3d = aug.augment_image(base_img3d)
    assert observed2d.shape == (12, 12)
    assert observed3d.shape == (12, 12, 3)
    assert 50 < np.average(observed2d) < 205
    assert 50 < np.average(observed3d) < 205

    # heatmaps
    aug = iaa.Resize({"height": 8, "width": 12})
    heatmaps_arr = (base_img2d / 255.0).astype(np.float32)
    heatmaps_aug = aug.augment_heatmaps([ia.HeatmapsOnImage(heatmaps_arr, shape=base_img3d.shape)])[0]
    assert heatmaps_aug.shape == (8, 12, 3)
    assert 0 - 1e-6 < heatmaps_aug.min_value < 0 + 1e-6
    assert 1 - 1e-6 < heatmaps_aug.max_value < 1 + 1e-6
    assert np.average(heatmaps_aug.get_arr()[0, :]) < 0.05
    assert np.average(heatmaps_aug.get_arr()[-1, :]) < 0.05
    assert np.average(heatmaps_aug.get_arr()[:, 0]) < 0.05
    assert 0.8 < np.average(heatmaps_aug.get_arr()[2:6, 2:10]) < 1 + 1e-6

    # heatmaps with different sizes than image
    aug = iaa.Resize({"width": 2.0, "height": 16})
    heatmaps_arr = (base_img2d / 255.0).astype(np.float32)
    heatmaps = ia.HeatmapsOnImage(heatmaps_arr, shape=(2*base_img3d.shape[0], 2*base_img3d.shape[1], 3))
    heatmaps_aug = aug.augment_heatmaps([heatmaps])[0]
    assert heatmaps_aug.shape == (16, int(base_img3d.shape[1]*2*2), 3)
    assert heatmaps_aug.arr_0to1.shape == (8, 16, 1)
    assert 0 - 1e-6 < heatmaps_aug.min_value < 0 + 1e-6
    assert 1 - 1e-6 < heatmaps_aug.max_value < 1 + 1e-6
    assert np.average(heatmaps_aug.get_arr()[0, :]) < 0.05
    assert np.average(heatmaps_aug.get_arr()[-1:, :]) < 0.05
    assert np.average(heatmaps_aug.get_arr()[:, 0]) < 0.05
    assert 0.8 < np.average(heatmaps_aug.get_arr()[2:6, 2:10]) < 1 + 1e-6

    # keypoints on 3d image
    aug = iaa.Resize({"height": 8, "width": 12})
    kpsoi = ia.KeypointsOnImage([ia.Keypoint(x=1, y=2), ia.Keypoint(x=4, y=1)], shape=base_img3d.shape)
    kpsoi_aug = aug.augment_keypoints([kpsoi])[0]
    assert len(kpsoi_aug.keypoints) == 2
    assert kpsoi_aug.shape == (8, 12, 3)
    assert np.allclose(kpsoi_aug.keypoints[0].x, 1.5)
    assert np.allclose(kpsoi_aug.keypoints[0].y, 4)
    assert np.allclose(kpsoi_aug.keypoints[1].x, 6)
    assert np.allclose(kpsoi_aug.keypoints[1].y, 2)

    # polygons on 3d image
    aug = iaa.Resize({"width": 12, "height": 8})
    psoi = ia.PolygonsOnImage([
        ia.Polygon([(0, 0), (8, 0), (8, 4)]),
        ia.Polygon([(1, 1), (7, 1), (7, 3), (1, 3)]),
    ], shape=(4, 8, 3))
    psoi_aug = aug.augment_polygons(psoi)
    assert len(psoi_aug.polygons) == 2
    assert psoi_aug.shape == (8, 12, 3)
    assert psoi_aug.polygons[0].exterior_almost_equals(
        ia.Polygon([(0, 0), (12, 0), (12, 8)])
    )
    assert psoi_aug.polygons[1].exterior_almost_equals(
        ia.Polygon([(1.5, 2), (10.5, 2), (10.5, 6), (1.5, 6)])
    )

    # keypoints on 2d image,
    # different resize factors per axis
    aug = iaa.Resize({"width": 3.0, "height": 8})
    kpsoi = ia.KeypointsOnImage([ia.Keypoint(x=1, y=2), ia.Keypoint(x=4, y=1)], shape=base_img2d.shape)
    kpsoi_aug = aug.augment_keypoints([kpsoi])[0]
    assert len(kpsoi_aug.keypoints) == 2
    assert kpsoi_aug.shape == (8, 24)
    assert np.allclose(kpsoi_aug.keypoints[0].x, 3)
    assert np.allclose(kpsoi_aug.keypoints[0].y, 4)
    assert np.allclose(kpsoi_aug.keypoints[1].x, 12)
    assert np.allclose(kpsoi_aug.keypoints[1].y, 2)

    # polygons on 2d image
    # with float resize factor
    aug = iaa.Resize({"width": 3.0, "height": 8})
    psoi = ia.PolygonsOnImage([
        ia.Polygon([(0, 0), (8, 0), (8, 4)]),
        ia.Polygon([(1, 1), (7, 1), (7, 3), (1, 3)]),
    ], shape=(4, 8))
    psoi_aug = aug.augment_polygons(psoi)
    assert len(psoi_aug.polygons) == 2
    assert psoi_aug.shape == (8, 24)
    assert psoi_aug.polygons[0].exterior_almost_equals(
        ia.Polygon([(3*0, 0), (3*8, 0), (3*8, 8)])
    )
    assert psoi_aug.polygons[1].exterior_almost_equals(
        ia.Polygon([(3*1, 2), (3*7, 2), (3*7, 6), (3*1, 6)])
    )

    # empty keypoints
    aug = iaa.Resize({"height": 8, "width": 12})
    kpsoi = ia.KeypointsOnImage([], shape=(4, 8, 3))
    kpsoi_aug = aug.augment_keypoints(kpsoi)
    assert len(kpsoi_aug.keypoints) == 0
    assert kpsoi_aug.shape == (8, 12, 3)

    # empty polygons
    aug = iaa.Resize({"height": 8, "width": 12})
    psoi = ia.PolygonsOnImage([], shape=(4, 8, 3))
    psoi_aug = aug.augment_polygons(psoi)
    assert len(psoi_aug.polygons) == 0
    assert psoi_aug.shape == (8, 12, 3)

    # images with stochastic parameter (choice)
    aug = iaa.Resize([12, 14])
    seen2d = [False, False]
    seen3d = [False, False]
    for _ in sm.xrange(100):
        observed2d = aug.augment_image(base_img2d)
        observed3d = aug.augment_image(base_img3d)
        assert observed2d.shape in [(12, 12), (14, 14)]
        assert observed3d.shape in [(12, 12, 3), (14, 14, 3)]
        if observed2d.shape == (12, 12):
            seen2d[0] = True
        else:
            seen2d[1] = True
        if observed3d.shape == (12, 12, 3):
            seen3d[0] = True
        else:
            seen3d[1] = True
        if all(seen2d) and all(seen3d):
            break
    assert all(seen2d)
    assert all(seen3d)

    # images with stochastic parameter (uniform)
    aug = iaa.Resize((12, 14))
    seen2d = [False, False, False]
    seen3d = [False, False, False]
    for _ in sm.xrange(100):
        observed2d = aug.augment_image(base_img2d)
        observed3d = aug.augment_image(base_img3d)
        assert observed2d.shape in [(12, 12), (13, 13), (14, 14)]
        assert observed3d.shape in [(12, 12, 3), (13, 13, 3), (14, 14, 3)]
        if observed2d.shape == (12, 12):
            seen2d[0] = True
        elif observed2d.shape == (13, 13):
            seen2d[1] = True
        else:
            seen2d[2] = True
        if observed3d.shape == (12, 12, 3):
            seen3d[0] = True
        elif observed3d.shape == (13, 13, 3):
            seen3d[1] = True
        else:
            seen3d[2] = True
        if all(seen2d) and all(seen3d):
            break
    assert all(seen2d)
    assert all(seen3d)

    # test "keep"
    aug = iaa.Resize("keep")
    observed2d = aug.augment_image(base_img2d)
    observed3d = aug.augment_image(base_img3d)
    assert observed2d.shape == base_img2d.shape
    assert observed3d.shape == base_img3d.shape

    # empty list, no change
    aug = iaa.Resize([])
    observed2d = aug.augment_image(base_img2d)
    observed3d = aug.augment_image(base_img3d)
    assert observed2d.shape == base_img2d.shape
    assert observed3d.shape == base_img3d.shape

    # empty dict, no change
    aug = iaa.Resize({})
    observed2d = aug.augment_image(base_img2d)
    observed3d = aug.augment_image(base_img3d)
    assert observed2d.shape == base_img2d.shape
    assert observed3d.shape == base_img3d.shape

    # only change height
    aug = iaa.Resize({"height": 11})
    observed2d = aug.augment_image(base_img2d)
    observed3d = aug.augment_image(base_img3d)
    assert observed2d.shape == (11, base_img2d.shape[1])
    assert observed3d.shape == (11, base_img3d.shape[1], 3)

    # only change width
    aug = iaa.Resize({"width": 13})
    observed2d = aug.augment_image(base_img2d)
    observed3d = aug.augment_image(base_img3d)
    assert observed2d.shape == (base_img2d.shape[0], 13)
    assert observed3d.shape == (base_img3d.shape[0], 13, 3)

    # change height and width
    aug = iaa.Resize({"height": 12, "width": 13})
    observed2d = aug.augment_image(base_img2d)
    observed3d = aug.augment_image(base_img3d)
    assert observed2d.shape == (12, 13)
    assert observed3d.shape == (12, 13, 3)

    # change height, keep width
    aug = iaa.Resize({"height": 12, "width": "keep"})
    observed2d = aug.augment_image(base_img2d)
    observed3d = aug.augment_image(base_img3d)
    assert observed2d.shape == (12, base_img2d.shape[1])
    assert observed3d.shape == (12, base_img3d.shape[1], 3)

    # keep height, change width
    aug = iaa.Resize({"height": "keep", "width": 12})
    observed2d = aug.augment_image(base_img2d)
    observed3d = aug.augment_image(base_img3d)
    assert observed2d.shape == (base_img2d.shape[0], 12)
    assert observed3d.shape == (base_img3d.shape[0], 12, 3)

    # change height, keep width at same aspect ratio
    aug = iaa.Resize({"height": 12, "width": "keep-aspect-ratio"})
    observed2d = aug.augment_image(base_img2d)
    observed3d = aug.augment_image(base_img3d)
    assert observed2d.shape == (12, int(12 * aspect_ratio2d))
    assert observed3d.shape == (12, int(12 * aspect_ratio3d), 3)

    # keep height at same aspect ration, change width
    aug = iaa.Resize({"height": "keep-aspect-ratio", "width": 12})
    observed2d = aug.augment_image(base_img2d)
    observed3d = aug.augment_image(base_img3d)
    assert observed2d.shape == (int(12 * (1/aspect_ratio2d)), 12)
    assert observed3d.shape == (int(12 * (1/aspect_ratio3d)), 12, 3)

    # change height randomly, width deterministically
    aug = iaa.Resize({"height": [12, 14], "width": 12})
    seen2d = [False, False]
    seen3d = [False, False]
    for _ in sm.xrange(100):
        observed2d = aug.augment_image(base_img2d)
        observed3d = aug.augment_image(base_img3d)
        assert observed2d.shape in [(12, 12), (14, 12)]
        assert observed3d.shape in [(12, 12, 3), (14, 12, 3)]
        if observed2d.shape == (12, 12):
            seen2d[0] = True
        else:
            seen2d[1] = True
        if observed3d.shape == (12, 12, 3):
            seen3d[0] = True
        else:
            seen3d[1] = True
        if all(seen2d) and all(seen3d):
            break
    assert all(seen2d)
    assert all(seen3d)

    # change height deterministically, width randomly
    aug = iaa.Resize({"height": 12, "width": [12, 14]})
    seen2d = [False, False]
    seen3d = [False, False]
    for _ in sm.xrange(100):
        observed2d = aug.augment_image(base_img2d)
        observed3d = aug.augment_image(base_img3d)
        assert observed2d.shape in [(12, 12), (12, 14)]
        assert observed3d.shape in [(12, 12, 3), (12, 14, 3)]
        if observed2d.shape == (12, 12):
            seen2d[0] = True
        else:
            seen2d[1] = True
        if observed3d.shape == (12, 12, 3):
            seen3d[0] = True
        else:
            seen3d[1] = True
        if all(seen2d) and all(seen3d):
            break
    assert all(seen2d)
    assert all(seen3d)

    # change height deterministically, width randomly
    aug = iaa.Resize({"height": 12, "width": iap.Choice([12, 14])})
    seen2d = [False, False]
    seen3d = [False, False]
    for _ in sm.xrange(100):
        observed2d = aug.augment_image(base_img2d)
        observed3d = aug.augment_image(base_img3d)
        assert observed2d.shape in [(12, 12), (12, 14)]
        assert observed3d.shape in [(12, 12, 3), (12, 14, 3)]
        if observed2d.shape == (12, 12):
            seen2d[0] = True
        else:
            seen2d[1] = True
        if observed3d.shape == (12, 12, 3):
            seen3d[0] = True
        else:
            seen3d[1] = True
        if all(seen2d) and all(seen3d):
            break
    assert all(seen2d)
    assert all(seen3d)

    # change height randomly, width deterministically
    aug = iaa.Resize({"height": (12, 14), "width": 12})
    seen2d = [False, False, False]
    seen3d = [False, False, False]
    for _ in sm.xrange(100):
        observed2d = aug.augment_image(base_img2d)
        observed3d = aug.augment_image(base_img3d)
        assert observed2d.shape in [(12, 12), (13, 12), (14, 12)]
        assert observed3d.shape in [(12, 12, 3), (13, 12, 3), (14, 12, 3)]
        if observed2d.shape == (12, 12):
            seen2d[0] = True
        elif observed2d.shape == (13, 12):
            seen2d[1] = True
        else:
            seen2d[2] = True
        if observed3d.shape == (12, 12, 3):
            seen3d[0] = True
        elif observed3d.shape == (13, 12, 3):
            seen3d[1] = True
        else:
            seen3d[2] = True
        if all(seen2d) and all(seen3d):
            break
    assert all(seen2d)
    assert all(seen3d)

    # increase size by a factor of 2.0
    aug = iaa.Resize(2.0)
    observed2d = aug.augment_image(base_img2d)
    observed3d = aug.augment_image(base_img3d)
    assert observed2d.shape == (base_img2d.shape[0]*2, base_img2d.shape[1]*2)
    assert observed3d.shape == (base_img3d.shape[0]*2, base_img3d.shape[1]*2, 3)
    assert intensity_low < np.average(observed2d) < intensity_high
    assert intensity_low < np.average(observed3d) < intensity_high

    # increase size by a random factor
    aug = iaa.Resize([2.0, 4.0])
    seen2d = [False, False]
    seen3d = [False, False]
    for _ in sm.xrange(100):
        observed2d = aug.augment_image(base_img2d)
        observed3d = aug.augment_image(base_img3d)
        assert observed2d.shape in [(base_img2d.shape[0]*2, base_img2d.shape[1]*2),
                                    (base_img2d.shape[0]*4, base_img2d.shape[1]*4)]
        assert observed3d.shape in [(base_img3d.shape[0]*2, base_img3d.shape[1]*2, 3),
                                    (base_img3d.shape[0]*4, base_img3d.shape[1]*4, 3)]
        if observed2d.shape == (base_img2d.shape[0]*2, base_img2d.shape[1]*2):
            seen2d[0] = True
        else:
            seen2d[1] = True
        if observed3d.shape == (base_img3d.shape[0]*2, base_img3d.shape[1]*2, 3):
            seen3d[0] = True
        else:
            seen3d[1] = True
        if all(seen2d) and all(seen3d):
            break
    assert all(seen2d)
    assert all(seen3d)

    # increase size by a random factor
    aug = iaa.Resize(iap.Choice([2.0, 4.0]))
    seen2d = [False, False]
    seen3d = [False, False]
    for _ in sm.xrange(100):
        observed2d = aug.augment_image(base_img2d)
        observed3d = aug.augment_image(base_img3d)
        assert observed2d.shape in [(base_img2d.shape[0]*2, base_img2d.shape[1]*2),
                                    (base_img2d.shape[0]*4, base_img2d.shape[1]*4)]
        assert observed3d.shape in [(base_img3d.shape[0]*2, base_img3d.shape[1]*2, 3),
                                    (base_img3d.shape[0]*4, base_img3d.shape[1]*4, 3)]
        if observed2d.shape == (base_img2d.shape[0]*2, base_img2d.shape[1]*2):
            seen2d[0] = True
        else:
            seen2d[1] = True
        if observed3d.shape == (base_img3d.shape[0]*2, base_img3d.shape[1]*2, 3):
            seen3d[0] = True
        else:
            seen3d[1] = True
        if all(seen2d) and all(seen3d):
            break
    assert all(seen2d)
    assert all(seen3d)

    # decrease size by a random factor
    base_img2d = base_img2d[0:4, 0:4]
    base_img3d = base_img3d[0:4, 0:4, :]
    aug = iaa.Resize((0.76, 1.0))
    not_seen2d = set()
    not_seen3d = set()
    for size in sm.xrange(3, 4+1):
        not_seen2d.add((size, size))
    for size in sm.xrange(3, 4+1):
        not_seen3d.add((size, size, 3))
    possible2d = set(list(not_seen2d))
    possible3d = set(list(not_seen3d))
    for _ in sm.xrange(100):
        observed2d = aug.augment_image(base_img2d)
        observed3d = aug.augment_image(base_img3d)
        assert observed2d.shape in possible2d
        assert observed3d.shape in possible3d
        if observed2d.shape in not_seen2d:
            not_seen2d.remove(observed2d.shape)
        if observed3d.shape in not_seen3d:
            not_seen3d.remove(observed3d.shape)
        if not not_seen2d and not not_seen3d:
            break
    assert not not_seen2d
    assert not not_seen3d

    # decrease size by random factors, one per side
    base_img2d = base_img2d[0:4, 0:4]
    base_img3d = base_img3d[0:4, 0:4, :]
    aug = iaa.Resize({"height": (0.76, 1.0), "width": (0.76, 1.0)})
    not_seen2d = set()
    not_seen3d = set()
    for hsize in sm.xrange(3, 4+1):
        for wsize in sm.xrange(3, 4+1):
            not_seen2d.add((hsize, wsize))
    for hsize in sm.xrange(3, 4+1):
        for wsize in sm.xrange(3, 4+1):
            not_seen3d.add((hsize, wsize, 3))
    possible2d = set(list(not_seen2d))
    possible3d = set(list(not_seen3d))
    for _ in sm.xrange(100):
        observed2d = aug.augment_image(base_img2d)
        observed3d = aug.augment_image(base_img3d)
        assert observed2d.shape in possible2d
        assert observed3d.shape in possible3d
        if observed2d.shape in not_seen2d:
            not_seen2d.remove(observed2d.shape)
        if observed3d.shape in not_seen3d:
            not_seen3d.remove(observed3d.shape)
        if not not_seen2d and not not_seen3d:
            break
    assert not not_seen2d
    assert not not_seen3d

    # test bad input
    got_exception = False
    try:
        aug = iaa.Resize("foo")
        _ = aug.augment_image(base_img2d)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # test get_parameters
    aug = iaa.Resize(size=1, interpolation="nearest")
    params = aug.get_parameters()
    assert isinstance(params[0], iap.Deterministic)
    assert isinstance(params[1], iap.Deterministic)
    assert params[0].value == 1
    assert params[1].value == "nearest"


def test_Pad():
    reseed()

    base_img = np.array([[0, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0]], dtype=np.uint8)
    base_img = base_img[:, :, np.newaxis]

    images = np.array([base_img])
    images_list = [base_img]

    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                      ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

    heatmaps_arr = np.float32([[0, 0, 0],
                               [0, 1.0, 0],
                               [0, 0, 0]])

    # test pad by 1 pixel on each side
    pads = [
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1),
    ]
    for pad in pads:
        top, right, bottom, left = pad
        aug = iaa.Pad(px=pad, keep_size=False)
        base_img_padded = np.pad(base_img, ((top, bottom), (left, right), (0, 0)),
                                 mode="constant",
                                 constant_values=0)
        observed = aug.augment_images(images)
        assert np.array_equal(observed, np.array([base_img_padded]))

        observed = aug.augment_images(images_list)
        assert array_equal_lists(observed, [base_img_padded])

        keypoints_moved = [keypoints[0].shift(x=left, y=top)]
        observed = aug.augment_keypoints(keypoints)
        assert keypoints_equal(observed, keypoints_moved)

        # heatmaps
        aug = iaa.Pad(px=pad, keep_size=False)
        heatmaps_arr_padded = np.pad(heatmaps_arr, ((top, bottom), (left, right)),
                                     mode="constant",
                                     constant_values=0)
        observed = aug.augment_heatmaps([ia.HeatmapsOnImage(heatmaps_arr, shape=base_img.shape)])[0]
        assert observed.shape == base_img_padded.shape
        assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
        assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
        assert np.array_equal(observed.get_arr(), heatmaps_arr_padded)

    # test pad by range of pixels
    pads = [
        ((0, 2), 0, 0, 0),
        (0, (0, 2), 0, 0),
        (0, 0, (0, 2), 0),
        (0, 0, 0, (0, 2)),
    ]
    for pad in pads:
        top, right, bottom, left = pad
        aug = iaa.Pad(px=pad, keep_size=False)
        aug_det = aug.to_deterministic()

        images_padded = []
        keypoints_padded = []
        top_range = top if isinstance(top, tuple) else (top, top)
        right_range = right if isinstance(right, tuple) else (right, right)
        bottom_range = bottom if isinstance(bottom, tuple) else (bottom, bottom)
        left_range = left if isinstance(left, tuple) else (left, left)
        for top_val in sm.xrange(top_range[0], top_range[1]+1):
            for right_val in sm.xrange(right_range[0], right_range[1]+1):
                for bottom_val in sm.xrange(bottom_range[0], bottom_range[1]+1):
                    for left_val in sm.xrange(left_range[0], left_range[1]+1):
                        images_padded.append(
                            np.pad(base_img, ((top_val, bottom_val), (left_val, right_val), (0, 0)),
                                   mode="constant", constant_values=0)
                        )
                        keypoints_padded.append(keypoints[0].shift(x=left_val, y=top_val))

        movements = []
        movements_det = []
        for i in sm.xrange(100):
            observed = aug.augment_images(images)

            matches = [1 if np.array_equal(observed, np.array([base_img_padded])) else 0
                       for base_img_padded in images_padded]
            movements.append(np.argmax(np.array(matches)))
            assert any([val == 1 for val in matches])

            observed = aug_det.augment_images(images)
            matches = [1 if np.array_equal(observed, np.array([base_img_padded])) else 0
                       for base_img_padded in images_padded]
            movements_det.append(np.argmax(np.array(matches)))
            assert any([val == 1 for val in matches])

            observed = aug.augment_images(images_list)
            assert any([array_equal_lists(observed, [base_img_padded])
                        for base_img_padded in images_padded])

            observed = aug.augment_keypoints(keypoints)
            assert any([keypoints_equal(observed, [kp]) for kp in keypoints_padded])

        assert len(set(movements)) == 3
        assert len(set(movements_det)) == 1

    # test pad by list of exact pixel values
    pads = [
        ([0, 2], 0, 0, 0),
        (0, [0, 2], 0, 0),
        (0, 0, [0, 2], 0),
        (0, 0, 0, [0, 2]),
    ]
    for pad in pads:
        top, right, bottom, left = pad
        aug = iaa.Pad(px=pad, keep_size=False)
        aug_det = aug.to_deterministic()

        images_padded = []
        keypoints_padded = []
        top_range = top if isinstance(top, list) else [top]
        right_range = right if isinstance(right, list) else [right]
        bottom_range = bottom if isinstance(bottom, list) else [bottom]
        left_range = left if isinstance(left, list) else [left]
        for top_val in top_range:
            for right_val in right_range:
                for bottom_val in bottom_range:
                    for left_val in left_range:
                        images_padded.append(
                            np.pad(base_img, ((top_val, bottom_val), (left_val, right_val), (0, 0)), mode="constant",
                                   constant_values=0)
                        )
                        keypoints_padded.append(keypoints[0].shift(x=left_val, y=top_val))

        movements = []
        movements_det = []
        for i in sm.xrange(100):
            observed = aug.augment_images(images)
            matches = [1 if np.array_equal(observed, np.array([base_img_padded])) else 0
                       for base_img_padded in images_padded]
            movements.append(np.argmax(np.array(matches)))
            assert any([val == 1 for val in matches])

            observed = aug_det.augment_images(images)
            matches = [1 if np.array_equal(observed, np.array([base_img_padded])) else 0
                       for base_img_padded in images_padded]
            movements_det.append(np.argmax(np.array(matches)))
            assert any([val == 1 for val in matches])

            observed = aug.augment_images(images_list)
            assert any([array_equal_lists(observed, [base_img_padded])
                        for base_img_padded in images_padded])

            observed = aug.augment_keypoints(keypoints)
            assert any([keypoints_equal(observed, [kp]) for kp in keypoints_padded])

        assert len(set(movements)) == 2
        assert len(set(movements_det)) == 1

    # pad smaller heatmaps
    # heatmap is (6, 4), image is (6, 16)
    # image is padded by (2, 4, 2, 4)
    # expected image size: (10, 24)
    # expected heatmap size: (10, 6)
    aug = iaa.Pad(px=(2, 4, 2, 4), keep_size=False)
    heatmaps_arr_small = np.float32([
        [0, 0, 0, 0],
        [0, 1.0, 1.0, 0],
        [0, 1.0, 1.0, 0],
        [0, 1.0, 1.0, 0],
        [0, 1.0, 1.0, 0],
        [0, 0, 0, 0]
    ])
    top, bottom, left, right = 2, 2, 1, 1
    heatmaps_arr_small_padded = np.pad(heatmaps_arr_small, ((top, bottom), (left, right)),
                                       mode="constant",
                                       constant_values=0)
    observed = aug.augment_heatmaps([ia.HeatmapsOnImage(heatmaps_arr_small, shape=(6, 16))])[0]
    assert observed.shape == (10, 24)
    assert observed.arr_0to1.shape == (10, 6, 1)
    assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
    assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
    assert np.allclose(observed.arr_0to1[..., 0], heatmaps_arr_small_padded)

    # pad smaller heatmaps, with keep_size=True
    # heatmap is (6, 4), image is (6, 16)
    # image is padded by (2, 4, 2, 4)
    # expected image size: (10, 24) -> (6, 16) after resize
    # expected heatmap size: (10, 6) -> (6, 4) after resize
    aug = iaa.Pad(px=(2, 4, 2, 4), keep_size=True)
    heatmaps_arr_small = np.float32([
        [0, 0, 0, 0],
        [0, 1.0, 1.0, 0],
        [0, 1.0, 1.0, 0],
        [0, 1.0, 1.0, 0],
        [0, 1.0, 1.0, 0],
        [0, 0, 0, 0]
    ])
    top, bottom, left, right = 2, 2, 1, 1
    heatmaps_arr_small_padded = np.pad(heatmaps_arr_small, ((top, bottom), (left, right)),
                                       mode="constant",
                                       constant_values=0)
    observed = aug.augment_heatmaps([ia.HeatmapsOnImage(heatmaps_arr_small, shape=(6, 16))])[0]
    assert observed.shape == (6, 16)
    assert observed.arr_0to1.shape == (6, 4, 1)
    assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
    assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
    assert np.allclose(
        observed.arr_0to1[..., 0],
        np.clip(
            ia.imresize_single_image(heatmaps_arr_small_padded, (6, 4), interpolation="cubic"),
            0, 1.0
        )
    )

    # keypoints
    kpsoi = ia.KeypointsOnImage([ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=0)], shape=(4, 4, 3))
    kpsoi_aug = iaa.Pad((2, 0, 4, 4), keep_size=False).augment_keypoints([kpsoi])[0]
    assert kpsoi_aug.shape == (10, 8, 3)
    assert len(kpsoi_aug.keypoints) == 2
    assert np.allclose(kpsoi_aug.keypoints[0].x, 4+1)
    assert np.allclose(kpsoi_aug.keypoints[0].y, 2+2)
    assert np.allclose(kpsoi_aug.keypoints[1].x, 4+3)
    assert np.allclose(kpsoi_aug.keypoints[1].y, 2+0)

    # keypoints, with keep_size=True
    kpsoi = ia.KeypointsOnImage([ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=0)], shape=(4, 4, 3))
    kpsoi_aug = iaa.Pad((2, 0, 4, 4), keep_size=True).augment_keypoints([kpsoi])[0]
    assert kpsoi_aug.shape == (4, 4, 3)
    assert len(kpsoi_aug.keypoints) == 2
    assert np.allclose(kpsoi_aug.keypoints[0].x, ((4+1)/8)*4)
    assert np.allclose(kpsoi_aug.keypoints[0].y, ((2+2)/10)*4)
    assert np.allclose(kpsoi_aug.keypoints[1].x, ((4+3)/8)*4)
    assert np.allclose(kpsoi_aug.keypoints[1].y, ((2+0)/10)*4)

    # polygons
    aug = iaa.Pad((2, 0, 4, 4), keep_size=False)
    psoi = ia.PolygonsOnImage([
        ia.Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
        ia.Polygon([(1, 1), (5, 1), (5, 5), (1, 5)])
    ], shape=(4, 4, 3))
    psoi_aug = aug.augment_polygons([psoi, psoi])
    assert len(psoi_aug) == 2
    for psoi_aug_i in psoi_aug:
        assert psoi_aug_i.shape == (10, 8, 3)
        assert len(psoi_aug_i.polygons) == 2
        assert psoi_aug_i.polygons[0].exterior_almost_equals(
            ia.Polygon([(4, 2), (8, 2), (8, 6), (4, 6)])
        )
        assert psoi_aug_i.polygons[1].exterior_almost_equals(
            ia.Polygon([(5, 3), (9, 3), (9, 7), (5, 7)])
        )

    # polygons, with keep_size=True
    aug = iaa.Pad((2, 0, 4, 4), keep_size=True)
    psoi = ia.PolygonsOnImage([
        ia.Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
        ia.Polygon([(1, 1), (5, 1), (5, 5), (1, 5)])
    ], shape=(4, 4, 3))
    psoi_aug = aug.augment_polygons([psoi, psoi])
    assert len(psoi_aug) == 2
    for psoi_aug_i in psoi_aug:
        assert psoi_aug_i.shape == (4, 4, 3)
        assert len(psoi_aug_i.polygons) == 2
        assert psoi_aug_i.polygons[0].exterior_almost_equals(
            ia.Polygon([(4*(4/8), 4*(2/10)),
                        (4*(8/8), 4*(2/10)),
                        (4*(8/8), 4*(6/10)),
                        (4*(4/8), 4*(6/10))])
        )
        assert psoi_aug_i.polygons[1].exterior_almost_equals(
            ia.Polygon([(4*(5/8), 4*(3/10)),
                        (4*(9/8), 4*(3/10)),
                        (4*(9/8), 4*(7/10)),
                        (4*(5/8), 4*(7/10))])
        )

    # pad modes
    image = np.zeros((1, 2), dtype=np.uint8)
    image[0, 0] = 100
    image[0, 1] = 50
    aug = iaa.Pad(px=(0, 1, 0, 0), pad_mode=iap.Choice(["constant", "maximum", "edge"]), pad_cval=0, keep_size=False)
    seen = [0, 0, 0]
    for _ in sm.xrange(300):
        observed = aug.augment_image(image)
        if observed[0, 2] == 0:
            seen[0] += 1
        elif observed[0, 2] == 100:
            seen[1] += 1
        elif observed[0, 2] == 50:
            seen[2] += 1
        else:
            assert False
    assert all([100 - 50 < v < 100 + 50 for v in seen])

    aug = iaa.Pad(px=(0, 1, 0, 0), pad_mode=ia.ALL, pad_cval=0, keep_size=False)
    expected = ["constant", "edge", "linear_ramp", "maximum", "mean", "median", "minimum", "reflect",
                "symmetric", "wrap"]
    assert isinstance(aug.pad_mode, iap.Choice)
    assert len(aug.pad_mode.a) == len(expected)
    assert all([v in aug.pad_mode.a for v in expected])

    aug = iaa.Pad(px=(0, 1, 0, 0), pad_mode=["constant", "maximum"], pad_cval=0, keep_size=False)
    expected = ["constant", "maximum"]
    assert isinstance(aug.pad_mode, iap.Choice)
    assert len(aug.pad_mode.a) == len(expected)
    assert all([v in aug.pad_mode.a for v in expected])

    got_exception = False
    try:
        _aug = iaa.Pad(px=(0, 1, 0, 0), pad_mode=False, pad_cval=0, keep_size=False)
    except Exception as exc:
        assert "Expected pad_mode to be " in str(exc)
        got_exception = True
    assert got_exception

    # pad modes, heatmaps
    heatmaps = ia.HeatmapsOnImage(np.ones((3, 3, 1), dtype=np.float32), shape=(3, 3, 3))
    aug = iaa.Pad(px=(0, 1, 0, 0), pad_mode="edge", pad_cval=0, keep_size=False)
    observed = aug.augment_heatmaps([heatmaps])[0]
    assert np.sum(observed.get_arr() <= 1e-4) == 3

    # pad cvals
    aug = iaa.Pad(px=(0, 1, 0, 0), pad_mode="constant", pad_cval=100, keep_size=False)
    observed = aug.augment_image(np.zeros((1, 1), dtype=np.uint8))
    assert observed[0, 0] == 0
    assert observed[0, 1] == 100

    image = np.zeros((1, 1), dtype=np.uint8)
    aug = iaa.Pad(px=(0, 1, 0, 0), pad_mode="constant", pad_cval=iap.Choice([50, 100]), keep_size=False)
    seen = [0, 0]
    for _ in sm.xrange(200):
        observed = aug.augment_image(image)
        if observed[0, 1] == 50:
            seen[0] += 1
        elif observed[0, 1] == 100:
            seen[1] += 1
        else:
            assert False
    assert all([100 - 50 < v < 100 + 50 for v in seen])

    aug = iaa.Pad(px=(0, 1, 0, 0), pad_mode="constant", pad_cval=[50, 100], keep_size=False)
    expected = [50, 100]
    assert isinstance(aug.pad_cval, iap.Choice)
    assert len(aug.pad_cval.a) == len(expected)
    assert all([v in aug.pad_cval.a for v in expected])

    image = np.zeros((1, 1), dtype=np.uint8)
    aug = iaa.Pad(px=(0, 1, 0, 0), pad_mode="constant", pad_cval=(50, 52), keep_size=False)
    seen = [0, 0, 0]
    for _ in sm.xrange(300):
        observed = aug.augment_image(image)
        if observed[0, 1] == 50:
            seen[0] += 1
        elif observed[0, 1] == 51:
            seen[1] += 1
        elif observed[0, 1] == 52:
            seen[2] += 1
        else:
            assert False
    assert all([100 - 50 < v < 100 + 50 for v in seen])

    got_exception = False
    try:
        aug = iaa.Pad(px=(0, 1, 0, 0), pad_mode="constant", pad_cval="test", keep_size=False)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # pad cvals, heatmaps
    heatmaps = ia.HeatmapsOnImage(np.zeros((3, 3, 1), dtype=np.float32), shape=(3, 3, 3))
    aug = iaa.Pad(px=(0, 1, 0, 0), pad_mode="constant", pad_cval=255, keep_size=False)
    observed = aug.augment_heatmaps([heatmaps])[0]
    assert np.sum(observed.get_arr() > 1e-4) == 0

    # ------------------
    # pad by percentages
    # ------------------
    # pad all sides by 100%
    aug = iaa.Pad(percent=1.0, keep_size=False)
    observed = aug.augment_image(np.zeros((4, 4), dtype=np.uint8) + 1)
    assert observed.shape == (4+4+4, 4+4+4)
    assert np.sum(observed[4:-4, 4:-4]) == 4*4
    assert np.sum(observed) == 4*4

    # pad all sides by StochasticParameter
    aug = iaa.Pad(percent=iap.Deterministic(1.0), keep_size=False)
    observed = aug.augment_image(np.zeros((4, 4), dtype=np.uint8) + 1)
    assert observed.shape == (4+4+4, 4+4+4)
    assert np.sum(observed[4:-4, 4:-4]) == 4*4
    assert np.sum(observed) == 4*4

    # pad all sides by 100-200%
    aug = iaa.Pad(percent=(1.0, 2.0), sample_independently=False, keep_size=False)
    observed = aug.augment_image(np.zeros((4, 4), dtype=np.uint8) + 1)
    assert np.sum(observed) == 4*4
    assert (observed.shape[0] - 4) % 2 == 0
    assert (observed.shape[1] - 4) % 2 == 0

    # pad by invalid value
    got_exception = False
    try:
        _ = iaa.Pad(percent="test", keep_size=False)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # test pad by 100% on each side
    image = np.zeros((4, 4), dtype=np.uint8)
    image[0, 0] = 255
    image[3, 0] = 255
    image[0, 3] = 255
    image[3, 3] = 255
    height, width = image.shape[0:2]
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=3, y=3),
                                      ia.Keypoint(x=3, y=3)], shape=image.shape)]
    pads = [
        (1.0, 0, 0, 0),
        (0, 1.0, 0, 0),
        (0, 0, 1.0, 0),
        (0, 0, 0, 1.0),
    ]
    for pad in pads:
        top, right, bottom, left = pad
        top_px = int(top * height)
        right_px = int(right * width)
        bottom_px = int(bottom * height)
        left_px = int(left * width)
        aug = iaa.Pad(percent=pad, keep_size=False)
        image_padded = np.pad(image, ((top_px, bottom_px), (left_px, right_px)),
                              mode="constant",
                              constant_values=0)
        observed = aug.augment_image(image)
        assert np.array_equal(observed, image_padded)

        keypoints_moved = [keypoints[0].shift(x=left_px, y=top_px)]
        observed = aug.augment_keypoints(keypoints)
        assert keypoints_equal(observed, keypoints_moved)

    # pad smaller heatmaps
    # heatmap is (6, 4), image is (6, 16)
    # image is padded by (0.5, 0.25, 0.5, 0.25)
    # expected image size: (12, 24)
    # expected heatmap size: (12, 6)
    aug = iaa.Pad(percent=(0.5, 0.25, 0.5, 0.25), keep_size=False)
    heatmaps_arr_small = np.float32([
        [0, 0, 0, 0],
        [0, 1.0, 1.0, 0],
        [0, 1.0, 1.0, 0],
        [0, 1.0, 1.0, 0],
        [0, 1.0, 1.0, 0],
        [0, 0, 0, 0]
    ])
    top, bottom, left, right = 3, 3, 1, 1
    heatmaps_arr_small_padded = np.pad(heatmaps_arr_small, ((top, bottom), (left, right)),
                                       mode="constant",
                                       constant_values=0)
    observed = aug.augment_heatmaps([ia.HeatmapsOnImage(heatmaps_arr_small, shape=(6, 16))])[0]
    assert observed.shape == (12, 24)
    assert observed.arr_0to1.shape == (12, 6, 1)
    assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
    assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
    assert np.allclose(observed.arr_0to1[..., 0], heatmaps_arr_small_padded)

    # pad smaller heatmaps, with keep_size=True
    # heatmap is (6, 4), image is (6, 16)
    # image is padded by (0.5, 0.25, 0.5, 0.25)
    # expected image size: (12, 24) -> (6, 16) after resize
    # expected heatmap size: (12, 6) -> (6, 4) after resize
    aug = iaa.Pad(percent=(0.5, 0.25, 0.5, 0.25), keep_size=True)
    heatmaps_arr_small = np.float32([
        [0, 0, 0, 0],
        [0, 1.0, 1.0, 0],
        [0, 1.0, 1.0, 0],
        [0, 1.0, 1.0, 0],
        [0, 1.0, 1.0, 0],
        [0, 0, 0, 0]
    ])
    top, bottom, left, right = 3, 3, 1, 1
    heatmaps_arr_small_padded = np.pad(heatmaps_arr_small, ((top, bottom), (left, right)),
                                       mode="constant",
                                       constant_values=0)
    observed = aug.augment_heatmaps([ia.HeatmapsOnImage(heatmaps_arr_small, shape=(6, 16))])[0]
    assert observed.shape == (6, 16)
    assert observed.arr_0to1.shape == (6, 4, 1)
    assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
    assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
    assert np.allclose(
        observed.arr_0to1[..., 0],
        np.clip(
            ia.imresize_single_image(heatmaps_arr_small_padded, (6, 4), interpolation="cubic"),
            0, 1.0
        )
    )

    # keypoints
    kpsoi = ia.KeypointsOnImage([ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=0)], shape=(4, 4, 3))
    kpsoi_aug = iaa.Pad(percent=(0.5, 0, 1.0, 1.0), keep_size=False).augment_keypoints([kpsoi])[0]
    assert kpsoi_aug.shape == (10, 8, 3)
    assert len(kpsoi_aug.keypoints) == 2
    assert np.allclose(kpsoi_aug.keypoints[0].x, 4+1)
    assert np.allclose(kpsoi_aug.keypoints[0].y, 2+2)
    assert np.allclose(kpsoi_aug.keypoints[1].x, 4+3)
    assert np.allclose(kpsoi_aug.keypoints[1].y, 2+0)

    # keypoints, with keep_size=True
    kpsoi = ia.KeypointsOnImage([ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=0)], shape=(4, 4, 3))
    kpsoi_aug = iaa.Pad(percent=(0.5, 0, 1.0, 1.0), keep_size=True).augment_keypoints([kpsoi])[0]
    assert kpsoi_aug.shape == (4, 4, 3)
    assert len(kpsoi_aug.keypoints) == 2
    assert np.allclose(kpsoi_aug.keypoints[0].x, ((4+1)/8)*4)
    assert np.allclose(kpsoi_aug.keypoints[0].y, ((2+2)/10)*4)
    assert np.allclose(kpsoi_aug.keypoints[1].x, ((4+3)/8)*4)
    assert np.allclose(kpsoi_aug.keypoints[1].y, ((2+0)/10)*4)

    # polygons
    aug = iaa.Pad(percent=(0.5, 0, 1.0, 1.0), keep_size=False)
    psoi = ia.PolygonsOnImage([
        ia.Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
        ia.Polygon([(1, 1), (5, 1), (5, 5), (1, 5)])
    ], shape=(4, 4, 3))
    psoi_aug = aug.augment_polygons([psoi, psoi])
    assert len(psoi_aug) == 2
    for psoi_aug_i in psoi_aug:
        assert psoi_aug_i.shape == (10, 8, 3)
        assert len(psoi_aug_i.polygons) == 2
        assert psoi_aug_i.polygons[0].exterior_almost_equals(
            ia.Polygon([(4, 2), (8, 2), (8, 6), (4, 6)])
        )
        assert psoi_aug_i.polygons[1].exterior_almost_equals(
            ia.Polygon([(5, 3), (9, 3), (9, 7), (5, 7)])
        )

    # polygons, with keep_size=True
    aug = iaa.Pad(percent=(0.5, 0, 1.0, 1.0), keep_size=True)
    psoi = ia.PolygonsOnImage([
        ia.Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
        ia.Polygon([(1, 1), (5, 1), (5, 5), (1, 5)])
    ], shape=(4, 4, 3))
    psoi_aug = aug.augment_polygons([psoi, psoi])
    assert len(psoi_aug) == 2
    for psoi_aug_i in psoi_aug:
        assert psoi_aug_i.shape == (4, 4, 3)
        assert len(psoi_aug_i.polygons) == 2
        assert psoi_aug_i.polygons[0].exterior_almost_equals(
            ia.Polygon([(4*(4/8), 4*(2/10)),
                        (4*(8/8), 4*(2/10)),
                        (4*(8/8), 4*(6/10)),
                        (4*(4/8), 4*(6/10))])
        )
        assert psoi_aug_i.polygons[1].exterior_almost_equals(
            ia.Polygon([(4*(5/8), 4*(3/10)),
                        (4*(9/8), 4*(3/10)),
                        (4*(9/8), 4*(7/10)),
                        (4*(5/8), 4*(7/10))])
        )

    # test pad by range of percentages
    aug = iaa.Pad(percent=((0, 1.0), 0, 0, 0), keep_size=False)
    seen = [0, 0, 0, 0, 0]
    for _ in sm.xrange(500):
        observed = aug.augment_image(np.zeros((4, 4), dtype=np.uint8) + 255)
        n_padded = 0
        while np.all(observed[0, :] == 0):
            n_padded += 1
            observed = observed[1:, :]
        seen[n_padded] += 1
    # note that we cant just check for 100-50 < x < 100+50 here. The first and last value (0px
    # and 4px) padding have half the probability of occuring compared to the other values.
    # E.g. 0px is padded if sampled p falls in range [0, 0.125). 1px is padded if sampled p
    # falls in range [0.125, 0.375].
    assert all([v > 30 for v in seen])

    aug = iaa.Pad(percent=(0, (0, 1.0), 0, 0), keep_size=False)
    seen = [0, 0, 0, 0, 0]
    for _ in sm.xrange(500):
        observed = aug.augment_image(np.zeros((4, 4), dtype=np.uint8) + 255)
        n_padded = 0
        while np.all(observed[:, -1] == 0):
            n_padded += 1
            observed = observed[:, 0:-1]
        seen[n_padded] += 1
    assert all([v > 30 for v in seen])

    # test pad by list of percentages
    aug = iaa.Pad(percent=([0.0, 1.0], 0, 0, 0), keep_size=False)
    seen = [0, 0, 0, 0, 0]
    for _ in sm.xrange(500):
        observed = aug.augment_image(np.zeros((4, 4), dtype=np.uint8) + 255)
        n_padded = 0
        while np.all(observed[0, :] == 0):
            n_padded += 1
            observed = observed[1:, :]
        seen[n_padded] += 1
    assert 250 - 50 < seen[0] < 250 + 50
    assert seen[1] == 0
    assert seen[2] == 0
    assert seen[3] == 0
    assert 250 - 50 < seen[4] < 250 + 50

    aug = iaa.Pad(percent=(0, [0.0, 1.0], 0, 0), keep_size=False)
    seen = [0, 0, 0, 0, 0]
    for _ in sm.xrange(500):
        observed = aug.augment_image(np.zeros((4, 4), dtype=np.uint8) + 255)
        n_padded = 0
        while np.all(observed[:, -1] == 0):
            n_padded += 1
            observed = observed[:, 0:-1]
        seen[n_padded] += 1
    assert 250 - 50 < seen[0] < 250 + 50
    assert seen[1] == 0
    assert seen[2] == 0
    assert seen[3] == 0
    assert 250 - 50 < seen[4] < 250 + 50

    ###################
    # test other dtypes
    ###################
    aug = iaa.Pad(px=(1, 0, 0, 0), keep_size=False)
    mask = np.zeros((4, 3), dtype=bool)
    mask[2, 1] = True

    # bool
    image = np.zeros((3, 3), dtype=bool)
    image[1, 1] = True
    image_aug = aug.augment_image(image)
    assert image_aug.dtype == image.dtype
    assert image_aug.shape == (4, 3)
    assert np.all(image_aug[~mask] == 0)
    assert np.all(image_aug[mask] == 1)

    # uint, int
    for dtype in [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64]:
        min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)

        if np.dtype(dtype).kind == "i":
            values = [1, 5, 10, 100, int(0.1 * max_value), int(0.2 * max_value),
                      int(0.5 * max_value), max_value - 100, max_value]
            values = values + [(-1) * value for value in values]
        else:
            values = [1, 5, 10, 100, int(center_value), int(0.1 * max_value), int(0.2 * max_value),
                      int(0.5 * max_value), max_value - 100, max_value]

        for value in values:
            image = np.zeros((3, 3), dtype=dtype)
            image[1, 1] = value
            image_aug = aug.augment_image(image)
            assert image_aug.dtype == np.dtype(dtype)
            assert image_aug.shape == (4, 3)
            assert np.all(image_aug[~mask] == 0)
            assert np.all(image_aug[mask] == value)

    # float
    for dtype in [np.float16, np.float32, np.float64, np.float128]:
        min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)

        def _isclose(a, b):
            atol = 1e-4 if dtype == np.float16 else 1e-8
            return np.isclose(a, b, atol=atol, rtol=0)

        isize = np.dtype(dtype).itemsize
        values = [0.01, 1.0, 10.0, 100.0, 500 ** (isize - 1), 1000 ** (isize - 1)]
        values = values + [(-1) * value for value in values]
        values = values + [min_value, max_value]
        for value in values:
            image = np.zeros((3, 3), dtype=dtype)
            image[1, 1] = value
            image_aug = aug.augment_image(image)
            assert image_aug.dtype == np.dtype(dtype)
            assert image_aug.shape == (4, 3)
            assert np.all(_isclose(image_aug[~mask], 0))
            assert np.all(_isclose(image_aug[mask], np.float128(value)))


def test_Crop():
    reseed()

    base_img = np.array([[0, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0]], dtype=np.uint8)
    base_img = base_img[:, :, np.newaxis]

    images = np.array([base_img])
    images_list = [base_img]

    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1),
                                      ia.Keypoint(x=2, y=2)], shape=base_img.shape)]

    heatmaps_arr = np.float32([[0, 0, 0],
                               [0, 1.0, 0],
                               [0, 0, 0]])

    # test crop by 1 pixel on each side
    crops = [
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1),
    ]
    for crop in crops:
        top, right, bottom, left = crop
        height, width = base_img.shape[0:2]
        aug = iaa.Crop(px=crop, keep_size=False)
        base_img_cropped = base_img[top:height-bottom, left:width-right, :]
        observed = aug.augment_images(images)
        assert np.array_equal(observed, np.array([base_img_cropped]))

        observed = aug.augment_images(images_list)
        assert array_equal_lists(observed, [base_img_cropped])

        keypoints_moved = [keypoints[0].shift(x=-left, y=-top)]
        observed = aug.augment_keypoints(keypoints)
        assert keypoints_equal(observed, keypoints_moved)

        height, width = heatmaps_arr.shape[0:2]
        aug = iaa.Crop(px=crop, keep_size=False)
        heatmaps_arr_cropped = heatmaps_arr[top:height-bottom, left:width-right]
        observed = aug.augment_heatmaps([ia.HeatmapsOnImage(heatmaps_arr, shape=base_img.shape)])[0]
        assert observed.shape == base_img_cropped.shape
        assert np.array_equal(observed.get_arr(), heatmaps_arr_cropped)

    # test crop by range of pixels
    crops = [
        ((0, 2), 0, 0, 0),
        (0, (0, 2), 0, 0),
        (0, 0, (0, 2), 0),
        (0, 0, 0, (0, 2)),
    ]
    for crop in crops:
        top, right, bottom, left = crop
        height, width = base_img.shape[0:2]
        aug = iaa.Crop(px=crop, keep_size=False)
        aug_det = aug.to_deterministic()

        images_cropped = []
        keypoints_cropped = []
        top_range = top if isinstance(top, tuple) else (top, top)
        right_range = right if isinstance(right, tuple) else (right, right)
        bottom_range = bottom if isinstance(bottom, tuple) else (bottom, bottom)
        left_range = left if isinstance(left, tuple) else (left, left)
        for top_val in sm.xrange(top_range[0], top_range[1]+1):
            for right_val in sm.xrange(right_range[0], right_range[1]+1):
                for bottom_val in sm.xrange(bottom_range[0], bottom_range[1]+1):
                    for left_val in sm.xrange(left_range[0], left_range[1]+1):

                        images_cropped.append(base_img[top_val:height-bottom_val, left_val:width-right_val, :])
                        keypoints_cropped.append(keypoints[0].shift(x=-left_val, y=-top_val))

        movements = []
        movements_det = []
        for i in sm.xrange(100):
            observed = aug.augment_images(images)

            matches = [1 if np.array_equal(observed, np.array([base_img_cropped])) else 0
                       for base_img_cropped in images_cropped]
            movements.append(np.argmax(np.array(matches)))
            assert any([val == 1 for val in matches])

            observed = aug_det.augment_images(images)
            matches = [1 if np.array_equal(observed, np.array([base_img_cropped])) else 0
                       for base_img_cropped in images_cropped]
            movements_det.append(np.argmax(np.array(matches)))
            assert any([val == 1 for val in matches])

            observed = aug.augment_images(images_list)
            assert any([array_equal_lists(observed, [base_img_cropped])
                        for base_img_cropped in images_cropped])

            observed = aug.augment_keypoints(keypoints)
            assert any([keypoints_equal(observed, [kp]) for kp in keypoints_cropped])

        assert len(set(movements)) == 3
        assert len(set(movements_det)) == 1

    # test crop by list of exact pixel values
    crops = [
        ([0, 2], 0, 0, 0),
        (0, [0, 2], 0, 0),
        (0, 0, [0, 2], 0),
        (0, 0, 0, [0, 2]),
    ]
    for crop in crops:
        top, right, bottom, left = crop
        height, width = base_img.shape[0:2]
        aug = iaa.Crop(px=crop, keep_size=False)
        aug_det = aug.to_deterministic()

        images_cropped = []
        keypoints_cropped = []
        top_range = top if isinstance(top, list) else [top]
        right_range = right if isinstance(right, list) else [right]
        bottom_range = bottom if isinstance(bottom, list) else [bottom]
        left_range = left if isinstance(left, list) else [left]
        for top_val in top_range:
            for right_val in right_range:
                for bottom_val in bottom_range:
                    for left_val in left_range:
                        images_cropped.append(base_img[top_val:height-bottom_val, left_val:width-right_val, :])
                        keypoints_cropped.append(keypoints[0].shift(x=-left_val, y=-top_val))

        movements = []
        movements_det = []
        for i in sm.xrange(100):
            observed = aug.augment_images(images)
            matches = [1 if np.array_equal(observed, np.array([base_img_cropped])) else 0
                       for base_img_cropped in images_cropped]
            movements.append(np.argmax(np.array(matches)))
            assert any([val == 1 for val in matches])

            observed = aug_det.augment_images(images)
            matches = [1 if np.array_equal(observed, np.array([base_img_cropped])) else 0
                       for base_img_cropped in images_cropped]
            movements_det.append(np.argmax(np.array(matches)))
            assert any([val == 1 for val in matches])

            observed = aug.augment_images(images_list)
            assert any([array_equal_lists(observed, [base_img_cropped])
                        for base_img_cropped in images_cropped])

            observed = aug.augment_keypoints(keypoints)
            assert any([keypoints_equal(observed, [kp]) for kp in keypoints_cropped])

        assert len(set(movements)) == 2
        assert len(set(movements_det)) == 1

    # crop smaller heatmaps
    # heatmap is (6, 8), image is (6, 16)
    # image is cropped by (1, 4, 1, 4)
    # expected image size: (4, 8)
    # expected heatmap size: (4, 4)
    aug = iaa.Crop(px=(1, 4, 1, 4), keep_size=False)
    heatmaps_arr_small = np.zeros((6, 8), dtype=np.float32)
    heatmaps_arr_small[1:-1, 1:-1] = 1.0
    top, bottom, left, right = 1, 1, 2, 2
    heatmaps_arr_small_cropped = heatmaps_arr_small[top:-bottom, left:-right]
    observed = aug.augment_heatmaps([ia.HeatmapsOnImage(heatmaps_arr_small, shape=(6, 16))])[0]
    assert observed.shape == (4, 8)
    assert observed.arr_0to1.shape == (4, 4, 1)
    assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
    assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
    assert np.allclose(observed.arr_0to1[..., 0], heatmaps_arr_small_cropped)

    # crop smaller heatmaps, with keep_size=True
    # heatmap is (6, 8), image is (6, 16)
    # image is cropped by (1, 4, 1, 4)
    # expected image size: (4, 8) -> (6, 16) after resize
    # expected heatmap size: (4, 4) -> (6, 4) after resize
    aug = iaa.Crop(px=(1, 4, 1, 4), keep_size=True)
    heatmaps_arr_small = np.zeros((6, 8), dtype=np.float32)
    heatmaps_arr_small[1:-1, 1:-1] = 1.0
    top, bottom, left, right = 1, 1, 2, 2
    heatmaps_arr_small_cropped = heatmaps_arr_small[top:-bottom, left:-right]
    observed = aug.augment_heatmaps([ia.HeatmapsOnImage(heatmaps_arr_small, shape=(6, 16))])[0]
    assert observed.shape == (6, 16)
    assert observed.arr_0to1.shape == (6, 8, 1)
    assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
    assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
    assert np.allclose(
        observed.arr_0to1[..., 0],
        np.clip(
            ia.imresize_single_image(heatmaps_arr_small_cropped, (6, 8), interpolation="cubic"),
            0, 1.0
        )
    )

    # keypoints
    kpsoi = ia.KeypointsOnImage([ia.Keypoint(x=3, y=6), ia.Keypoint(x=8, y=5)], shape=(14, 14, 3))
    kpsoi_aug = iaa.Crop((1, 0, 4, 4), keep_size=False).augment_keypoints([kpsoi])[0]
    assert kpsoi_aug.shape == (9, 10, 3)
    assert len(kpsoi_aug.keypoints) == 2
    assert np.allclose(kpsoi_aug.keypoints[0].x, 3-4)
    assert np.allclose(kpsoi_aug.keypoints[0].y, 6-1)
    assert np.allclose(kpsoi_aug.keypoints[1].x, 8-4)
    assert np.allclose(kpsoi_aug.keypoints[1].y, 5-1)

    # keypoints, with keep_size=True
    kpsoi = ia.KeypointsOnImage([ia.Keypoint(x=3, y=6), ia.Keypoint(x=8, y=5)], shape=(14, 14, 3))
    kpsoi_aug = iaa.Crop((1, 0, 4, 4), keep_size=True).augment_keypoints([kpsoi])[0]
    assert kpsoi_aug.shape == (14, 14, 3)
    assert len(kpsoi_aug.keypoints) == 2
    assert np.allclose(kpsoi_aug.keypoints[0].x, ((3-4)/10)*14)
    assert np.allclose(kpsoi_aug.keypoints[0].y, ((6-1)/9)*14)
    assert np.allclose(kpsoi_aug.keypoints[1].x, ((8-4)/10)*14)
    assert np.allclose(kpsoi_aug.keypoints[1].y, ((5-1)/9)*14)

    # polygons
    aug = iaa.Crop((1, 0, 4, 4), keep_size=False)
    psoi = ia.PolygonsOnImage([
        ia.Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
        ia.Polygon([(1, 1), (5, 1), (5, 5), (1, 5)])
    ], shape=(10, 10, 3))
    psoi_aug = aug.augment_polygons([psoi, psoi])
    assert len(psoi_aug) == 2
    for psoi_aug_i in psoi_aug:
        assert psoi_aug_i.shape == (5, 6, 3)
        assert len(psoi_aug_i.polygons) == 2
        assert psoi_aug_i.polygons[0].exterior_almost_equals(
            ia.Polygon([(0-4, 0-1), (4-4, 0-1), (4-4, 4-1), (0-4, 4-1)])
        )
        assert psoi_aug_i.polygons[1].exterior_almost_equals(
            ia.Polygon([(1-4, 1-1), (5-4, 1-1), (5-4, 5-1), (1-4, 5-1)])
        )

    # polygons, with keep_size=True
    aug = iaa.Crop((1, 0, 4, 4), keep_size=True)
    psoi = ia.PolygonsOnImage([
        ia.Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
        ia.Polygon([(1, 1), (5, 1), (5, 5), (1, 5)])
    ], shape=(10, 10, 3))
    psoi_aug = aug.augment_polygons([psoi, psoi])
    assert len(psoi_aug) == 2
    for psoi_aug_i in psoi_aug:
        assert psoi_aug_i.shape == (10, 10, 3)
        assert len(psoi_aug_i.polygons) == 2
        assert psoi_aug_i.polygons[0].exterior_almost_equals(
            ia.Polygon([(10*(-4/6), 10*(-1/5)),
                        (10*(0/6), 10*(-1/5)),
                        (10*(0/6), 10*(3/5)),
                        (10*(-4/6), 10*(3/5))])
        )
        assert psoi_aug_i.polygons[1].exterior_almost_equals(
            ia.Polygon([(10*(-3/6), 10*(0/5)),
                        (10*(1/6), 10*(0/5)),
                        (10*(1/6), 10*(4/5)),
                        (10*(-3/6), 10*(4/5))])
        )

    # ------------------
    # crop by percentages
    # ------------------
    # crop all sides by 10%
    aug = iaa.Crop(percent=0.1, keep_size=False)
    image = np.random.randint(0, 255, size=(50, 50), dtype=np.uint8)
    observed = aug.augment_image(image)
    assert observed.shape == (40, 40)
    assert np.all(observed == image[5:-5, 5:-5])

    # crop all sides by StochasticParameter
    aug = iaa.Crop(percent=iap.Deterministic(0.1), keep_size=False)
    image = np.random.randint(0, 255, size=(50, 50), dtype=np.uint8)
    observed = aug.augment_image(image)
    assert observed.shape == (40, 40)
    assert np.all(observed == image[5:-5, 5:-5])

    # crop all sides by 10-20%
    image = np.random.randint(0, 255, size=(50, 50), dtype=np.uint8)
    aug = iaa.Crop(percent=(0.1, 0.2), keep_size=False)
    observed = aug.augment_image(image)
    assert 30 <= observed.shape[0] <= 40
    assert 30 <= observed.shape[1] <= 40

    # crop by invalid value
    got_exception = False
    try:
        _ = iaa.Crop(percent="test", keep_size=False)
    except Exception as exc:
        assert "Expected " in str(exc)
        got_exception = True
    assert got_exception

    # test crop by 10% on each side
    image = np.random.randint(0, 255, size=(50, 50), dtype=np.uint8)
    height, width = image.shape[0:2]
    keypoints = [ia.KeypointsOnImage([ia.Keypoint(x=10, y=11), ia.Keypoint(x=20, y=21),
                                      ia.Keypoint(x=30, y=31)], shape=image.shape)]
    crops = [
        (0.1, 0, 0, 0),
        (0, 0.1, 0, 0),
        (0, 0, 0.1, 0),
        (0, 0, 0, 0.1),
    ]
    for crop in crops:
        top, right, bottom, left = crop
        top_px = int(round(top * height))
        right_px = int(round(right * width))
        bottom_px = int(round(bottom * height))
        left_px = int(round(left * width))
        aug = iaa.Crop(percent=crop, keep_size=False)
        # dont use :-bottom_px and ;-right_px here, because these values can be 0
        image_cropped = image[top_px:50-bottom_px, left_px:50-right_px]
        observed = aug.augment_image(image)
        assert np.array_equal(observed, image_cropped)

        keypoints_moved = [keypoints[0].shift(x=-left_px, y=-top_px)]
        observed = aug.augment_keypoints(keypoints)
        assert keypoints_equal(observed, keypoints_moved)

    # crop smaller heatmaps
    # heatmap is (8, 12), image is (16, 32)
    # image is cropped by (0.25, 0.25, 0.25, 0.25)
    # expected image size: (8, 16)
    # expected heatmap size: (4, 6)
    aug = iaa.Crop(percent=(0.25, 0.25, 0.25, 0.25), keep_size=False)
    heatmaps_arr_small = np.zeros((8, 12), dtype=np.float32)
    heatmaps_arr_small[2:-2, 4:-4] = 1.0
    top, bottom, left, right = 2, 2, 3, 3
    heatmaps_arr_small_cropped = heatmaps_arr_small[top:-bottom, left:-right]
    observed = aug.augment_heatmaps([ia.HeatmapsOnImage(heatmaps_arr_small, shape=(16, 32))])[0]
    assert observed.shape == (8, 16)
    assert observed.arr_0to1.shape == (4, 6, 1)
    assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
    assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
    assert np.allclose(observed.arr_0to1[..., 0], heatmaps_arr_small_cropped)

    # crop smaller heatmaps, with keep_size=True
    # heatmap is (8, 12), image is (16, 32)
    # image is cropped by (0.25, 0.25, 0.25, 0.25)
    # expected image size: (8, 16) -> (16, 32) after resize
    # expected heatmap size: (4, 6) -> (8, 12) after resize
    aug = iaa.Crop(percent=(0.25, 0.25, 0.25, 0.25), keep_size=True)
    heatmaps_arr_small = np.zeros((8, 12), dtype=np.float32)
    heatmaps_arr_small[2:-2, 4:-4] = 1.0
    top, bottom, left, right = 2, 2, 3, 3
    heatmaps_arr_small_cropped = heatmaps_arr_small[top:-bottom, left:-right]
    observed = aug.augment_heatmaps([ia.HeatmapsOnImage(heatmaps_arr_small, shape=(16, 32))])[0]
    assert observed.shape == (16, 32)
    assert observed.arr_0to1.shape == (8, 12, 1)
    assert 0 - 1e-6 < observed.min_value < 0 + 1e-6
    assert 1 - 1e-6 < observed.max_value < 1 + 1e-6
    assert np.allclose(
        observed.arr_0to1[..., 0],
        np.clip(
            ia.imresize_single_image(heatmaps_arr_small_cropped, (8, 12), interpolation="cubic"),
            0, 1.0
        )
    )

    # keypoints
    kpsoi = ia.KeypointsOnImage([ia.Keypoint(x=12, y=10), ia.Keypoint(x=8, y=12)], shape=(16, 20, 3))
    kpsoi_aug = iaa.Crop(percent=(0.25, 0, 0.5, 0.1), keep_size=False).augment_keypoints([kpsoi])[0]
    assert kpsoi_aug.shape == (4, 18, 3)
    assert len(kpsoi_aug.keypoints) == 2
    assert np.allclose(kpsoi_aug.keypoints[0].x, 12-2)
    assert np.allclose(kpsoi_aug.keypoints[0].y, 10-4)
    assert np.allclose(kpsoi_aug.keypoints[1].x, 8-2)
    assert np.allclose(kpsoi_aug.keypoints[1].y, 12-4)

    # keypoints, with keep_size=True
    kpsoi = ia.KeypointsOnImage([ia.Keypoint(x=12, y=10), ia.Keypoint(x=8, y=12)], shape=(16, 20, 3))
    kpsoi_aug = iaa.Crop(percent=(0.25, 0, 0.5, 0.1), keep_size=True).augment_keypoints([kpsoi])[0]
    assert kpsoi_aug.shape == (16, 20, 3)
    assert len(kpsoi_aug.keypoints) == 2
    assert np.allclose(kpsoi_aug.keypoints[0].x, ((12-2)/18)*20)
    assert np.allclose(kpsoi_aug.keypoints[0].y, ((10-4)/4)*16)
    assert np.allclose(kpsoi_aug.keypoints[1].x, ((8-2)/18)*20)
    assert np.allclose(kpsoi_aug.keypoints[1].y, ((12-4)/4)*16)

    # polygons
    aug = iaa.Crop(percent=(0.2, 0, 0.5, 0.1), keep_size=False)
    psoi = ia.PolygonsOnImage([
        ia.Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
        ia.Polygon([(1, 1), (5, 1), (5, 5), (1, 5)])
    ], shape=(10, 10, 3))
    psoi_aug = aug.augment_polygons([psoi, psoi])
    assert len(psoi_aug) == 2
    for psoi_aug_i in psoi_aug:
        assert psoi_aug_i.shape == (3, 9, 3)
        assert len(psoi_aug_i.polygons) == 2
        assert psoi_aug_i.polygons[0].exterior_almost_equals(
            ia.Polygon([(0-1, 0-2), (4-1, 0-2), (4-1, 4-2), (0-1, 4-2)])
        )
        assert psoi_aug_i.polygons[1].exterior_almost_equals(
            ia.Polygon([(1-1, 1-2), (5-1, 1-2), (5-1, 5-2), (1-1, 5-2)])
        )

    # polygons, with keep_size=True
    aug = iaa.Crop(percent=(0.2, 0, 0.5, 0.1), keep_size=True)
    psoi = ia.PolygonsOnImage([
        ia.Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
        ia.Polygon([(1, 1), (5, 1), (5, 5), (1, 5)])
    ], shape=(10, 10, 3))
    psoi_aug = aug.augment_polygons([psoi, psoi])
    assert len(psoi_aug) == 2
    for psoi_aug_i in psoi_aug:
        assert psoi_aug_i.shape == (10, 10, 3)
        assert len(psoi_aug_i.polygons) == 2
        assert psoi_aug_i.polygons[0].exterior_almost_equals(
            ia.Polygon([(10*(-1/9), 10*(-2/3)),
                        (10*(3/9), 10*(-2/3)),
                        (10*(3/9), 10*(2/3)),
                        (10*(-1/9), 10*(2/3))])
        )
        assert psoi_aug_i.polygons[1].exterior_almost_equals(
            ia.Polygon([(10*(0/9), 10*(-1/3)),
                        (10*(4/9), 10*(-1/3)),
                        (10*(4/9), 10*(3/3)),
                        (10*(0/9), 10*(3/3))])
        )

    # test crop by range of percentages
    aug = iaa.Crop(percent=((0, 0.1), 0, 0, 0), keep_size=False)
    seen = [0, 0, 0, 0, 0]
    for _ in sm.xrange(500):
        observed = aug.augment_image(np.zeros((40, 40), dtype=np.uint8))
        n_cropped = 40 - observed.shape[0]
        seen[n_cropped] += 1
    # note that we cant just check for 100-50 < x < 100+50 here. The first and last value (0px
    # and 4px) have half the probability of occuring compared to the other values.
    # E.g. 0px is cropped if sampled p falls in range [0, 0.125). 1px is cropped if sampled p
    # falls in range [0.125, 0.375].
    assert all([v > 30 for v in seen])

    aug = iaa.Crop(percent=(0, (0, 0.1), 0, 0), keep_size=False)
    seen = [0, 0, 0, 0, 0]
    for _ in sm.xrange(500):
        observed = aug.augment_image(np.zeros((40, 40), dtype=np.uint8) + 255)
        n_cropped = 40 - observed.shape[1]
        seen[n_cropped] += 1
    assert all([v > 30 for v in seen])

    # test crop by list of percentages
    aug = iaa.Crop(percent=([0.0, 0.1], 0, 0, 0), keep_size=False)
    seen = [0, 0, 0, 0, 0]
    for _ in sm.xrange(500):
        observed = aug.augment_image(np.zeros((40, 40), dtype=np.uint8) + 255)
        n_cropped = 40 - observed.shape[0]
        seen[n_cropped] += 1
    assert 250 - 50 < seen[0] < 250 + 50
    assert seen[1] == 0
    assert seen[2] == 0
    assert seen[3] == 0
    assert 250 - 50 < seen[4] < 250 + 50

    aug = iaa.Crop(percent=(0, [0.0, 0.1], 0, 0), keep_size=False)
    seen = [0, 0, 0, 0, 0]
    for _ in sm.xrange(500):
        observed = aug.augment_image(np.zeros((40, 40), dtype=np.uint8) + 255)
        n_cropped = 40 - observed.shape[1]
        seen[n_cropped] += 1
    assert 250 - 50 < seen[0] < 250 + 50
    assert seen[1] == 0
    assert seen[2] == 0
    assert seen[3] == 0
    assert 250 - 50 < seen[4] < 250 + 50

    ###################
    # test other dtypes
    ###################
    aug = iaa.Crop(px=(1, 0, 0, 0), keep_size=False)
    mask = np.zeros((2, 3), dtype=bool)
    mask[0, 1] = True

    # bool
    image = np.zeros((3, 3), dtype=bool)
    image[1, 1] = True
    image_aug = aug.augment_image(image)
    assert image_aug.dtype == image.dtype
    assert image_aug.shape == (2, 3)
    assert np.all(image_aug[~mask] == 0)
    assert np.all(image_aug[mask] == 1)

    # uint, int
    for dtype in [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64]:
        min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)

        if np.dtype(dtype).kind == "i":
            values = [1, 5, 10, 100, int(0.1 * max_value), int(0.2 * max_value),
                      int(0.5 * max_value), max_value - 100, max_value]
            values = values + [(-1) * value for value in values]
        else:
            values = [1, 5, 10, 100, int(center_value), int(0.1 * max_value), int(0.2 * max_value),
                      int(0.5 * max_value), max_value - 100, max_value]

        for value in values:
            image = np.zeros((3, 3), dtype=dtype)
            image[1, 1] = value
            image_aug = aug.augment_image(image)
            assert image_aug.dtype == np.dtype(dtype)
            assert image_aug.shape == (2, 3)
            assert np.all(image_aug[~mask] == 0)
            assert np.all(image_aug[mask] == value)

    # float
    for dtype in [np.float16, np.float32, np.float64, np.float128]:
        min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)

        def _isclose(a, b):
            atol = 1e-4 if dtype == np.float16 else 1e-8
            return np.isclose(a, b, atol=atol, rtol=0)

        isize = np.dtype(dtype).itemsize
        values = [0.01, 1.0, 10.0, 100.0, 500 ** (isize - 1), 1000 ** (isize - 1)]
        values = values + [(-1) * value for value in values]
        values = values + [min_value, max_value]
        for value in values:
            image = np.zeros((3, 3), dtype=dtype)
            image[1, 1] = value
            image_aug = aug.augment_image(image)
            assert image_aug.dtype == np.dtype(dtype)
            assert image_aug.shape == (2, 3)
            assert np.all(_isclose(image_aug[~mask], 0))
            assert np.all(_isclose(image_aug[mask], np.float128(value)))


def test_PadToFixedSize():
    reseed()

    img = np.uint8([[255]])
    img3d = img[:, :, np.newaxis]
    img3d_rgb = np.tile(img3d, (1, 1, 3))

    # basic functionality
    aug = iaa.PadToFixedSize(height=5, width=5)
    observed = aug.augment_image(img)
    assert observed.dtype.type == np.uint8
    assert observed.shape == (5, 5)

    observed = aug.augment_image(img3d)
    assert observed.dtype.type == np.uint8
    assert observed.shape == (5, 5, 1)

    observed = aug.augment_image(img3d_rgb)
    assert observed.dtype.type == np.uint8
    assert observed.shape == (5, 5, 3)

    # test float32, float64, int32
    for dtype in [np.float32, np.float64, np.int32]:
        aug = iaa.PadToFixedSize(height=5, width=5)
        observed = aug.augment_image(img.astype(dtype))
        assert observed.dtype.type == dtype
        assert observed.shape == (5, 5)

    # change only one side when other side has already desired size
    aug = iaa.PadToFixedSize(height=5, width=5)
    observed = aug.augment_image(np.zeros((1, 5, 3), dtype=np.uint8))
    assert observed.dtype.type == np.uint8
    assert observed.shape == (5, 5, 3)

    aug = iaa.PadToFixedSize(height=5, width=5)
    observed = aug.augment_image(np.zeros((5, 1, 3), dtype=np.uint8))
    assert observed.dtype.type == np.uint8
    assert observed.shape == (5, 5, 3)

    # change no side when all sides have exactly desired size
    img5x5 = np.zeros((5, 5, 3), dtype=np.uint8)
    img5x5[2, 2, :] = 255
    aug = iaa.PadToFixedSize(height=5, width=5)
    observed = aug.augment_image(img5x5)
    assert observed.dtype.type == np.uint8
    assert observed.shape == (5, 5, 3)
    assert np.array_equal(observed, img5x5)

    # change no side when all sides have larger than desired size
    img6x6 = np.zeros((6, 6, 3), dtype=np.uint8)
    img6x6[3, 3, :] = 255
    aug = iaa.PadToFixedSize(height=5, width=5)
    observed = aug.augment_image(img6x6)
    assert observed.dtype.type == np.uint8
    assert observed.shape == (6, 6, 3)
    assert np.array_equal(observed, img6x6)

    # make sure that pad mode is recognized
    aug = iaa.PadToFixedSize(height=4, width=4, pad_mode="edge")
    aug.position = (iap.Deterministic(0.5), iap.Deterministic(0.5))
    img2x2 = np.uint8([
        [50, 100],
        [150, 200]
    ])
    expected = np.uint8([
        [50, 50, 100, 100],
        [50, 50, 100, 100],
        [150, 150, 200, 200],
        [150, 150, 200, 200]
    ])
    observed = aug.augment_image(img2x2)
    assert observed.dtype.type == np.uint8
    assert observed.shape == (4, 4)
    assert np.array_equal(observed, expected)

    # explicit non-center position test
    aug = iaa.PadToFixedSize(height=3, width=3, pad_mode="constant", pad_cval=128, position="left-top")
    img1x1 = np.uint8([[255]])
    observed = aug.augment_image(img1x1)
    expected = np.uint8([
        [128, 128, 128],
        [128, 128, 128],
        [128, 128, 255]
    ])
    assert observed.dtype.type == np.uint8
    assert observed.shape == (3, 3)
    assert np.array_equal(observed, expected)

    aug = iaa.PadToFixedSize(height=3, width=3, pad_mode="constant", pad_cval=128, position="right-bottom")
    img1x1 = np.uint8([[255]])
    observed = aug.augment_image(img1x1)
    expected = np.uint8([
        [255, 128, 128],
        [128, 128, 128],
        [128, 128, 128]
    ])
    assert observed.dtype.type == np.uint8
    assert observed.shape == (3, 3)
    assert np.array_equal(observed, expected)

    aug = iaa.PadToFixedSize(height=3, width=3, pad_mode="constant", pad_cval=128, position=(0.5, 1.0))
    img1x1 = np.uint8([[255]])
    observed = aug.augment_image(img1x1)
    expected = np.uint8([
        [128, 255, 128],
        [128, 128, 128],
        [128, 128, 128]
    ])
    assert observed.dtype.type == np.uint8
    assert observed.shape == (3, 3)
    assert np.array_equal(observed, expected)

    # basic keypoint test
    kpsoi = ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)], shape=(3, 3))
    aug = iaa.PadToFixedSize(height=4, width=4, pad_mode="edge", position="center")
    observed = aug.augment_keypoints([kpsoi])
    expected = ia.KeypointsOnImage([ia.Keypoint(x=2, y=2)], shape=(4, 4))
    assert observed[0].shape == expected.shape
    assert keypoints_equal(observed, [expected])

    # keypoint test with shape not being changed
    kpsoi = ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)], shape=(3, 3))
    aug = iaa.PadToFixedSize(height=3, width=3, pad_mode="edge", position="center")
    observed = aug.augment_keypoints([kpsoi])
    expected = ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)], shape=(3, 3))
    assert observed[0].shape == expected.shape
    assert keypoints_equal(observed, [expected])

    # keypoint test with explicit non-center position
    kpsoi = ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)], shape=(3, 3))
    aug = iaa.PadToFixedSize(height=4, width=4, pad_mode="edge", position="left-top")
    observed = aug.augment_keypoints([kpsoi])
    expected = ia.KeypointsOnImage([ia.Keypoint(x=2, y=2)], shape=(4, 4))
    assert observed[0].shape == expected.shape
    assert keypoints_equal(observed, [expected])

    kpsoi = ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)], shape=(3, 3))
    aug = iaa.PadToFixedSize(height=4, width=4, pad_mode="edge", position="right-bottom")
    observed = aug.augment_keypoints([kpsoi])
    expected = ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)], shape=(4, 4))
    assert observed[0].shape == expected.shape
    assert keypoints_equal(observed, [expected])

    # basic heatmaps test
    # pad_mode should be ignored for heatmaps
    heatmaps = ia.HeatmapsOnImage(np.zeros((1, 1, 1), dtype=np.float32) + 1.0, shape=(1, 1, 3))
    aug = iaa.PadToFixedSize(height=3, width=3, pad_mode="edge", position="center")
    observed = aug.augment_heatmaps([heatmaps])[0]
    expected = np.float32([
        [0, 0, 0],
        [0, 1.0, 0],
        [0, 0, 0]
    ])
    expected = expected[..., np.newaxis]
    assert observed.shape == (3, 3, 3)
    assert np.allclose(observed.arr_0to1, expected)

    # heatmaps with size unequal to image
    # pad_mode should be ignored for heatmaps
    heatmaps = ia.HeatmapsOnImage(np.zeros((15, 15, 1), dtype=np.float32) + 1.0, shape=(30, 30, 3))
    aug = iaa.PadToFixedSize(height=32, width=32, pad_mode="edge", position="left-top")
    observed = aug.augment_heatmaps([heatmaps])[0]
    expected = np.zeros((16, 16, 1), dtype=np.float32) + 1.0
    expected[:, 0, 0] = 0.0
    expected[0, :, 0] = 0.0
    assert observed.shape == (32, 32, 3)
    assert np.allclose(observed.arr_0to1, expected)

    ###################
    # test other dtypes
    ###################
    aug = iaa.PadToFixedSize(height=4, width=3, position="center-top")
    mask = np.zeros((4, 3), dtype=bool)
    mask[2, 1] = True

    # bool
    image = np.zeros((3, 3), dtype=bool)
    image[1, 1] = True
    image_aug = aug.augment_image(image)
    assert image_aug.dtype == image.dtype
    assert image_aug.shape == (4, 3)
    assert np.all(image_aug[~mask] == 0)
    assert np.all(image_aug[mask] == 1)

    # uint, int
    for dtype in [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64]:
        min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)

        if np.dtype(dtype).kind == "i":
            values = [1, 5, 10, 100, int(0.1 * max_value), int(0.2 * max_value),
                      int(0.5 * max_value), max_value - 100, max_value]
            values = values + [(-1) * value for value in values]
        else:
            values = [1, 5, 10, 100, int(center_value), int(0.1 * max_value), int(0.2 * max_value),
                      int(0.5 * max_value), max_value - 100, max_value]

        for value in values:
            image = np.zeros((3, 3), dtype=dtype)
            image[1, 1] = value
            image_aug = aug.augment_image(image)
            assert image_aug.dtype == np.dtype(dtype)
            assert image_aug.shape == (4, 3)
            assert np.all(image_aug[~mask] == 0)
            assert np.all(image_aug[mask] == value)

    # float
    for dtype in [np.float16, np.float32, np.float64, np.float128]:
        min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)

        def _isclose(a, b):
            atol = 1e-4 if dtype == np.float16 else 1e-8
            return np.isclose(a, b, atol=atol, rtol=0)

        isize = np.dtype(dtype).itemsize
        values = [0.01, 1.0, 10.0, 100.0, 500 ** (isize - 1), 1000 ** (isize - 1)]
        values = values + [(-1) * value for value in values]
        values = values + [min_value, max_value]
        for value in values:
            image = np.zeros((3, 3), dtype=dtype)
            image[1, 1] = value
            image_aug = aug.augment_image(image)
            assert image_aug.dtype == np.dtype(dtype)
            assert image_aug.shape == (4, 3)
            assert np.all(_isclose(image_aug[~mask], 0))
            assert np.all(_isclose(image_aug[mask], np.float128(value)))


def test_CropToFixedSize():
    reseed()

    img = np.uint8([
        [128, 129, 130],
        [131, 132, 133],
        [134, 135, 136]
    ])
    img3d = img[:, :, np.newaxis]
    img3d_rgb = np.tile(img3d, (1, 1, 3))

    # basic functionality
    aug = iaa.CropToFixedSize(height=1, width=1)
    observed = aug.augment_image(img)
    assert observed.dtype.type == np.uint8
    assert observed.shape == (1, 1)

    observed = aug.augment_image(img3d)
    assert observed.dtype.type == np.uint8
    assert observed.shape == (1, 1, 1)

    observed = aug.augment_image(img3d_rgb)
    assert observed.dtype.type == np.uint8
    assert observed.shape == (1, 1, 3)

    # test float32, float64, int32
    for dtype in [np.float32, np.float64, np.int32]:
        aug = iaa.CropToFixedSize(height=1, width=1)
        observed = aug.augment_image(img.astype(dtype))
        assert observed.dtype.type == dtype
        assert observed.shape == (1, 1)

    # change only one side when other side has already desired size
    aug = iaa.CropToFixedSize(height=3, width=5)
    observed = aug.augment_image(np.zeros((3, 5, 3), dtype=np.uint8))
    assert observed.dtype.type == np.uint8
    assert observed.shape == (3, 5, 3)

    aug = iaa.CropToFixedSize(height=5, width=3)
    observed = aug.augment_image(np.zeros((5, 3, 3), dtype=np.uint8))
    assert observed.dtype.type == np.uint8
    assert observed.shape == (5, 3, 3)

    # change no side when all sides have exactly desired size
    img5x5 = np.zeros((5, 5, 3), dtype=np.uint8)
    img5x5[2, 2, :] = 255
    aug = iaa.CropToFixedSize(height=5, width=5)
    observed = aug.augment_image(img5x5)
    assert observed.dtype.type == np.uint8
    assert observed.shape == (5, 5, 3)
    assert np.array_equal(observed, img5x5)

    # change no side when all sides have smaller than desired size
    img4x4 = np.zeros((4, 4, 3), dtype=np.uint8)
    img4x4[2, 2, :] = 255
    aug = iaa.CropToFixedSize(height=5, width=5)
    observed = aug.augment_image(img4x4)
    assert observed.dtype.type == np.uint8
    assert observed.shape == (4, 4, 3)
    assert np.array_equal(observed, img4x4)

    # explicit non-center position test
    aug = iaa.CropToFixedSize(height=3, width=3, position="left-top")
    img5x5 = np.arange(25, dtype=np.uint8).reshape((5, 5))
    observed = aug.augment_image(img5x5)
    expected = img5x5[2:, 2:]
    assert observed.dtype.type == np.uint8
    assert observed.shape == (3, 3)
    assert np.array_equal(observed, expected)

    aug = iaa.CropToFixedSize(height=3, width=3, position="right-bottom")
    img5x5 = np.arange(25, dtype=np.uint8).reshape((5, 5))
    observed = aug.augment_image(img5x5)
    expected = img5x5[:3, :3]
    assert observed.dtype.type == np.uint8
    assert observed.shape == (3, 3)
    assert np.array_equal(observed, expected)

    aug = iaa.CropToFixedSize(height=3, width=3, position=(0.5, 1.0))
    img5x5 = np.arange(25, dtype=np.uint8).reshape((5, 5))
    observed = aug.augment_image(img5x5)
    expected = img5x5[:3, 1:4]
    assert observed.dtype.type == np.uint8
    assert observed.shape == (3, 3)
    assert np.array_equal(observed, expected)

    # basic keypoint test
    kpsoi = ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)], shape=(3, 3))
    aug = iaa.CropToFixedSize(height=1, width=1, position="center")
    observed = aug.augment_keypoints([kpsoi])
    expected = ia.KeypointsOnImage([ia.Keypoint(x=0, y=0)], shape=(1, 1))
    assert observed[0].shape == expected.shape
    assert keypoints_equal(observed, [expected])

    # keypoint test with shape not being changed
    kpsoi = ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)], shape=(3, 3))
    aug = iaa.CropToFixedSize(height=3, width=3, position="center")
    observed = aug.augment_keypoints([kpsoi])
    expected = ia.KeypointsOnImage([ia.Keypoint(x=1, y=1)], shape=(3, 3))
    assert observed[0].shape == expected.shape
    assert keypoints_equal(observed, [expected])

    # keypoint test with explicit non-center position
    kpsoi = ia.KeypointsOnImage([ia.Keypoint(x=2, y=2)], shape=(5, 5))
    aug = iaa.CropToFixedSize(height=3, width=3, position="left-top")
    observed = aug.augment_keypoints([kpsoi])
    expected = ia.KeypointsOnImage([ia.Keypoint(x=0, y=0)], shape=(3, 3))
    assert observed[0].shape == expected.shape
    assert keypoints_equal(observed, [expected])

    kpsoi = ia.KeypointsOnImage([ia.Keypoint(x=2, y=2)], shape=(5, 5))
    aug = iaa.CropToFixedSize(height=3, width=3, position="right-bottom")
    observed = aug.augment_keypoints([kpsoi])
    expected = ia.KeypointsOnImage([ia.Keypoint(x=2, y=2)], shape=(3, 3))
    assert observed[0].shape == expected.shape
    assert keypoints_equal(observed, [expected])

    # basic heatmaps test
    heatmaps = ia.HeatmapsOnImage(np.zeros((5, 5, 1), dtype=np.float32) + 1.0, shape=(5, 5, 3))
    aug = iaa.CropToFixedSize(height=3, width=3, position="center")
    observed = aug.augment_heatmaps([heatmaps])[0]
    expected = np.zeros((3, 3, 1), dtype=np.float32) + 1.0
    assert observed.shape == (3, 3, 3)
    assert np.allclose(observed.arr_0to1, expected)

    # heatmaps, crop at non-center position
    heatmaps = np.linspace(0.0, 1.0, 5 * 5 * 1).reshape((5, 5, 1)).astype(np.float32)
    heatmaps_oi = ia.HeatmapsOnImage(heatmaps, shape=(5, 5, 3))
    aug = iaa.CropToFixedSize(height=3, width=3, position="left-top")
    observed = aug.augment_heatmaps([heatmaps_oi])[0]
    expected = heatmaps[2:, 2:, :]
    assert observed.shape == (3, 3, 3)
    assert np.allclose(observed.arr_0to1, expected)

    # heatmaps, crop at non-center position
    heatmaps = np.linspace(0.0, 1.0, 5 * 5 * 1).reshape((5, 5, 1)).astype(np.float32)
    heatmaps_oi = ia.HeatmapsOnImage(heatmaps, shape=(5, 5, 3))
    aug = iaa.CropToFixedSize(height=3, width=3, position="right-bottom")
    observed = aug.augment_heatmaps([heatmaps_oi])[0]
    expected = heatmaps[:3, :3, :]
    assert observed.shape == (3, 3, 3)
    assert np.allclose(observed.arr_0to1, expected)

    # heatmaps with size unequal to image
    heatmaps = ia.HeatmapsOnImage(np.zeros((17, 17, 1), dtype=np.float32) + 1.0, shape=(34, 34, 3))
    aug = iaa.CropToFixedSize(height=32, width=32, position="left-top")
    observed = aug.augment_heatmaps([heatmaps])[0]
    expected = np.zeros((16, 16, 1), dtype=np.float32) + 1.0
    assert observed.shape == (32, 32, 3)
    assert np.allclose(observed.arr_0to1, expected)

    ###################
    # test other dtypes
    ###################
    aug = iaa.CropToFixedSize(height=2, width=3, position="center-top")
    mask = np.zeros((2, 3), dtype=bool)
    mask[0, 1] = True

    # bool
    image = np.zeros((3, 3), dtype=bool)
    image[1, 1] = True
    image_aug = aug.augment_image(image)
    assert image_aug.dtype == image.dtype
    assert image_aug.shape == (2, 3)
    assert np.all(image_aug[~mask] == 0)
    assert np.all(image_aug[mask] == 1)

    # uint, int
    for dtype in [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64]:
        min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)

        if np.dtype(dtype).kind == "i":
            values = [1, 5, 10, 100, int(0.1 * max_value), int(0.2 * max_value),
                      int(0.5 * max_value), max_value - 100, max_value]
            values = values + [(-1) * value for value in values]
        else:
            values = [1, 5, 10, 100, int(center_value), int(0.1 * max_value), int(0.2 * max_value),
                      int(0.5 * max_value), max_value - 100, max_value]

        for value in values:
            image = np.zeros((3, 3), dtype=dtype)
            image[1, 1] = value
            image_aug = aug.augment_image(image)
            assert image_aug.dtype == np.dtype(dtype)
            assert image_aug.shape == (2, 3)
            assert np.all(image_aug[~mask] == 0)
            assert np.all(image_aug[mask] == value)

    # float
    for dtype in [np.float16, np.float32, np.float64, np.float128]:
        min_value, center_value, max_value = iadt.get_value_range_of_dtype(dtype)

        def _isclose(a, b):
            atol = 1e-4 if dtype == np.float16 else 1e-8
            return np.isclose(a, b, atol=atol, rtol=0)

        isize = np.dtype(dtype).itemsize
        values = [0.01, 1.0, 10.0, 100.0, 500 ** (isize - 1), 1000 ** (isize - 1)]
        values = values + [(-1) * value for value in values]
        values = values + [min_value, max_value]
        for value in values:
            image = np.zeros((3, 3), dtype=dtype)
            image[1, 1] = value
            image_aug = aug.augment_image(image)
            assert image_aug.dtype == np.dtype(dtype)
            assert image_aug.shape == (2, 3)
            assert np.all(_isclose(image_aug[~mask], 0))
            assert np.all(_isclose(image_aug[mask], np.float128(value)))


def test_KeepSizeByResize():
    reseed()

    children = iaa.Crop((1, 0, 0, 0), keep_size=False)

    aug = iaa.KeepSizeByResize(children, interpolation="cubic", interpolation_heatmaps="linear")
    samples, samples_heatmaps = aug._draw_samples(1000, ia.new_random_state(1), True)
    assert "cubic" in samples
    assert len(set(samples)) == 1
    assert "linear" in samples_heatmaps
    assert len(set(samples_heatmaps)) == 1

    aug = iaa.KeepSizeByResize(children, interpolation=iaa.KeepSizeByResize.NO_RESIZE,
                               interpolation_heatmaps=iaa.KeepSizeByResize.SAME_AS_IMAGES)
    samples, samples_heatmaps = aug._draw_samples(1000, ia.new_random_state(1), True)
    assert iaa.KeepSizeByResize.NO_RESIZE in samples
    assert len(set(samples)) == 1
    assert iaa.KeepSizeByResize.NO_RESIZE in samples_heatmaps
    assert len(set(samples_heatmaps)) == 1

    aug = iaa.KeepSizeByResize(children, interpolation=cv2.INTER_LINEAR,
                               interpolation_heatmaps=cv2.INTER_NEAREST)
    samples, samples_heatmaps = aug._draw_samples(1000, ia.new_random_state(1), True)
    assert cv2.INTER_LINEAR in samples
    assert len(set(samples)) == 1
    assert cv2.INTER_NEAREST in samples_heatmaps
    assert len(set(samples_heatmaps)) == 1

    aug = iaa.KeepSizeByResize(children, interpolation=["cubic", "nearest"],
                               interpolation_heatmaps=["linear", iaa.KeepSizeByResize.SAME_AS_IMAGES])
    samples, samples_heatmaps = aug._draw_samples(5000, ia.new_random_state(1), True)
    assert "cubic" in samples
    assert "nearest" in samples
    assert len(set(samples)) == 2
    assert "linear" in samples_heatmaps
    assert "nearest" in samples_heatmaps
    assert len(set(samples_heatmaps)) == 3
    assert 0.5 - 0.1 < np.sum(samples == samples_heatmaps) / samples_heatmaps.size < 0.5 + 0.1

    aug = iaa.KeepSizeByResize(children, interpolation=iap.Choice(["cubic", "linear"]),
                               interpolation_heatmaps=iap.Choice(["linear", "nearest"]))
    samples, samples_heatmaps = aug._draw_samples(10000, ia.new_random_state(1), True)
    assert "cubic" in samples
    assert "linear" in samples
    assert len(set(samples)) == 2
    assert "linear" in samples_heatmaps
    assert "nearest" in samples_heatmaps
    assert len(set(samples_heatmaps)) == 2

    img = np.arange(0, 4*4*3, 1).reshape((4, 4, 3)).astype(np.uint8)
    aug = iaa.KeepSizeByResize(children, interpolation="cubic")
    observed = aug.augment_image(img)
    assert observed.shape == (4, 4, 3)
    assert observed.dtype.type == np.uint8
    expected = ia.imresize_single_image(img[1:, :, :], img.shape[0:2], interpolation="cubic")
    assert np.allclose(observed, expected)

    aug = iaa.KeepSizeByResize(children, interpolation=iaa.KeepSizeByResize.NO_RESIZE)
    observed = aug.augment_image(img)
    expected = img[1:, :, :]
    assert observed.shape == (3, 4, 3)
    assert observed.dtype.type == np.uint8
    assert np.allclose(observed, expected)

    # keypoints
    keypoints = [ia.Keypoint(x=0, y=1), ia.Keypoint(x=1, y=1), ia.Keypoint(x=2, y=3)]
    kpoi = ia.KeypointsOnImage(keypoints, shape=(4, 4, 3))
    aug = iaa.KeepSizeByResize(children, interpolation="cubic")
    kpoi_aug = aug.augment_keypoints([kpoi])[0]
    assert kpoi_aug.shape == (4, 4, 3)
    assert 0 - 1e-4 < kpoi_aug.keypoints[0].x < 0 + 1e-4
    assert ((1-1)/3)*4 - 1e-4 < kpoi_aug.keypoints[0].y < ((1-1)/3)*4 + 1e-4
    assert 1 - 1e-4 < kpoi_aug.keypoints[1].x < 1 + 1e-4
    assert ((1-1)/3)*4 - 1e-4 < kpoi_aug.keypoints[1].y < ((1-1)/3)*4 + 1e-4
    assert 2 - 1e-4 < kpoi_aug.keypoints[2].x < 2 + 1e-4
    assert ((3-1)/3)*4 - 1e-4 < kpoi_aug.keypoints[2].y < ((3-1)/3)*4 + 1e-4

    kpoi = ia.KeypointsOnImage(keypoints, shape=(4, 4, 3))
    aug = iaa.KeepSizeByResize(children, interpolation=iaa.KeepSizeByResize.NO_RESIZE)
    kpoi_aug = aug.augment_keypoints([kpoi])[0]
    assert kpoi_aug.shape == (3, 4, 3)
    assert 0 - 1e-4 < kpoi_aug.keypoints[0].x < 0 + 1e-4
    assert 0 - 1e-4 < kpoi_aug.keypoints[0].y < 0 + 1e-4
    assert 1 - 1e-4 < kpoi_aug.keypoints[1].x < 1 + 1e-4
    assert 0 - 1e-4 < kpoi_aug.keypoints[1].y < 0 + 1e-4
    assert 2 - 1e-4 < kpoi_aug.keypoints[2].x < 2 + 1e-4
    assert 2 - 1e-4 < kpoi_aug.keypoints[2].y < 2 + 1e-4

    # heatmaps
    heatmaps = np.linspace(0.0, 1.0, 4*4*1).reshape((4, 4, 1)).astype(np.float32)
    heatmaps_oi = ia.HeatmapsOnImage(heatmaps, shape=(4, 4, 1))
    heatmaps_oi_cubic = ia.HeatmapsOnImage(heatmaps[1:, :, :], shape=(3, 4, 3)).resize((4, 4), interpolation="cubic")
    heatmaps_oi_cubic.shape = (4, 4, 3)
    heatmaps_oi_nearest = ia.HeatmapsOnImage(heatmaps[1:, :, :], shape=(3, 4, 1)).resize((4, 4), interpolation="nearest")
    heatmaps_oi_nearest.shape = (4, 4, 3)

    aug = iaa.KeepSizeByResize(children, interpolation="cubic", interpolation_heatmaps="nearest")
    heatmaps_oi_aug = aug.augment_heatmaps([heatmaps_oi])[0]
    assert heatmaps_oi_aug.arr_0to1.shape == (4, 4, 1)
    assert np.allclose(heatmaps_oi_aug.arr_0to1, heatmaps_oi_nearest.arr_0to1)

    aug = iaa.KeepSizeByResize(children, interpolation="cubic", interpolation_heatmaps=["nearest", "cubic"])
    heatmaps_oi_aug = aug.augment_heatmaps([heatmaps_oi])[0]
    assert heatmaps_oi_aug.arr_0to1.shape == (4, 4, 1)
    assert np.allclose(heatmaps_oi_aug.arr_0to1, heatmaps_oi_nearest.arr_0to1) \
        or np.allclose(heatmaps_oi_aug.arr_0to1, heatmaps_oi_cubic.arr_0to1)

    aug = iaa.KeepSizeByResize(children, interpolation="cubic", interpolation_heatmaps=iaa.KeepSizeByResize.NO_RESIZE)
    heatmaps_oi_aug = aug.augment_heatmaps([heatmaps_oi])[0]
    assert heatmaps_oi_aug.arr_0to1.shape == (3, 4, 1)
    assert np.allclose(heatmaps_oi_aug.arr_0to1, heatmaps_oi.arr_0to1[1:, :, :])

    aug = iaa.KeepSizeByResize(children, interpolation="cubic",
                               interpolation_heatmaps=iaa.KeepSizeByResize.SAME_AS_IMAGES)
    heatmaps_oi_aug = aug.augment_heatmaps([heatmaps_oi])[0]
    assert heatmaps_oi_aug.arr_0to1.shape == (4, 4, 1)
    assert np.allclose(heatmaps_oi_aug.arr_0to1, heatmaps_oi_cubic.arr_0to1)


if __name__ == "__main__":
    main()
