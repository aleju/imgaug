from __future__ import print_function, division, absolute_import

import time
import warnings
import sys
# unittest only added in 3.4 self.subTest()
if sys.version_info[0] < 3 or sys.version_info[1] < 4:
    import unittest2 as unittest
else:
    import unittest
# unittest.mock is not available in 2.7 (though unittest2 might contain it?)
try:
    import unittest.mock as mock
except ImportError:
    import mock

import matplotlib
matplotlib.use('Agg')  # fix execution of tests involving matplotlib on travis
import numpy as np
import imgaug as ia


def main():
    time_start = time.time()

    test_Keypoint()
    test_KeypointsOnImage()

    time_end = time.time()
    print("<%s> Finished without errors in %.4fs." % (__file__, time_end - time_start,))


def test_Keypoint():
    eps = 1e-8

    # -------------
    # x/y/x_int/y_int
    # -------------
    kp = ia.Keypoint(y=1, x=2)
    assert kp.y == 1
    assert kp.x == 2
    assert kp.y_int == 1
    assert kp.x_int == 2
    kp = ia.Keypoint(y=1.1, x=2.7)
    assert 1.1 - eps < kp.y < 1.1 + eps
    assert 2.7 - eps < kp.x < 2.7 + eps
    assert kp.y_int == 1
    assert kp.x_int == 3

    # -------------
    # project
    # -------------
    kp = ia.Keypoint(y=1, x=2)
    kp2 = kp.project((10, 10), (10, 10))
    assert kp2.y == 1
    assert kp2.x == 2
    kp2 = kp.project((10, 10), (20, 10))
    assert kp2.y == 2
    assert kp2.x == 2
    kp2 = kp.project((10, 10), (10, 20))
    assert kp2.y == 1
    assert kp2.x == 4
    kp2 = kp.project((10, 10), (20, 20))
    assert kp2.y == 2
    assert kp2.x == 4

    # -------------
    # shift
    # -------------
    kp = ia.Keypoint(y=1, x=2)
    kp2 = kp.shift(y=1)
    assert kp2.y == 2
    assert kp2.x == 2
    kp2 = kp.shift(y=-1)
    assert kp2.y == 0
    assert kp2.x == 2
    kp2 = kp.shift(x=1)
    assert kp2.y == 1
    assert kp2.x == 3
    kp2 = kp.shift(x=-1)
    assert kp2.y == 1
    assert kp2.x == 1
    kp2 = kp.shift(y=1, x=2)
    assert kp2.y == 2
    assert kp2.x == 4

    # -------------
    # draw_on_image
    # -------------
    kp = ia.Keypoint(x=0, y=0)
    image = np.zeros((5, 5, 3), dtype=np.uint8) + 10
    image_kp = kp.draw_on_image(
        image, color=(0, 255, 0), alpha=1, size=1, copy=True,
        raise_if_out_of_image=False)
    assert np.all(image_kp[0, 0, :] == [0, 255, 0])
    assert np.all(image_kp[1:, :, :] == 10)
    assert np.all(image_kp[:, 1:, :] == 10)

    kp = ia.Keypoint(x=4, y=4)
    image = np.zeros((5, 5, 3), dtype=np.uint8) + 10
    image_kp = kp.draw_on_image(
        image, color=(0, 255, 0), alpha=1, size=1, copy=True,
        raise_if_out_of_image=False)
    assert np.all(image_kp[4, 4, :] == [0, 255, 0])
    assert np.all(image_kp[:4, :, :] == 10)
    assert np.all(image_kp[:, :4, :] == 10)

    kp = ia.Keypoint(x=4, y=4)
    image = np.zeros((5, 5, 3), dtype=np.uint8) + 10
    image_kp = kp.draw_on_image(
        image, color=(0, 255, 0), alpha=1, size=5, copy=True,
        raise_if_out_of_image=False)
    assert np.all(image_kp[2:, 2:, :] == [0, 255, 0])
    assert np.all(image_kp[:2, :, :] == 10)
    assert np.all(image_kp[:, :2, :] == 10)

    kp = ia.Keypoint(x=5, y=5)
    image = np.zeros((5, 5, 3), dtype=np.uint8) + 10
    image_kp = kp.draw_on_image(
        image, color=(0, 255, 0), alpha=1, size=5, copy=True,
        raise_if_out_of_image=False)
    assert np.all(image_kp[3:, 3:, :] == [0, 255, 0])
    assert np.all(image_kp[:3, :, :] == 10)
    assert np.all(image_kp[:, :3, :] == 10)

    kp = ia.Keypoint(x=0, y=0)
    image = np.zeros((5, 5, 3), dtype=np.uint8) + 10
    image_kp = kp.draw_on_image(
        image, color=(0, 255, 0), alpha=1, size=5, copy=True,
        raise_if_out_of_image=False)
    assert np.all(image_kp[:3, :3, :] == [0, 255, 0])
    assert np.all(image_kp[3:, :, :] == 10)
    assert np.all(image_kp[:, 3:, :] == 10)

    kp = ia.Keypoint(x=-1, y=-1)
    image = np.zeros((5, 5, 3), dtype=np.uint8) + 10
    image_kp = kp.draw_on_image(
        image, color=(0, 255, 0), alpha=1, size=5, copy=True,
        raise_if_out_of_image=False)
    assert np.all(image_kp[:2, :2, :] == [0, 255, 0])
    assert np.all(image_kp[2:, :, :] == 10)
    assert np.all(image_kp[:, 2:, :] == 10)

    kp = ia.Keypoint(x=0, y=0)
    image = np.zeros((5, 5, 3), dtype=np.uint8) + 10
    image_kp = kp.draw_on_image(
        image, color=(0, 200, 0), alpha=0.5, size=1, copy=True,
        raise_if_out_of_image=False)
    assert np.all(image_kp[0, 0, :] == [0 + 5, 100 + 5, 0 + 5])
    assert np.all(image_kp[1:, :, :] == 10)
    assert np.all(image_kp[:, 1:, :] == 10)

    # -------------
    # generate_similar_points_manhattan
    # -------------
    kp = ia.Keypoint(y=4, x=5)
    kps_manhatten = kp.generate_similar_points_manhattan(0, 1.0, return_array=False)
    assert len(kps_manhatten) == 1
    assert kps_manhatten[0].y == 4
    assert kps_manhatten[0].x == 5

    kps_manhatten = kp.generate_similar_points_manhattan(1, 1.0, return_array=False)
    assert len(kps_manhatten) == 5
    expected = [(4, 5), (3, 5), (4, 6), (5, 5), (4, 4)]
    for y, x in expected:
        assert any([np.allclose([y, x], [kp_manhatten.y, kp_manhatten.x]) for kp_manhatten in kps_manhatten])

    kps_manhatten = kp.generate_similar_points_manhattan(1, 1.0, return_array=True)
    assert kps_manhatten.shape == (5, 2)
    expected = [(4, 5), (3, 5), (4, 6), (5, 5), (4, 4)]
    for y, x in expected:
        assert any([np.allclose([y, x], [kp_manhatten_y, kp_manhatten_x])
                    for kp_manhatten_x, kp_manhatten_y in kps_manhatten])

    # -------------
    # __repr__ / __str_
    # -------------
    kp = ia.Keypoint(y=1, x=2)
    assert kp.__repr__() == kp.__str__() == "Keypoint(x=2.00000000, y=1.00000000)"
    kp = ia.Keypoint(y=1.2, x=2.7)
    assert kp.__repr__() == kp.__str__() == "Keypoint(x=2.70000000, y=1.20000000)"


def test_KeypointsOnImage():
    eps = 1e-8

    kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)]

    # height/width
    kpi = ia.KeypointsOnImage(keypoints=kps, shape=(10, 20, 3))
    assert kpi.height == 10
    assert kpi.width == 20

    # image instead of shape
    kpi = ia.KeypointsOnImage(keypoints=kps, shape=np.zeros((10, 20, 3), dtype=np.uint8))
    assert kpi.shape == (10, 20, 3)

    # -------------
    # on()
    # -------------
    kpi2 = kpi.on((10, 20, 3))
    assert all([kp_i.x == kp_j.x and kp_i.y == kp_j.y for kp_i, kp_j in zip(kpi.keypoints, kpi2.keypoints)])

    kpi2 = kpi.on((20, 40, 3))
    assert kpi2.keypoints[0].x == 2
    assert kpi2.keypoints[0].y == 4
    assert kpi2.keypoints[1].x == 6
    assert kpi2.keypoints[1].y == 8

    kpi2 = kpi.on(np.zeros((20, 40, 3), dtype=np.uint8))
    assert kpi2.keypoints[0].x == 2
    assert kpi2.keypoints[0].y == 4
    assert kpi2.keypoints[1].x == 6
    assert kpi2.keypoints[1].y == 8

    # -------------
    # draw_on_image
    # -------------
    kpi = ia.KeypointsOnImage(keypoints=kps, shape=(5, 5, 3))
    image = np.zeros((5, 5, 3), dtype=np.uint8) + 10
    kps_mask = np.zeros(image.shape[0:2], dtype=np.bool)
    kps_mask[2, 1] = 1
    kps_mask[4, 3] = 1
    image_kps = kpi.draw_on_image(image, color=[0, 255, 0], size=1, copy=True, raise_if_out_of_image=False)
    assert np.all(image_kps[kps_mask] == [0, 255, 0])
    assert np.all(image_kps[~kps_mask] == [10, 10, 10])

    image_kps = kpi.draw_on_image(image, color=[0, 255, 0], alpha=0.5, size=1, copy=True, raise_if_out_of_image=False)
    assert np.all(image_kps[kps_mask] == [int(0.5*10+0), int(0.5*10+0.5*255), int(10*0.5+0)])
    assert np.all(image_kps[~kps_mask] == [10, 10, 10])

    image_kps = kpi.draw_on_image(image, color=[0, 255, 0], size=3, copy=True, raise_if_out_of_image=False)
    kps_mask_size3 = np.copy(kps_mask)
    kps_mask_size3[2-1:2+1+1, 1-1:1+1+1] = 1
    kps_mask_size3[4-1:4+1+1, 3-1:3+1+1] = 1
    assert np.all(image_kps[kps_mask_size3] == [0, 255, 0])
    assert np.all(image_kps[~kps_mask_size3] == [10, 10, 10])

    image_kps = kpi.draw_on_image(image, color=[0, 0, 255], size=1, copy=True, raise_if_out_of_image=False)
    assert np.all(image_kps[kps_mask] == [0, 0, 255])
    assert np.all(image_kps[~kps_mask] == [10, 10, 10])

    image_kps = kpi.draw_on_image(image, color=255, size=1, copy=True, raise_if_out_of_image=False)
    assert np.all(image_kps[kps_mask] == [255, 255, 255])
    assert np.all(image_kps[~kps_mask] == [10, 10, 10])

    image2 = np.copy(image)
    image_kps = kpi.draw_on_image(image2, color=[0, 255, 0], size=1, copy=False, raise_if_out_of_image=False)
    assert np.all(image2 == image_kps)
    assert np.all(image_kps[kps_mask] == [0, 255, 0])
    assert np.all(image_kps[~kps_mask] == [10, 10, 10])
    assert np.all(image2[kps_mask] == [0, 255, 0])
    assert np.all(image2[~kps_mask] == [10, 10, 10])

    kpi = ia.KeypointsOnImage(keypoints=kps + [ia.Keypoint(x=100, y=100)], shape=(5, 5, 3))
    image = np.zeros((5, 5, 3), dtype=np.uint8) + 10
    kps_mask = np.zeros(image.shape[0:2], dtype=np.bool)
    kps_mask[2, 1] = 1
    kps_mask[4, 3] = 1
    image_kps = kpi.draw_on_image(image, color=[0, 255, 0], size=1, copy=True, raise_if_out_of_image=False)
    assert np.all(image_kps[kps_mask] == [0, 255, 0])
    assert np.all(image_kps[~kps_mask] == [10, 10, 10])

    kpi = ia.KeypointsOnImage(keypoints=kps + [ia.Keypoint(x=100, y=100)], shape=(5, 5, 3))
    image = np.zeros((5, 5, 3), dtype=np.uint8) + 10
    got_exception = False
    try:
        image_kps = kpi.draw_on_image(image, color=[0, 255, 0], size=1, copy=True, raise_if_out_of_image=True)
        assert np.all(image_kps[kps_mask] == [0, 255, 0])
        assert np.all(image_kps[~kps_mask] == [10, 10, 10])
    except Exception:
        got_exception = True
    assert got_exception

    kpi = ia.KeypointsOnImage(keypoints=kps + [ia.Keypoint(x=5, y=5)], shape=(5, 5, 3))
    image = np.zeros((5, 5, 3), dtype=np.uint8) + 10
    kps_mask = np.zeros(image.shape[0:2], dtype=np.bool)
    kps_mask[2, 1] = 1
    kps_mask[4, 3] = 1
    image_kps = kpi.draw_on_image(image, color=[0, 255, 0], size=1, copy=True, raise_if_out_of_image=False)
    assert np.all(image_kps[kps_mask] == [0, 255, 0])
    assert np.all(image_kps[~kps_mask] == [10, 10, 10])

    got_exception = False
    try:
        image_kps = kpi.draw_on_image(image, color=[0, 255, 0], size=1, copy=True, raise_if_out_of_image=True)
        assert np.all(image_kps[kps_mask] == [0, 255, 0])
        assert np.all(image_kps[~kps_mask] == [10, 10, 10])
    except Exception:
        got_exception = True
    assert got_exception

    # -------------
    # shift
    # -------------
    kpi = ia.KeypointsOnImage(keypoints=kps, shape=(5, 5, 3))
    kpi2 = kpi.shift(x=0, y=0)
    assert kpi2.keypoints[0].x == kpi.keypoints[0].x
    assert kpi2.keypoints[0].y == kpi.keypoints[0].y
    assert kpi2.keypoints[1].x == kpi.keypoints[1].x
    assert kpi2.keypoints[1].y == kpi.keypoints[1].y

    kpi2 = kpi.shift(x=1)
    assert kpi2.keypoints[0].x == kpi.keypoints[0].x + 1
    assert kpi2.keypoints[0].y == kpi.keypoints[0].y
    assert kpi2.keypoints[1].x == kpi.keypoints[1].x + 1
    assert kpi2.keypoints[1].y == kpi.keypoints[1].y

    kpi2 = kpi.shift(x=-1)
    assert kpi2.keypoints[0].x == kpi.keypoints[0].x - 1
    assert kpi2.keypoints[0].y == kpi.keypoints[0].y
    assert kpi2.keypoints[1].x == kpi.keypoints[1].x - 1
    assert kpi2.keypoints[1].y == kpi.keypoints[1].y

    kpi2 = kpi.shift(y=1)
    assert kpi2.keypoints[0].x == kpi.keypoints[0].x
    assert kpi2.keypoints[0].y == kpi.keypoints[0].y + 1
    assert kpi2.keypoints[1].x == kpi.keypoints[1].x
    assert kpi2.keypoints[1].y == kpi.keypoints[1].y + 1

    kpi2 = kpi.shift(y=-1)
    assert kpi2.keypoints[0].x == kpi.keypoints[0].x
    assert kpi2.keypoints[0].y == kpi.keypoints[0].y - 1
    assert kpi2.keypoints[1].x == kpi.keypoints[1].x
    assert kpi2.keypoints[1].y == kpi.keypoints[1].y - 1

    kpi2 = kpi.shift(x=1, y=2)
    assert kpi2.keypoints[0].x == kpi.keypoints[0].x + 1
    assert kpi2.keypoints[0].y == kpi.keypoints[0].y + 2
    assert kpi2.keypoints[1].x == kpi.keypoints[1].x + 1
    assert kpi2.keypoints[1].y == kpi.keypoints[1].y + 2

    # -------------
    # to_xy_array
    # -------------
    kpi = ia.KeypointsOnImage(keypoints=kps, shape=(5, 5, 3))
    observed = kpi.to_xy_array()
    expected = np.float32([
        [1, 2],
        [3, 4]
    ])
    assert np.allclose(observed, expected)

    # -------------
    # from_xy_array
    # -------------
    arr = np.float32([
        [1, 2],
        [3, 4]
    ])
    kpi = ia.KeypointsOnImage.from_xy_array(arr, shape=(5, 5, 3))
    assert 1 - eps < kpi.keypoints[0].x < 1 + eps
    assert 2 - eps < kpi.keypoints[0].y < 2 + eps
    assert 3 - eps < kpi.keypoints[1].x < 3 + eps
    assert 4 - eps < kpi.keypoints[1].y < 4 + eps

    # -------------
    # to_keypoint_image
    # -------------
    kpi = ia.KeypointsOnImage(keypoints=kps, shape=(5, 5, 3))
    image = kpi.to_keypoint_image(size=1)
    image_size3 = kpi.to_keypoint_image(size=3)
    kps_mask = np.zeros((5, 5, 2), dtype=np.bool)
    kps_mask[2, 1, 0] = 1
    kps_mask[4, 3, 1] = 1
    kps_mask_size3 = np.zeros_like(kps_mask)
    kps_mask_size3[2-1:2+1+1, 1-1:1+1+1, 0] = 1
    kps_mask_size3[4-1:4+1+1, 3-1:3+1+1, 1] = 1
    assert np.all(image[kps_mask] == 255)
    assert np.all(image[~kps_mask] == 0)
    assert np.all(image_size3[kps_mask] == 255)
    assert np.all(image_size3[kps_mask_size3] >= 128)
    assert np.all(image_size3[~kps_mask_size3] == 0)

    # -------------
    # from_keypoint_image()
    # -------------
    kps_image = np.zeros((5, 5, 2), dtype=np.uint8)
    kps_image[2, 1, 0] = 255
    kps_image[4, 3, 1] = 255
    kpi2 = ia.KeypointsOnImage.from_keypoint_image(kps_image, nb_channels=3)
    assert kpi2.shape == (5, 5, 3)
    assert len(kpi2.keypoints) == 2
    assert kpi2.keypoints[0].y == 2
    assert kpi2.keypoints[0].x == 1
    assert kpi2.keypoints[1].y == 4
    assert kpi2.keypoints[1].x == 3

    kps_image = np.zeros((5, 5, 2), dtype=np.uint8)
    kps_image[2, 1, 0] = 255
    kps_image[4, 3, 1] = 10
    kpi2 = ia.KeypointsOnImage.from_keypoint_image(kps_image, if_not_found_coords={"x": -1, "y": -2}, threshold=20,
                                                   nb_channels=3)
    assert kpi2.shape == (5, 5, 3)
    assert len(kpi2.keypoints) == 2
    assert kpi2.keypoints[0].y == 2
    assert kpi2.keypoints[0].x == 1
    assert kpi2.keypoints[1].y == -2
    assert kpi2.keypoints[1].x == -1

    kps_image = np.zeros((5, 5, 2), dtype=np.uint8)
    kps_image[2, 1, 0] = 255
    kps_image[4, 3, 1] = 10
    kpi2 = ia.KeypointsOnImage.from_keypoint_image(kps_image, if_not_found_coords=(-1, -2), threshold=20,
                                                   nb_channels=3)
    assert kpi2.shape == (5, 5, 3)
    assert len(kpi2.keypoints) == 2
    assert kpi2.keypoints[0].y == 2
    assert kpi2.keypoints[0].x == 1
    assert kpi2.keypoints[1].y == -2
    assert kpi2.keypoints[1].x == -1

    kps_image = np.zeros((5, 5, 2), dtype=np.uint8)
    kps_image[2, 1, 0] = 255
    kps_image[4, 3, 1] = 10
    kpi2 = ia.KeypointsOnImage.from_keypoint_image(kps_image, if_not_found_coords=None, threshold=20, nb_channels=3)
    assert kpi2.shape == (5, 5, 3)
    assert len(kpi2.keypoints) == 1
    assert kpi2.keypoints[0].y == 2
    assert kpi2.keypoints[0].x == 1

    got_exception = False
    try:
        kps_image = np.zeros((5, 5, 2), dtype=np.uint8)
        kps_image[2, 1, 0] = 255
        kps_image[4, 3, 1] = 10
        _ = ia.KeypointsOnImage.from_keypoint_image(kps_image, if_not_found_coords="exception-please", threshold=20,
                                                    nb_channels=3)
    except Exception as exc:
        assert "Expected if_not_found_coords to be" in str(exc)
        got_exception = True
    assert got_exception

    # -------------
    # to_distance_maps()
    # -------------
    kpi = ia.KeypointsOnImage(keypoints=[ia.Keypoint(x=2, y=3)], shape=(5, 5, 3))
    distance_map = kpi.to_distance_maps()
    expected_xx = np.float32([
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4]
    ])
    expected_yy = np.float32([
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4]
    ])
    expected = np.sqrt((expected_xx - 2)**2 + (expected_yy - 3)**2)
    assert distance_map.shape == (5, 5, 1)
    assert np.allclose(distance_map, expected[..., np.newaxis])

    distance_map_inv = kpi.to_distance_maps(inverted=True)
    expected_inv = np.divide(np.ones_like(expected), expected+1)
    assert np.allclose(distance_map_inv, expected_inv[..., np.newaxis])

    # to_distance_maps() with two keypoints
    # positions on (4, 4) map (X=position, 1=KP 1 is closest, 2=KP 2 is closest, B=close to both)
    # [1, X, 1, 1]
    # [1, 1, 1, B]
    # [B, 2, 2, 2]
    # [2, 2, X, 2]
    # this test could have been done a bit better by simply splitting the distance maps, one per keypoint, considering
    # the function returns one distance map per keypoint
    kpi = ia.KeypointsOnImage(keypoints=[ia.Keypoint(x=2, y=3), ia.Keypoint(x=1, y=0)], shape=(4, 4, 3))
    expected = np.float32([
        [(0-1)**2 + (0-0)**2, (1-1)**2 + (0-0)**2, (2-1)**2 + (0-0)**2, (3-1)**2 + (0-0)**2],
        [(0-1)**2 + (1-0)**2, (1-1)**2 + (1-0)**2, (2-1)**2 + (1-0)**2, (3-1)**2 + (1-0)**2],
        [(0-1)**2 + (2-0)**2, (1-2)**2 + (2-3)**2, (2-2)**2 + (2-3)**2, (3-2)**2 + (2-3)**2],
        [(0-2)**2 + (3-3)**2, (1-2)**2 + (3-3)**2, (2-2)**2 + (3-3)**2, (3-2)**2 + (3-3)**2],
    ])
    distance_map = kpi.to_distance_maps()
    expected = np.sqrt(expected)
    assert np.allclose(np.min(distance_map, axis=2), expected)

    distance_map_inv = kpi.to_distance_maps(inverted=True)
    expected_inv = np.divide(np.ones_like(expected), expected+1)
    assert np.allclose(np.max(distance_map_inv, axis=2), expected_inv)

    # -------------
    # from_distance_maps()
    # -------------
    distance_map1 = np.float32([
        [2, 2, 2, 2, 2],
        [2, 1, 1, 1, 2],
        [2, 1, 0, 1, 2],
        [2, 1, 1, 1, 2]
    ])
    distance_map2 = np.float32([
        [4, 3, 2, 2, 2],
        [4, 3, 2, 1, 1],
        [4, 3, 2, 1, 0.1],
        [4, 3, 2, 1, 1]
    ])
    distance_maps = np.concatenate([distance_map1[..., np.newaxis], distance_map2[..., np.newaxis]], axis=2)
    kpi = ia.KeypointsOnImage.from_distance_maps(distance_maps, nb_channels=4)
    assert len(kpi.keypoints) == 2
    assert kpi.keypoints[0].x == 2
    assert kpi.keypoints[0].y == 2
    assert kpi.keypoints[1].x == 4
    assert kpi.keypoints[1].y == 2
    assert kpi.shape == (4, 5, 4)

    kpi = ia.KeypointsOnImage.from_distance_maps(np.divide(np.ones_like(distance_maps), distance_maps+1),
                                                 inverted=True)
    assert len(kpi.keypoints) == 2
    assert kpi.keypoints[0].x == 2
    assert kpi.keypoints[0].y == 2
    assert kpi.keypoints[1].x == 4
    assert kpi.keypoints[1].y == 2
    assert kpi.shape == (4, 5)

    kpi = ia.KeypointsOnImage.from_distance_maps(distance_maps, if_not_found_coords=(1, 1), threshold=0.09)
    assert len(kpi.keypoints) == 2
    assert kpi.keypoints[0].x == 2
    assert kpi.keypoints[0].y == 2
    assert kpi.keypoints[1].x == 1
    assert kpi.keypoints[1].y == 1
    assert kpi.shape == (4, 5)

    kpi = ia.KeypointsOnImage.from_distance_maps(distance_maps, if_not_found_coords={"x": 1, "y": 2}, threshold=0.09)
    assert len(kpi.keypoints) == 2
    assert kpi.keypoints[0].x == 2
    assert kpi.keypoints[0].y == 2
    assert kpi.keypoints[1].x == 1
    assert kpi.keypoints[1].y == 2
    assert kpi.shape == (4, 5)

    kpi = ia.KeypointsOnImage.from_distance_maps(distance_maps, if_not_found_coords=None, threshold=0.09)
    assert len(kpi.keypoints) == 1
    assert kpi.keypoints[0].x == 2
    assert kpi.keypoints[0].y == 2
    assert kpi.shape == (4, 5)

    got_exception = False
    try:
        _ = ia.KeypointsOnImage.from_distance_maps(distance_maps, if_not_found_coords=False, threshold=0.09)
    except Exception as exc:
        assert "Expected if_not_found_coords to be" in str(exc)
        got_exception = True
    assert got_exception

    # -------------
    # copy()
    # -------------
    kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)]
    kpi = ia.KeypointsOnImage(keypoints=kps, shape=(5, 5, 3))
    kpi2 = kpi.copy()
    assert kpi2.keypoints[0].x == 1
    assert kpi2.keypoints[0].y == 2
    assert kpi2.keypoints[1].x == 3
    assert kpi2.keypoints[1].y == 4
    kps[0].x = 100
    assert kpi2.keypoints[0].x == 100
    assert kpi2.keypoints[0].y == 2
    assert kpi2.keypoints[1].x == 3
    assert kpi2.keypoints[1].y == 4

    # -------------
    # deepcopy()
    # -------------
    kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)]
    kpi = ia.KeypointsOnImage(keypoints=kps, shape=(5, 5, 3))
    kpi2 = kpi.deepcopy()
    assert kpi2.keypoints[0].x == 1
    assert kpi2.keypoints[0].y == 2
    assert kpi2.keypoints[1].x == 3
    assert kpi2.keypoints[1].y == 4
    kps[0].x = 100
    assert kpi2.keypoints[0].x == 1
    assert kpi2.keypoints[0].y == 2
    assert kpi2.keypoints[1].x == 3
    assert kpi2.keypoints[1].y == 4

    # -------------
    # repr/str
    # -------------
    kps = [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)]
    kpi = ia.KeypointsOnImage(keypoints=kps, shape=(5, 5, 3))
    expected = "KeypointsOnImage([Keypoint(x=1.00000000, y=2.00000000), Keypoint(x=3.00000000, y=4.00000000)], " \
               + "shape=(5, 5, 3))"
    assert kpi.__repr__() == kpi.__str__() == expected
