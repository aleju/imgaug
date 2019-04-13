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
from imgaug.testutils import reseed


def main():
    time_start = time.time()

    test_BoundingBox()
    test_BoundingBoxesOnImage()

    time_end = time.time()
    print("<%s> Finished without errors in %.4fs." % (__file__, time_end - time_start,))


def test_BoundingBox():
    eps = 1e-8

    # properties with ints
    bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    assert bb.y1_int == 10
    assert bb.x1_int == 20
    assert bb.y2_int == 30
    assert bb.x2_int == 40
    assert bb.width == 40 - 20
    assert bb.height == 30 - 10
    center_x = bb.x1 + (bb.x2 - bb.x1)/2
    center_y = bb.y1 + (bb.y2 - bb.y1)/2
    assert center_x - eps < bb.center_x < center_x + eps
    assert center_y - eps < bb.center_y < center_y + eps

    # wrong order of y1/y2, x1/x2
    bb = ia.BoundingBox(y1=30, x1=40, y2=10, x2=20, label=None)
    assert bb.y1_int == 10
    assert bb.x1_int == 20
    assert bb.y2_int == 30
    assert bb.x2_int == 40

    # properties with floats
    bb = ia.BoundingBox(y1=10.1, x1=20.1, y2=30.9, x2=40.9, label=None)
    assert bb.y1_int == 10
    assert bb.x1_int == 20
    assert bb.y2_int == 31
    assert bb.x2_int == 41
    assert bb.width == 40.9 - 20.1
    assert bb.height == 30.9 - 10.1
    center_x = bb.x1 + (bb.x2 - bb.x1)/2
    center_y = bb.y1 + (bb.y2 - bb.y1)/2
    assert center_x - eps < bb.center_x < center_x + eps
    assert center_y - eps < bb.center_y < center_y + eps

    # area
    bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    assert bb.area == (30-10) * (40-20)

    # contains
    bb = ia.BoundingBox(y1=1, x1=2, y2=1+4, x2=2+5, label=None)
    assert bb.contains(ia.Keypoint(x=2.5, y=1.5)) is True
    assert bb.contains(ia.Keypoint(x=2, y=1)) is True
    assert bb.contains(ia.Keypoint(x=0, y=0)) is False

    # project
    bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb2 = bb.project((10, 10), (10, 10))
    assert 10 - eps < bb2.y1 < 10 + eps
    assert 20 - eps < bb2.x1 < 20 + eps
    assert 30 - eps < bb2.y2 < 30 + eps
    assert 40 - eps < bb2.x2 < 40 + eps

    bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb2 = bb.project((10, 10), (20, 20))
    assert 10*2 - eps < bb2.y1 < 10*2 + eps
    assert 20*2 - eps < bb2.x1 < 20*2 + eps
    assert 30*2 - eps < bb2.y2 < 30*2 + eps
    assert 40*2 - eps < bb2.x2 < 40*2 + eps

    bb2 = bb.project((10, 10), (5, 5))
    assert 10*0.5 - eps < bb2.y1 < 10*0.5 + eps
    assert 20*0.5 - eps < bb2.x1 < 20*0.5 + eps
    assert 30*0.5 - eps < bb2.y2 < 30*0.5 + eps
    assert 40*0.5 - eps < bb2.x2 < 40*0.5 + eps

    bb2 = bb.project((10, 10), (10, 20))
    assert 10*1 - eps < bb2.y1 < 10*1 + eps
    assert 20*2 - eps < bb2.x1 < 20*2 + eps
    assert 30*1 - eps < bb2.y2 < 30*1 + eps
    assert 40*2 - eps < bb2.x2 < 40*2 + eps

    bb2 = bb.project((10, 10), (20, 10))
    assert 10*2 - eps < bb2.y1 < 10*2 + eps
    assert 20*1 - eps < bb2.x1 < 20*1 + eps
    assert 30*2 - eps < bb2.y2 < 30*2 + eps
    assert 40*1 - eps < bb2.x2 < 40*1 + eps

    # extend
    bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb2 = bb.extend(all_sides=1)
    assert bb2.y1 == 10-1
    assert bb2.y2 == 30+1
    assert bb2.x1 == 20-1
    assert bb2.x2 == 40+1

    bb2 = bb.extend(all_sides=-1)
    assert bb2.y1 == 10-(-1)
    assert bb2.y2 == 30+(-1)
    assert bb2.x1 == 20-(-1)
    assert bb2.x2 == 40+(-1)

    bb2 = bb.extend(top=1)
    assert bb2.y1 == 10-1
    assert bb2.y2 == 30+0
    assert bb2.x1 == 20-0
    assert bb2.x2 == 40+0

    bb2 = bb.extend(right=1)
    assert bb2.y1 == 10-0
    assert bb2.y2 == 30+0
    assert bb2.x1 == 20-0
    assert bb2.x2 == 40+1

    bb2 = bb.extend(bottom=1)
    assert bb2.y1 == 10-0
    assert bb2.y2 == 30+1
    assert bb2.x1 == 20-0
    assert bb2.x2 == 40+0

    bb2 = bb.extend(left=1)
    assert bb2.y1 == 10-0
    assert bb2.y2 == 30+0
    assert bb2.x1 == 20-1
    assert bb2.x2 == 40+0

    # intersection
    bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb2 = ia.BoundingBox(y1=10, x1=39, y2=30, x2=59, label=None)
    bb_inter = bb1.intersection(bb2)
    assert bb_inter.x1 == 39
    assert bb_inter.x2 == 40
    assert bb_inter.y1 == 10
    assert bb_inter.y2 == 30

    bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb2 = ia.BoundingBox(y1=10, x1=41, y2=30, x2=61, label=None)
    bb_inter = bb1.intersection(bb2, default=False)
    assert bb_inter is False

    # union
    bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb2 = ia.BoundingBox(y1=10, x1=39, y2=30, x2=59, label=None)
    bb_union = bb1.union(bb2)
    assert bb_union.x1 == 20
    assert bb_union.x2 == 59
    assert bb_union.y1 == 10
    assert bb_union.y2 == 30

    # iou
    bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb2 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    iou = bb1.iou(bb2)
    assert 1.0 - eps < iou < 1.0 + eps

    bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb2 = ia.BoundingBox(y1=10, x1=41, y2=30, x2=61, label=None)
    iou = bb1.iou(bb2)
    assert 0.0 - eps < iou < 0.0 + eps

    bb1 = ia.BoundingBox(y1=10, x1=10, y2=20, x2=20, label=None)
    bb2 = ia.BoundingBox(y1=15, x1=15, y2=25, x2=25, label=None)
    iou = bb1.iou(bb2)
    area_union = 10 * 10 + 10 * 10 - 5 * 5
    area_intersection = 5 * 5
    iou_expected = area_intersection / area_union
    assert iou_expected - eps < iou < iou_expected + eps

    # is_fully_within_image
    bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    assert bb.is_fully_within_image((100, 100, 3)) is True
    assert bb.is_fully_within_image((20, 100, 3)) is False
    assert bb.is_fully_within_image((100, 30, 3)) is False
    assert bb.is_fully_within_image((1, 1, 3)) is False

    # is_partly_within_image
    bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    assert bb.is_partly_within_image((100, 100, 3)) is True
    assert bb.is_partly_within_image((20, 100, 3)) is True
    assert bb.is_partly_within_image((100, 30, 3)) is True
    assert bb.is_partly_within_image((1, 1, 3)) is False

    # is_out_of_image()
    bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    assert bb.is_out_of_image((100, 100, 3), partly=True, fully=True) is False
    assert bb.is_out_of_image((100, 100, 3), partly=False, fully=True) is False
    assert bb.is_out_of_image((100, 100, 3), partly=True, fully=False) is False
    assert bb.is_out_of_image((20, 100, 3), partly=True, fully=True) is True
    assert bb.is_out_of_image((20, 100, 3), partly=False, fully=True) is False
    assert bb.is_out_of_image((20, 100, 3), partly=True, fully=False) is True
    assert bb.is_out_of_image((100, 30, 3), partly=True, fully=True) is True
    assert bb.is_out_of_image((100, 30, 3), partly=False, fully=True) is False
    assert bb.is_out_of_image((100, 30, 3), partly=True, fully=False) is True
    assert bb.is_out_of_image((1, 1, 3), partly=True, fully=True) is True
    assert bb.is_out_of_image((1, 1, 3), partly=False, fully=True) is True
    assert bb.is_out_of_image((1, 1, 3), partly=True, fully=False) is False

    # clip_out_of_image
    bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb_cut = bb.clip_out_of_image((100, 100, 3))
    eps = np.finfo(np.float32).eps
    assert bb_cut.y1 == 10
    assert bb_cut.x1 == 20
    assert bb_cut.y2 == 30
    assert bb_cut.x2 == 40
    bb_cut = bb.clip_out_of_image(np.zeros((100, 100, 3), dtype=np.uint8))
    assert bb_cut.y1 == 10
    assert bb_cut.x1 == 20
    assert bb_cut.y2 == 30
    assert bb_cut.x2 == 40
    bb_cut = bb.clip_out_of_image((20, 100, 3))
    assert bb_cut.y1 == 10
    assert bb_cut.x1 == 20
    assert 20 - 2*eps < bb_cut.y2 < 20
    assert bb_cut.x2 == 40
    bb_cut = bb.clip_out_of_image((100, 30, 3))
    assert bb_cut.y1 == 10
    assert bb_cut.x1 == 20
    assert bb_cut.y2 == 30
    assert 30 - 2*eps < bb_cut.x2 < 30

    # shift
    bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb_top = bb.shift(top=0)
    bb_right = bb.shift(right=0)
    bb_bottom = bb.shift(bottom=0)
    bb_left = bb.shift(left=0)
    assert bb_top.y1 == 10
    assert bb_top.x1 == 20
    assert bb_top.y2 == 30
    assert bb_top.x2 == 40
    assert bb_right.y1 == 10
    assert bb_right.x1 == 20
    assert bb_right.y2 == 30
    assert bb_right.x2 == 40
    assert bb_bottom.y1 == 10
    assert bb_bottom.x1 == 20
    assert bb_bottom.y2 == 30
    assert bb_bottom.x2 == 40
    assert bb_left.y1 == 10
    assert bb_left.x1 == 20
    assert bb_left.y2 == 30
    assert bb_left.x2 == 40
    bb_top = bb.shift(top=1)
    bb_right = bb.shift(right=1)
    bb_bottom = bb.shift(bottom=1)
    bb_left = bb.shift(left=1)
    assert bb_top.y1 == 10+1
    assert bb_top.x1 == 20
    assert bb_top.y2 == 30+1
    assert bb_top.x2 == 40
    assert bb_right.y1 == 10
    assert bb_right.x1 == 20-1
    assert bb_right.y2 == 30
    assert bb_right.x2 == 40-1
    assert bb_bottom.y1 == 10-1
    assert bb_bottom.x1 == 20
    assert bb_bottom.y2 == 30-1
    assert bb_bottom.x2 == 40
    assert bb_left.y1 == 10
    assert bb_left.x1 == 20+1
    assert bb_left.y2 == 30
    assert bb_left.x2 == 40+1
    bb_top = bb.shift(top=-1)
    bb_right = bb.shift(right=-1)
    bb_bottom = bb.shift(bottom=-1)
    bb_left = bb.shift(left=-1)
    assert bb_top.y1 == 10-1
    assert bb_top.x1 == 20
    assert bb_top.y2 == 30-1
    assert bb_top.x2 == 40
    assert bb_right.y1 == 10
    assert bb_right.x1 == 20+1
    assert bb_right.y2 == 30
    assert bb_right.x2 == 40+1
    assert bb_bottom.y1 == 10+1
    assert bb_bottom.x1 == 20
    assert bb_bottom.y2 == 30+1
    assert bb_bottom.x2 == 40
    assert bb_left.y1 == 10
    assert bb_left.x1 == 20-1
    assert bb_left.y2 == 30
    assert bb_left.x2 == 40-1
    bb_mix = bb.shift(top=1, bottom=2, left=3, right=4)
    assert bb_mix.y1 == 10+1-2
    assert bb_mix.x1 == 20+3-4
    assert bb_mix.y2 == 30+3-4
    assert bb_mix.x2 == 40+1-2

    # draw_on_image()
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    bb = ia.BoundingBox(y1=1, x1=1, y2=3, x2=3, label=None)
    bb_mask = np.zeros(image.shape[0:2], dtype=np.bool)
    bb_mask[1:3+1, 1] = True
    bb_mask[1:3+1, 3] = True
    bb_mask[1, 1:3+1] = True
    bb_mask[3, 1:3+1] = True
    image_bb = bb.draw_on_image(image, color=[255, 255, 255], alpha=1.0, size=1, copy=True,
                                raise_if_out_of_image=False)
    assert np.all(image_bb[bb_mask] == [255, 255, 255])
    assert np.all(image_bb[~bb_mask] == [0, 0, 0])
    assert np.all(image == 0)

    image_bb = bb.draw_on_image(image, color=[255, 0, 0], alpha=1.0, size=1, copy=True,
                                raise_if_out_of_image=False)
    assert np.all(image_bb[bb_mask] == [255, 0, 0])
    assert np.all(image_bb[~bb_mask] == [0, 0, 0])

    image_bb = bb.draw_on_image(image, color=128, alpha=1.0, size=1, copy=True, raise_if_out_of_image=False)
    assert np.all(image_bb[bb_mask] == [128, 128, 128])
    assert np.all(image_bb[~bb_mask] == [0, 0, 0])

    image_bb = bb.draw_on_image(image + 100, color=[200, 200, 200], alpha=0.5, size=1, copy=True,
                                raise_if_out_of_image=False)
    assert np.all(image_bb[bb_mask] == [150, 150, 150])
    assert np.all(image_bb[~bb_mask] == [100, 100, 100])

    image_bb = bb.draw_on_image((image+100).astype(np.float32), color=[200, 200, 200], alpha=0.5, size=1,
                                copy=True, raise_if_out_of_image=False)
    assert np.sum(np.abs((image_bb - [150, 150, 150])[bb_mask])) < 0.1
    assert np.sum(np.abs((image_bb - [100, 100, 100])[~bb_mask])) < 0.1

    image_bb = bb.draw_on_image(image, color=[255, 255, 255], alpha=1.0, size=1, copy=False,
                                raise_if_out_of_image=False)
    assert np.all(image_bb[bb_mask] == [255, 255, 255])
    assert np.all(image_bb[~bb_mask] == [0, 0, 0])
    assert np.all(image[bb_mask] == [255, 255, 255])
    assert np.all(image[~bb_mask] == [0, 0, 0])

    image = np.zeros_like(image)
    bb = ia.BoundingBox(y1=-1, x1=-1, y2=2, x2=2, label=None)
    bb_mask = np.zeros(image.shape[0:2], dtype=np.bool)
    bb_mask[2, 0:3] = True
    bb_mask[0:3, 2] = True
    image_bb = bb.draw_on_image(image, color=[255, 255, 255], alpha=1.0, size=1, copy=True,
                                raise_if_out_of_image=False)
    assert np.all(image_bb[bb_mask] == [255, 255, 255])
    assert np.all(image_bb[~bb_mask] == [0, 0, 0])

    bb = ia.BoundingBox(y1=1, x1=1, y2=3, x2=3, label=None)
    bb_mask = np.zeros(image.shape[0:2], dtype=np.bool)
    bb_mask[0:5, 0:5] = True
    bb_mask[2, 2] = False
    image_bb = bb.draw_on_image(image, color=[255, 255, 255], alpha=1.0, size=2, copy=True,
                                raise_if_out_of_image=False)
    assert np.all(image_bb[bb_mask] == [255, 255, 255])
    assert np.all(image_bb[~bb_mask] == [0, 0, 0])

    bb = ia.BoundingBox(y1=-1, x1=-1, y2=1, x2=1, label=None)
    bb_mask = np.zeros(image.shape[0:2], dtype=np.bool)
    bb_mask[0:1+1, 1] = True
    bb_mask[1, 0:1+1] = True
    image_bb = bb.draw_on_image(image, color=[255, 255, 255], alpha=1.0, size=1, copy=True,
                                raise_if_out_of_image=False)
    assert np.all(image_bb[bb_mask] == [255, 255, 255])
    assert np.all(image_bb[~bb_mask] == [0, 0, 0])

    bb = ia.BoundingBox(y1=-1, x1=-1, y2=1, x2=1, label=None)
    got_exception = False
    try:
        _ = bb.draw_on_image(image, color=[255, 255, 255], alpha=1.0, size=1, copy=True,
                             raise_if_out_of_image=True)
    except Exception:
        got_exception = True
    assert got_exception is False

    bb = ia.BoundingBox(y1=-5, x1=-5, y2=-1, x2=-1, label=None)
    got_exception = False
    try:
        _ = bb.draw_on_image(image, color=[255, 255, 255], alpha=1.0, size=1, copy=True,
                             raise_if_out_of_image=True)
    except Exception:
        got_exception = True
    assert got_exception is True

    # extract_from_image()
    image = np.random.RandomState(1234).randint(0, 255, size=(10, 10, 3))
    bb = ia.BoundingBox(y1=1, y2=3, x1=1, x2=3, label=None)
    image_sub = bb.extract_from_image(image)
    assert np.array_equal(image_sub, image[1:3, 1:3, :])

    image = np.random.RandomState(1234).randint(0, 255, size=(10, 10))
    bb = ia.BoundingBox(y1=1, y2=3, x1=1, x2=3, label=None)
    image_sub = bb.extract_from_image(image)
    assert np.array_equal(image_sub, image[1:3, 1:3])

    image = np.random.RandomState(1234).randint(0, 255, size=(10, 10))
    bb = ia.BoundingBox(y1=1, y2=3, x1=1, x2=3, label=None)
    image_sub = bb.extract_from_image(image)
    assert np.array_equal(image_sub, image[1:3, 1:3])

    image = np.random.RandomState(1234).randint(0, 255, size=(10, 10, 3))
    image_pad = np.pad(image, ((0, 1), (0, 1), (0, 0)), mode="constant", constant_values=0)
    bb = ia.BoundingBox(y1=8, y2=11, x1=8, x2=11, label=None)
    image_sub = bb.extract_from_image(image)
    assert np.array_equal(image_sub, image_pad[8:11, 8:11, :])

    image = np.random.RandomState(1234).randint(0, 255, size=(10, 10))
    image_pad = np.pad(image, ((0, 1), (0, 1)), mode="constant", constant_values=0)
    bb = ia.BoundingBox(y1=8, y2=11, x1=8, x2=11, label=None)
    image_sub = bb.extract_from_image(image)
    assert np.array_equal(image_sub, image_pad[8:11, 8:11])

    image = np.random.RandomState(1234).randint(0, 255, size=(10, 10, 3))
    image_pad = np.pad(image, ((1, 0), (1, 0), (0, 0)), mode="constant", constant_values=0)
    bb = ia.BoundingBox(y1=-1, y2=3, x1=-1, x2=4, label=None)
    image_sub = bb.extract_from_image(image)
    assert np.array_equal(image_sub, image_pad[0:4, 0:5, :])

    image = np.random.RandomState(1234).randint(0, 255, size=(10, 10, 3))
    bb = ia.BoundingBox(y1=1, y2=1.99999, x1=1, x2=1.99999, label=None)
    image_sub = bb.extract_from_image(image)
    assert np.array_equal(image_sub, image[1:1+1, 1:1+1, :])

    image = np.random.RandomState(1234).randint(0, 255, size=(10, 10, 3))
    bb = ia.BoundingBox(y1=1, y2=1, x1=2, x2=4, label=None)
    image_sub = bb.extract_from_image(image)
    assert np.array_equal(image_sub, image[1:1+1, 2:4, :])

    image = np.random.RandomState(1234).randint(0, 255, size=(10, 10, 3))
    bb = ia.BoundingBox(y1=1, y2=1, x1=2, x2=2, label=None)
    image_sub = bb.extract_from_image(image)
    assert np.array_equal(image_sub, image[1:1+1, 2:2+1, :])

    # to_keypoints()
    bb = ia.BoundingBox(y1=1, y2=3, x1=1, x2=3, label=None)
    kps = bb.to_keypoints()
    assert kps[0].y == 1
    assert kps[0].x == 1
    assert kps[1].y == 1
    assert kps[1].x == 3
    assert kps[2].y == 3
    assert kps[2].x == 3
    assert kps[3].y == 3
    assert kps[3].x == 1

    # copy()
    bb = ia.BoundingBox(y1=1, y2=3, x1=1, x2=3, label="test")
    bb2 = bb.copy()
    assert bb2.y1 == 1
    assert bb2.y2 == 3
    assert bb2.x1 == 1
    assert bb2.x2 == 3
    assert bb2.label == "test"

    bb2 = bb.copy(y1=10, x1=20, y2=30, x2=40, label="test2")
    assert bb2.y1 == 10
    assert bb2.x1 == 20
    assert bb2.y2 == 30
    assert bb2.x2 == 40
    assert bb2.label == "test2"

    # deepcopy()
    bb = ia.BoundingBox(y1=1, y2=3, x1=1, x2=3, label=["test"])
    bb2 = bb.deepcopy()
    assert bb2.y1 == 1
    assert bb2.y2 == 3
    assert bb2.x1 == 1
    assert bb2.x2 == 3
    assert bb2.label[0] == "test"

    # BoundingBox_repr()
    bb = ia.BoundingBox(y1=1, y2=3, x1=1, x2=3, label=None)
    assert bb.__repr__() == "BoundingBox(x1=1.0000, y1=1.0000, x2=3.0000, y2=3.0000, label=None)"

    # test_BoundingBox_str()
    bb = ia.BoundingBox(y1=1, y2=3, x1=1, x2=3, label=None)
    assert bb.__str__() == "BoundingBox(x1=1.0000, y1=1.0000, x2=3.0000, y2=3.0000, label=None)"


def test_BoundingBoxesOnImage():
    reseed()

    # test height/width
    bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=45, label=None)
    bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))
    assert bbsoi.height == 40
    assert bbsoi.width == 50

    bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=45, label=None)
    bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=np.zeros((40, 50, 3), dtype=np.uint8))
    assert bbsoi.height == 40
    assert bbsoi.width == 50

    # empty
    bb = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bbsoi = ia.BoundingBoxesOnImage([bb], shape=(40, 50, 3))
    assert not bbsoi.empty

    bbsoi = ia.BoundingBoxesOnImage([], shape=(40, 50, 3))
    assert bbsoi.empty

    # on()
    bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=45, label=None)
    bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=np.zeros((40, 50, 3), dtype=np.uint8))

    bbsoi_projected = bbsoi.on((40, 50))
    assert bbsoi_projected.bounding_boxes[0].y1 == 10
    assert bbsoi_projected.bounding_boxes[0].x1 == 20
    assert bbsoi_projected.bounding_boxes[0].y2 == 30
    assert bbsoi_projected.bounding_boxes[0].x2 == 40
    assert bbsoi_projected.bounding_boxes[1].y1 == 15
    assert bbsoi_projected.bounding_boxes[1].x1 == 25
    assert bbsoi_projected.bounding_boxes[1].y2 == 35
    assert bbsoi_projected.bounding_boxes[1].x2 == 45

    bbsoi_projected = bbsoi.on((40*2, 50*2, 3))
    assert bbsoi_projected.bounding_boxes[0].y1 == 10*2
    assert bbsoi_projected.bounding_boxes[0].x1 == 20*2
    assert bbsoi_projected.bounding_boxes[0].y2 == 30*2
    assert bbsoi_projected.bounding_boxes[0].x2 == 40*2
    assert bbsoi_projected.bounding_boxes[1].y1 == 15*2
    assert bbsoi_projected.bounding_boxes[1].x1 == 25*2
    assert bbsoi_projected.bounding_boxes[1].y2 == 35*2
    assert bbsoi_projected.bounding_boxes[1].x2 == 45*2

    bbsoi_projected = bbsoi.on(np.zeros((40*2, 50*2, 3), dtype=np.uint8))
    assert bbsoi_projected.bounding_boxes[0].y1 == 10*2
    assert bbsoi_projected.bounding_boxes[0].x1 == 20*2
    assert bbsoi_projected.bounding_boxes[0].y2 == 30*2
    assert bbsoi_projected.bounding_boxes[0].x2 == 40*2
    assert bbsoi_projected.bounding_boxes[1].y1 == 15*2
    assert bbsoi_projected.bounding_boxes[1].x1 == 25*2
    assert bbsoi_projected.bounding_boxes[1].y2 == 35*2
    assert bbsoi_projected.bounding_boxes[1].x2 == 45*2

    # from_xyxy_array()
    bbsoi = ia.BoundingBoxesOnImage.from_xyxy_array(
        np.float32([
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 2.0, 3.0, 4.0]
        ]),
        shape=(40, 50, 3)
    )
    assert len(bbsoi.bounding_boxes) == 2
    assert np.allclose(bbsoi.bounding_boxes[0].x1, 0.0)
    assert np.allclose(bbsoi.bounding_boxes[0].y1, 0.0)
    assert np.allclose(bbsoi.bounding_boxes[0].x2, 1.0)
    assert np.allclose(bbsoi.bounding_boxes[0].y2, 1.0)
    assert np.allclose(bbsoi.bounding_boxes[1].x1, 1.0)
    assert np.allclose(bbsoi.bounding_boxes[1].y1, 2.0)
    assert np.allclose(bbsoi.bounding_boxes[1].x2, 3.0)
    assert np.allclose(bbsoi.bounding_boxes[1].y2, 4.0)
    assert bbsoi.shape == (40, 50, 3)

    bbsoi = ia.BoundingBoxesOnImage.from_xyxy_array(
        np.int32([
            [0, 0, 1, 1],
            [1, 2, 3, 4]
        ]),
        shape=(40, 50, 3)
    )
    assert len(bbsoi.bounding_boxes) == 2
    assert np.allclose(bbsoi.bounding_boxes[0].x1, 0.0)
    assert np.allclose(bbsoi.bounding_boxes[0].y1, 0.0)
    assert np.allclose(bbsoi.bounding_boxes[0].x2, 1.0)
    assert np.allclose(bbsoi.bounding_boxes[0].y2, 1.0)
    assert np.allclose(bbsoi.bounding_boxes[1].x1, 1.0)
    assert np.allclose(bbsoi.bounding_boxes[1].y1, 2.0)
    assert np.allclose(bbsoi.bounding_boxes[1].x2, 3.0)
    assert np.allclose(bbsoi.bounding_boxes[1].y2, 4.0)
    assert bbsoi.shape == (40, 50, 3)

    bbsoi = ia.BoundingBoxesOnImage.from_xyxy_array(
        np.zeros((0, 4), dtype=np.float32),
        shape=(40, 50, 3)
    )
    assert len(bbsoi.bounding_boxes) == 0
    assert bbsoi.shape == (40, 50, 3)

    # to_xyxy_array()
    xyxy_arr = np.float32([
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 2.0, 3.0, 4.0]
    ])
    bbsoi = ia.BoundingBoxesOnImage.from_xyxy_array(xyxy_arr, shape=(40, 50, 3))
    xyxy_arr_out = bbsoi.to_xyxy_array()
    assert np.allclose(xyxy_arr, xyxy_arr_out)
    assert xyxy_arr_out.dtype == np.float32

    xyxy_arr_out = bbsoi.to_xyxy_array(dtype=np.int32)
    assert np.allclose(xyxy_arr.astype(np.int32), xyxy_arr_out)
    assert xyxy_arr_out.dtype == np.int32

    xyxy_arr_out = ia.BoundingBoxesOnImage([], shape=(40, 50, 3)).to_xyxy_array(dtype=np.int32)
    assert xyxy_arr_out.shape == (0, 4)

    # draw_on_image()
    bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=45, label=None)
    bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))
    image = bbsoi.draw_on_image(np.zeros(bbsoi.shape, dtype=np.uint8), color=[0, 255, 0], alpha=1.0, size=1,
                                copy=True, raise_if_out_of_image=False)
    assert np.all(image[10-1, 20-1, :] == [0, 0, 0])
    assert np.all(image[10-1, 20-0, :] == [0, 0, 0])
    assert np.all(image[10-0, 20-1, :] == [0, 0, 0])
    assert np.all(image[10-0, 20-0, :] == [0, 255, 0])
    assert np.all(image[10+1, 20+1, :] == [0, 0, 0])

    assert np.all(image[30-1, 40-1, :] == [0, 0, 0])
    assert np.all(image[30+1, 40-0, :] == [0, 0, 0])
    assert np.all(image[30+0, 40+1, :] == [0, 0, 0])
    assert np.all(image[30+0, 40+0, :] == [0, 255, 0])
    assert np.all(image[30+1, 40+1, :] == [0, 0, 0])

    assert np.all(image[15-1, 25-1, :] == [0, 0, 0])
    assert np.all(image[15-1, 25-0, :] == [0, 0, 0])
    assert np.all(image[15-0, 25-1, :] == [0, 0, 0])
    assert np.all(image[15-0, 25-0, :] == [0, 255, 0])
    assert np.all(image[15+1, 25+1, :] == [0, 0, 0])

    assert np.all(image[35-1, 45-1, :] == [0, 0, 0])
    assert np.all(image[35+1, 45+0, :] == [0, 0, 0])
    assert np.all(image[35+0, 45+1, :] == [0, 0, 0])
    assert np.all(image[35+0, 45+0, :] == [0, 255, 0])
    assert np.all(image[35+1, 45+1, :] == [0, 0, 0])

    # remove_out_of_image()
    bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=51, label=None)
    bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))
    bbsoi_slim = bbsoi.remove_out_of_image(fully=True, partly=True)
    assert len(bbsoi_slim.bounding_boxes) == 1
    assert bbsoi_slim.bounding_boxes[0] == bb1

    # clip_out_of_image()
    bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=51, label=None)
    bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))
    eps = np.finfo(np.float32).eps
    bbsoi_clip = bbsoi.clip_out_of_image()
    assert len(bbsoi_clip.bounding_boxes) == 2
    assert bbsoi_clip.bounding_boxes[0].y1 == 10
    assert bbsoi_clip.bounding_boxes[0].x1 == 20
    assert bbsoi_clip.bounding_boxes[0].y2 == 30
    assert bbsoi_clip.bounding_boxes[0].x2 == 40
    assert bbsoi_clip.bounding_boxes[1].y1 == 15
    assert bbsoi_clip.bounding_boxes[1].x1 == 25
    assert bbsoi_clip.bounding_boxes[1].y2 == 35
    assert 50 - 2*eps < bbsoi_clip.bounding_boxes[1].x2 < 50

    # shift()
    bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=51, label=None)
    bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))
    bbsoi_shifted = bbsoi.shift(right=1)
    assert len(bbsoi_shifted.bounding_boxes) == 2
    assert bbsoi_shifted.bounding_boxes[0].y1 == 10
    assert bbsoi_shifted.bounding_boxes[0].x1 == 20 - 1
    assert bbsoi_shifted.bounding_boxes[0].y2 == 30
    assert bbsoi_shifted.bounding_boxes[0].x2 == 40 - 1
    assert bbsoi_shifted.bounding_boxes[1].y1 == 15
    assert bbsoi_shifted.bounding_boxes[1].x1 == 25 - 1
    assert bbsoi_shifted.bounding_boxes[1].y2 == 35
    assert bbsoi_shifted.bounding_boxes[1].x2 == 51 - 1

    # copy()
    bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=51, label=None)
    bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))
    bbsoi_copy = bbsoi.copy()
    assert len(bbsoi.bounding_boxes) == 2
    assert bbsoi_copy.bounding_boxes[0].y1 == 10
    assert bbsoi_copy.bounding_boxes[0].x1 == 20
    assert bbsoi_copy.bounding_boxes[0].y2 == 30
    assert bbsoi_copy.bounding_boxes[0].x2 == 40
    assert bbsoi_copy.bounding_boxes[1].y1 == 15
    assert bbsoi_copy.bounding_boxes[1].x1 == 25
    assert bbsoi_copy.bounding_boxes[1].y2 == 35
    assert bbsoi_copy.bounding_boxes[1].x2 == 51

    bbsoi.bounding_boxes[0].y1 = 0
    assert bbsoi_copy.bounding_boxes[0].y1 == 0

    # deepcopy()
    bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=51, label=None)
    bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))
    bbsoi_copy = bbsoi.deepcopy()
    assert len(bbsoi.bounding_boxes) == 2
    assert bbsoi_copy.bounding_boxes[0].y1 == 10
    assert bbsoi_copy.bounding_boxes[0].x1 == 20
    assert bbsoi_copy.bounding_boxes[0].y2 == 30
    assert bbsoi_copy.bounding_boxes[0].x2 == 40
    assert bbsoi_copy.bounding_boxes[1].y1 == 15
    assert bbsoi_copy.bounding_boxes[1].x1 == 25
    assert bbsoi_copy.bounding_boxes[1].y2 == 35
    assert bbsoi_copy.bounding_boxes[1].x2 == 51

    bbsoi.bounding_boxes[0].y1 = 0
    assert bbsoi_copy.bounding_boxes[0].y1 == 10

    # repr() / str()
    bb1 = ia.BoundingBox(y1=10, x1=20, y2=30, x2=40, label=None)
    bb2 = ia.BoundingBox(y1=15, x1=25, y2=35, x2=51, label=None)
    bbsoi = ia.BoundingBoxesOnImage([bb1, bb2], shape=(40, 50, 3))
    bb1_expected = "BoundingBox(x1=20.0000, y1=10.0000, x2=40.0000, y2=30.0000, label=None)"
    bb2_expected = "BoundingBox(x1=25.0000, y1=15.0000, x2=51.0000, y2=35.0000, label=None)"
    expected = "BoundingBoxesOnImage([%s, %s], shape=(40, 50, 3))" % (bb1_expected, bb2_expected)
    assert bbsoi.__repr__() == bbsoi.__str__() == expected
