from __future__ import print_function, division, absolute_import

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
import os
try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
import cv2
import imageio

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import random as iarandom
from imgaug.testutils import reseed, TemporaryDirectory
import imgaug.augmenters.debug as debuglib


class Test_draw_debug_image(unittest.TestCase):
    @classmethod
    def _find_in_image_avg_diff(cls, find_image, in_image):
        res = cv2.matchTemplate(in_image, find_image, cv2.TM_SQDIFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        top_left = min_loc
        bottom_right = (top_left[0] + find_image.shape[1],
                        top_left[1] + find_image.shape[0])
        image_found = in_image[top_left[1]:bottom_right[1],
                               top_left[0]:bottom_right[0],
                               :]
        diff = np.abs(image_found.astype(np.float32)
                      - find_image.astype(np.float32))
        return np.average(diff)

    @classmethod
    def _image_contains(cls, find_image, in_image, threshold=2.0):
        return cls._find_in_image_avg_diff(find_image, in_image) <= threshold

    def test_one_image(self):
        rng = iarandom.RNG(0)
        image = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)

        debug_image = iaa.draw_debug_image([image])

        assert self._image_contains(image, debug_image)

    def test_two_images(self):
        rng = iarandom.RNG(0)
        images = rng.integers(0, 256, size=(2, 256, 256, 3), dtype=np.uint8)

        debug_image = iaa.draw_debug_image(images)

        assert self._image_contains(images[0, ...], debug_image)
        assert self._image_contains(images[1, ...], debug_image)

    def test_two_images_of_different_sizes(self):
        rng = iarandom.RNG(0)
        image1 = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
        image2 = rng.integers(0, 256, size=(512, 256, 3), dtype=np.uint8)

        debug_image = iaa.draw_debug_image([image1, image2])

        assert self._image_contains(image1, debug_image)
        assert self._image_contains(image2, debug_image)

    def test_two_images_and_heatmaps(self):
        rng = iarandom.RNG(0)
        images = rng.integers(0, 256, size=(2, 256, 256, 3), dtype=np.uint8)
        heatmap = np.zeros((256, 256, 1), dtype=np.float32)
        heatmap[128-25:128+25, 128-25:128+25] = 1.0
        heatmap1 = ia.HeatmapsOnImage(np.copy(heatmap), shape=images[0].shape)
        heatmap2 = ia.HeatmapsOnImage(1.0 - heatmap, shape=images[1].shape)
        image1_w_overlay = heatmap1.draw_on_image(images[0])[0]
        image2_w_overlay = heatmap2.draw_on_image(images[1])[0]

        debug_image = iaa.draw_debug_image(images,
                                           heatmaps=[heatmap1, heatmap2])

        assert self._image_contains(images[0, ...], debug_image)
        assert self._image_contains(images[1, ...], debug_image)
        assert self._image_contains(image1_w_overlay, debug_image)
        assert self._image_contains(image2_w_overlay, debug_image)

    def test_two_images_and_segmaps(self):
        rng = iarandom.RNG(0)
        images = rng.integers(0, 256, size=(2, 256, 256, 3), dtype=np.uint8)
        sm1 = np.zeros((256, 256, 1), dtype=np.int32)
        sm1[128-25:128+25, 128-25:128+25] = 1
        sm2 = np.zeros((256, 256, 1), dtype=np.int32)
        sm2[64-25:64+25, 64-25:64+25] = 2
        sm2[192-25:192+25, 192-25:192+25] = 3
        segmap1 = ia.SegmentationMapsOnImage(sm1, shape=images[0].shape)
        segmap2 = ia.SegmentationMapsOnImage(sm2, shape=images[1].shape)
        image1_w_overlay = segmap1.draw_on_image(images[0],
                                                 draw_background=True)[0]
        image2_w_overlay = segmap2.draw_on_image(images[1],
                                                 draw_background=True)[0]

        debug_image = iaa.draw_debug_image(images,
                                           segmentation_maps=[segmap1, segmap2])

        assert self._image_contains(images[0, ...], debug_image)
        assert self._image_contains(images[1, ...], debug_image)
        assert self._image_contains(image1_w_overlay, debug_image)
        assert self._image_contains(image2_w_overlay, debug_image)

    def test_two_images_and_heatmaps__map_size_differs_from_image(self):
        rng = iarandom.RNG(0)
        images = rng.integers(0, 256, size=(2, 256, 256, 3), dtype=np.uint8)
        heatmap = np.zeros((128, 128, 1), dtype=np.float32)
        heatmap[64-25:64+25, 64-25:64+25] = 1.0
        heatmap1 = ia.HeatmapsOnImage(np.copy(heatmap), shape=images[0].shape)
        heatmap2 = ia.HeatmapsOnImage(1.0 - heatmap, shape=images[1].shape)
        image1_w_overlay = heatmap1.draw_on_image(images[0])[0]
        image2_w_overlay = heatmap2.draw_on_image(images[1])[0]

        debug_image = iaa.draw_debug_image(images,
                                           heatmaps=[heatmap1, heatmap2])

        assert self._image_contains(images[0, ...], debug_image)
        assert self._image_contains(images[1, ...], debug_image)
        assert self._image_contains(image1_w_overlay, debug_image)
        assert self._image_contains(image2_w_overlay, debug_image)

    def test_two_images_and_heatmaps__multichannel(self):
        rng = iarandom.RNG(0)
        images = rng.integers(0, 256, size=(2, 256, 256, 3), dtype=np.uint8)
        heatmap = np.zeros((256, 256, 2), dtype=np.float32)
        heatmap[100-25:100+25, 100-25:100+25, 0] = 1.0
        heatmap[200-25:200+25, 200-25:200+25, 1] = 1.0
        heatmap1 = ia.HeatmapsOnImage(np.copy(heatmap), shape=images[0].shape)
        heatmap2 = ia.HeatmapsOnImage(1.0 - heatmap, shape=images[1].shape)
        image1_w_overlay_c1, image1_w_overlay_c2 = \
            heatmap1.draw_on_image(images[0])
        image2_w_overlay_c1, image2_w_overlay_c2 = \
            heatmap2.draw_on_image(images[1])

        debug_image = iaa.draw_debug_image(images, heatmaps=[heatmap1, heatmap2])

        assert self._image_contains(images[0, ...], debug_image)
        assert self._image_contains(images[1, ...], debug_image)
        assert self._image_contains(image1_w_overlay_c1, debug_image)
        assert self._image_contains(image1_w_overlay_c2, debug_image)
        assert self._image_contains(image2_w_overlay_c1, debug_image)
        assert self._image_contains(image2_w_overlay_c2, debug_image)

    def test_two_images_and_keypoints(self):
        rng = iarandom.RNG(0)
        images = rng.integers(0, 256, size=(2, 256, 256, 3), dtype=np.uint8)
        kps = []
        for x in np.linspace(0, 256, 10):
            for y in np.linspace(0, 256, 10):
                kps.append(ia.Keypoint(x=x, y=y))
        kpsoi1 = ia.KeypointsOnImage(kps, shape=images[0].shape)
        kpsoi2 = kpsoi1.shift(x=20)
        image1_w_overlay = kpsoi1.draw_on_image(images[0])
        image2_w_overlay = kpsoi2.draw_on_image(images[1])

        debug_image = iaa.draw_debug_image(images, keypoints=[kpsoi1, kpsoi2])

        assert self._image_contains(images[0, ...], debug_image)
        assert self._image_contains(images[1, ...], debug_image)
        assert self._image_contains(image1_w_overlay, debug_image)
        assert self._image_contains(image2_w_overlay, debug_image)

    def test_two_images_and_bounding_boxes(self):
        rng = iarandom.RNG(0)
        images = rng.integers(0, 256, size=(2, 256, 256, 3), dtype=np.uint8)
        bbs = []
        for x in np.linspace(0, 256, 5):
            for y in np.linspace(0, 256, 5):
                bbs.append(ia.BoundingBox(x1=x, y1=y, x2=x+20, y2=y+20))
        bbsoi1 = ia.BoundingBoxesOnImage(bbs, shape=images[0].shape)
        bbsoi2 = bbsoi1.shift(x=20)
        image1_w_overlay = bbsoi1.draw_on_image(images[0])
        image2_w_overlay = bbsoi2.draw_on_image(images[1])

        debug_image = iaa.draw_debug_image(images,
                                           bounding_boxes=[bbsoi1, bbsoi2])

        assert self._image_contains(images[0, ...], debug_image)
        assert self._image_contains(images[1, ...], debug_image)
        assert self._image_contains(image1_w_overlay, debug_image)
        assert self._image_contains(image2_w_overlay, debug_image)

    def test_two_images_and_polygons(self):
        rng = iarandom.RNG(0)
        images = rng.integers(0, 256, size=(2, 32, 32, 3), dtype=np.uint8)
        polys = []
        for x in np.linspace(0, 256, 4):
            for y in np.linspace(0, 256, 4):
                polys.append(ia.Polygon([(x, y), (x+20, y), (x+20, y+20),
                                         (x, y+20)]))
        psoi1 = ia.PolygonsOnImage(polys, shape=images[0].shape)
        psoi2 = psoi1.shift(x=20)
        image1_w_overlay = psoi1.draw_on_image(images[0])
        image2_w_overlay = psoi2.draw_on_image(images[1])

        debug_image = iaa.draw_debug_image(images,
                                           polygons=[psoi1, psoi2])

        assert self._image_contains(images[0, ...], debug_image)
        assert self._image_contains(images[1, ...], debug_image)
        assert self._image_contains(image1_w_overlay, debug_image)
        assert self._image_contains(image2_w_overlay, debug_image)

    def test_two_images_and_line_strings(self):
        rng = iarandom.RNG(0)
        images = rng.integers(0, 256, size=(2, 32, 32, 3), dtype=np.uint8)
        ls = []
        for x in np.linspace(0, 256, 4):
            for y in np.linspace(0, 256, 4):
                ls.append(ia.LineString([(x, y), (x+20, y), (x+20, y+20),
                                         (x, y+20)]))
        lsoi1 = ia.LineStringsOnImage(ls, shape=images[0].shape)
        lsoi2 = lsoi1.deepcopy()
        image1_w_overlay = lsoi1.draw_on_image(images[0])
        image2_w_overlay = lsoi2.draw_on_image(images[1])

        debug_image = iaa.draw_debug_image(images,
                                           line_strings=[lsoi1, lsoi2])

        assert self._image_contains(images[0, ...], debug_image)
        assert self._image_contains(images[1, ...], debug_image)
        assert self._image_contains(image1_w_overlay, debug_image)
        assert self._image_contains(image2_w_overlay, debug_image)

    def test_one_image_float32(self):
        rng = iarandom.RNG(0)
        image = rng.random(size=(256, 256, 3)).astype(np.float32)

        debug_image = iaa.draw_debug_image([image])

        assert self._image_contains((image * 255).astype(np.uint8),
                                    debug_image)

    def test_one_image_float32_and_heatmap(self):
        rng = iarandom.RNG(0)
        image = rng.random(size=(256, 256, 3)).astype(np.float32)
        heatmap = np.zeros((256, 256, 1), dtype=np.float32)
        heatmap[128-25:128+25, 128-25:128+25] = 1.0
        heatmap = ia.HeatmapsOnImage(heatmap, shape=image.shape)
        image1_w_overlay = heatmap.draw_on_image(
            (image*255).astype(np.uint8))[0]

        debug_image = iaa.draw_debug_image([image], heatmaps=[heatmap])

        assert self._image_contains((image * 255).astype(np.uint8), debug_image)
        assert self._image_contains(image1_w_overlay, debug_image)


class SaveDebugImageEveryNBatches(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_mocked(self):
        class _DummyDestination(debuglib._IImageDestination):
            def __init__(self):
                self.received = []

            def receive(self, image):
                self.received.append(np.copy(image))

        image = iarandom.RNG(0).integers(0, 256, size=(256, 256, 3),
                                         dtype=np.uint8)
        destination = _DummyDestination()
        aug = iaa.SaveDebugImageEveryNBatches(destination, 10)

        for _ in np.arange(20):
            _ = aug(image=image)

        expected = iaa.draw_debug_image([image])
        assert len(destination.received) == 2
        assert np.array_equal(destination.received[0], expected)
        assert np.array_equal(destination.received[1], expected)

    def test_temp_directory(self):
        with TemporaryDirectory() as folder_path:
            image = iarandom.RNG(0).integers(0, 256, size=(256, 256, 3),
                                             dtype=np.uint8)
            aug = iaa.SaveDebugImageEveryNBatches(folder_path, 10)

            for _ in np.arange(20):
                _ = aug(image=image)

            expected = iaa.draw_debug_image([image])
            path1 = os.path.join(folder_path, "batch_000000.png")
            path2 = os.path.join(folder_path, "batch_000010.png")
            path_latest = os.path.join(folder_path, "batch_latest.png")
            assert len(list(os.listdir(folder_path))) == 3
            assert os.path.isfile(path1)
            assert os.path.isfile(path2)
            assert os.path.isfile(path_latest)
            assert np.array_equal(imageio.imread(path1), expected)
            assert np.array_equal(imageio.imread(path2), expected)
            assert np.array_equal(imageio.imread(path_latest), expected)

    def test_pickleable(self):
        shape = (16, 16, 3)
        image = np.mod(np.arange(int(np.prod(shape))), 256).astype(np.uint8)
        image = image.reshape(shape)

        with TemporaryDirectory() as folder_path:
            path1 = os.path.join(folder_path, "batch_000000.png")
            path2 = os.path.join(folder_path, "batch_000010.png")

            augmenter = iaa.SaveDebugImageEveryNBatches(folder_path, 10)
            augmenter_pkl = pickle.loads(pickle.dumps(augmenter, protocol=-1))

            # save two images via augmenter without pickling
            for _ in np.arange(20):
                _ = augmenter(image=image)

            img11 = imageio.imread(path1)
            img12 = imageio.imread(path2)

            # reset folder content
            os.remove(path1)
            os.remove(path2)

            # save two images via augmenter that was pickled
            for _ in np.arange(20):
                _ = augmenter_pkl(image=image)

            img21 = imageio.imread(path1)
            img22 = imageio.imread(path2)

            # compare the two images of original/pickled augmenters
            assert np.array_equal(img11, img21)
            assert np.array_equal(img12, img22)
