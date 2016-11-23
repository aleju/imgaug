"""Tests functionality of the ImageAugmenter class."""
from __future__ import print_function

# make sure that ImageAugmenter can be imported from parent directory
if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import unittest
import numpy as np
from ImageAugmenter import ImageAugmenter
import random
from skimage import data

random.seed(123456789)
np.random.seed(123456789)

class TestImageAugmenter(unittest.TestCase):
    """Tests functionality of the ImageAugmenter class."""

    def test_rotation(self):
        """Test rotation of 90 degrees on an image that should change
        upon rotation."""
        image_before = [[0, 255, 0],
                        [0, 255, 0],
                        [0, 255, 0]]
        image_target = [[  0,   0,   0],
                        [1.0, 1.0, 1.0],
                        [  0,   0,   0]]
        images = np.array([image_before]).astype(np.uint8)

        augmenter = ImageAugmenter(3, 3, rotation_deg=(90, 90))

        image_after = augmenter.augment_batch(images)[0]
        self.assertTrue(np.allclose(image_target, image_after))

    def test_rotation_invariant(self):
        """Test rotation of -90 to 90 degrees on an rotation invariant image."""
        image_before = [[0,   0, 0],
                        [0, 255, 0],
                        [0,   0, 0]]
        image_target = [[0,   0, 0],
                        [0, 1.0, 0],
                        [0,   0, 0]]
        images = np.array([image_before]).astype(np.uint8)

        # random rotation of up to 180 degress
        augmenter = ImageAugmenter(3, 3, rotation_deg=180)

        # all must be similar to target
        nb_similar = 0
        for _ in range(100):
            image_after = augmenter.augment_batch(images)[0]
            # some tolerance here - interpolation problems can let the image
            # change a bit, even though it should be invariant to rotations
            if np.allclose(image_target, image_after, atol=0.1):
                nb_similar += 1
        self.assertEquals(nb_similar, 100)

    def test_scaling(self):
        """Rough test for zooming/scaling (only zoom in / scaling >1.0).
        The test is rough, because interpolation problems make the result
        of scaling on synthetic images rather hard to predict (and unintuitive).
        """

        size_x = 4
        size_y = 4

        # a 4x4 image of which the center 3x3 pixels are bright white,
        # everything else black
        image_before = np.zeros((size_y, size_x))
        image_before[1:size_y-1, 1:size_x-1] = 255

        images = np.array([image_before]).astype(np.uint8)

        # about 200% zoom in
        augmenter = ImageAugmenter(size_x, size_y, scale_to_percent=(1.99, 1.99),
                                   scale_axis_equally=True)

        image_after = augmenter.augment_batch(images)[0]
        # we scale positively (zoom in), therefor we expect the center bright
        # spot to grow, resulting in a higher total brightness
        self.assertTrue(np.sum(image_after) > np.sum(image_before)/255)

    def test_shear(self):
        """Very rough test of shear: It simply measures whether image tend
        to be significantly different after shear (any change)."""

        image_before = [[0, 255, 0],
                        [0, 255, 0],
                        [0, 255, 0]]
        image_target = [[0, 1.0, 0],
                        [0, 1.0, 0],
                        [0, 1.0, 0]]
        images = np.array([image_before]).astype(np.uint8)
        augmenter = ImageAugmenter(3, 3, shear_deg=50)

        # the majority should be different from the source image
        nb_different = 0
        nb_augment = 1000
        for _ in range(nb_augment):
            image_after = augmenter.augment_batch(images)[0]
            if not np.allclose(image_target, image_after):
                nb_different += 1
        self.assertTrue(nb_different > nb_augment*0.9)

    def test_translation_x(self):
        """Testing translation on the x-axis."""
        #image_before = np.zeros((2, 2), dtype=np.uint8)
        image_before = [[255,   0],
                        [255,   0]]
        #image_after = np.zeros((2, 2), dtype=np.float32)
        image_target = [[0, 1.0],
                        [0, 1.0]]
        images = np.array([image_before]).astype(np.uint8)
        augmenter = ImageAugmenter(2, 2, translation_x_px=(1,1))

        # all must be similar
        for _ in range(100):
            image_after = augmenter.augment_batch(images)[0]
            self.assertTrue(np.allclose(image_target, image_after))

    def test_translation_y(self):
        """Testing translation on the y-axis."""
        image_before = [[  0,   0],
                        [255, 255]]
        image_target = [[1.0, 1.0],
                        [  0,   0]]
        images = np.array([image_before]).astype(np.uint8)
        # translate always by -1px on y-axis
        augmenter = ImageAugmenter(2, 2, translation_y_px=(-1,-1))

        # all must be similar
        for _ in range(100):
            image_after = augmenter.augment_batch(images)[0]
            self.assertTrue(np.allclose(image_target, image_after))

    def test_single_channel(self):
        """Tests images with channels (e.g. RGB channels)."""
        # One single channel
        # channel is last axis
        # test by translating an image with one channel on the x-axis (1 px)
        image_before = np.zeros((2, 2, 1), dtype=np.uint8)
        image_before[0, 0, 0] = 255
        image_before[1, 0, 0] = 255

        image_target = np.zeros((2, 2, 1), dtype=np.float32)
        image_target[0, 1, 0] = 1.0
        image_target[1, 1, 0] = 1.0

        images = np.array([image_before]).astype(np.uint8)
        augmenter = ImageAugmenter(2, 2, translation_x_px=(1,1))

        # all must be similar
        for _ in range(100):
            image_after = augmenter.augment_batch(images)[0]
            self.assertTrue(np.allclose(image_target, image_after))

        # One single channel
        # channel is first axis
        # test by translating an image with one channel on the x-axis (1 px)
        image_before = np.zeros((1, 2, 2), dtype=np.uint8)
        image_before[0] = [[255, 0],
                           [255, 0]]

        image_target = np.zeros((1, 2, 2), dtype=np.float32)
        image_target[0] = [[0, 1.0],
                           [0, 1.0]]

        images = np.array([image_before]).astype(np.uint8)
        augmenter = ImageAugmenter(2, 2, translation_x_px=(1,1),
                                   channel_is_first_axis=True)

        # all must be similar
        for _ in range(100):
            image_after = augmenter.augment_batch(images)[0]
            self.assertTrue(np.allclose(image_target, image_after))

    def test_two_channels(self):
        """Tests augmentation of images with two channels (either first or last
        axis of each image). Tested using x-translation."""

        # -----------------------------------------------
        # two channels,
        # channel is the FIRST axis of each image
        # -----------------------------------------------
        augmenter = ImageAugmenter(2, 2, translation_y_px=(0,1),
                                   channel_is_first_axis=True)

        image_before = np.zeros((2, 2, 2)).astype(np.uint8)
        # 1st channel: top row white, bottom row black
        image_before[0][0][0] = 255
        image_before[0][0][1] = 255
        image_before[0][1][0] = 0
        image_before[0][1][1] = 0

        # 2nd channel: top right corner white, everything else black
        image_before[1][0][0] = 0
        image_before[1][0][1] = 255
        image_before[1][1][0] = 0
        image_before[1][1][1] = 0
        #            ^        channel
        #               ^     y (row)
        #                  ^  x (column)

        image_target = np.zeros((2, 2, 2)).astype(np.float32)
        # 1st channel: bottom row white, bottom row black
        image_target[0][0][0] = 0
        image_target[0][0][1] = 0
        image_target[0][1][0] = 1.0
        image_target[0][1][1] = 1.0

        # 2nd channel: bottom right corner white, everything else black
        image_target[1][0][0] = 0
        image_target[1][0][1] = 0
        image_target[1][1][0] = 0
        image_target[1][1][1] = 1.0

        nb_augment = 1000
        image = np.array([image_before]).astype(np.uint8)
        images = np.resize(image, (nb_augment, 2, 2, 2))
        images_augmented = augmenter.augment_batch(images)

        nb_similar = 0
        for image_after in images_augmented:
            if np.allclose(image_target, image_after):
                nb_similar += 1
        self.assertTrue(nb_similar > (nb_augment*0.4) and nb_similar < (nb_augment*0.6))

        # -----------------------------------------------
        # two channels,
        # channel is the LAST axis of each image
        # -----------------------------------------------
        augmenter = ImageAugmenter(2, 2, translation_y_px=(0,1),
                                   channel_is_first_axis=False)

        image_before = np.zeros((2, 2, 2)).astype(np.uint8)
        # 1st channel: top row white, bottom row black
        image_before[0][0][0] = 255
        image_before[0][1][0] = 255
        image_before[1][0][0] = 0
        image_before[1][1][0] = 0

        # 2nd channel: top right corner white, everything else black
        image_before[0][0][1] = 0
        image_before[0][1][1] = 255
        image_before[1][0][1] = 0
        image_before[1][1][1] = 0
        #            ^        y
        #               ^     x
        #                  ^  channel

        image_target = np.zeros((2, 2, 2)).astype(np.float32)
        # 1st channel: bottom row white, bottom row black
        image_target[0][0][0] = 0
        image_target[0][1][0] = 0
        image_target[1][0][0] = 1.0
        image_target[1][1][0] = 1.0

        # 2nd channel: bottom right corner white, everything else black
        image_target[0][0][1] = 0
        image_target[0][1][1] = 0
        image_target[1][0][1] = 0
        image_target[1][1][1] = 1.0

        nb_augment = 1000
        image = np.array([image_before]).astype(np.uint8)
        images = np.resize(image, (nb_augment, 2, 2, 2))
        images_augmented = augmenter.augment_batch(images)

        nb_similar = 0
        for image_after in images_augmented:
            if np.allclose(image_target, image_after):
                nb_similar += 1
        self.assertTrue(nb_similar > (nb_augment*0.4) and nb_similar < (nb_augment*0.6))

    def test_transform_channels_unequally(self):
        """Tests whether 2 or more channels can be augmented non-identically
        at the same time.

        E.g. channel 0 is rotated by 20 degress, channel 1 (of the same image)
        is rotated by 5 degrees.
        """
        # two channels, channel is first axis of each image
        augmenter = ImageAugmenter(3, 3, translation_x_px=(0,1),
                                   transform_channels_equally=False,
                                   channel_is_first_axis=True)

        image_before = np.zeros((2, 3, 3)).astype(np.uint8)
        image_before[0] = [[255,   0,   0],
                           [  0,   0,   0],
                           [  0,   0,   0]]

        image_before[1] = [[  0,   0,   0],
                           [  0,   0,   0],
                           [  0, 255,   0]]
        #            ^ channel

        image_target = np.zeros((2, 3, 3)).astype(np.float32)
        image_target[0] = [[  0, 1.0,   0],
                           [  0,   0,   0],
                           [  0,   0,   0]]

        image_target[1] = [[  0,   0,   0],
                           [  0,   0,   0],
                           [  0,   0, 1.0]]

        nb_similar_channel_0 = 0
        nb_similar_channel_1 = 0
        nb_equally_transformed = 0
        #nb_unequally_transformed = 0

        nb_augment = 1000
        image = np.array([image_before]).astype(np.uint8)
        images = np.resize(image, (nb_augment, 2, 3, 3))
        images_augmented = augmenter.augment_batch(images)

        # augment 1000 times and count how often the channels were transformed
        # in equal or unequal ways.
        for image_after in images_augmented:
            similar_channel_0 = np.allclose(image_target[0], image_after[0])
            similar_channel_1 = np.allclose(image_target[1], image_after[1])
            if similar_channel_0:
                nb_similar_channel_0 += 1
            if similar_channel_1:
                nb_similar_channel_1 += 1
            if similar_channel_0 == similar_channel_1:
                nb_equally_transformed += 1
            #else:
            #    nb_unequally_transformed += 1
        # each one should be around 50%
        self.assertTrue(nb_similar_channel_0 > 0.40*nb_augment
                        and nb_similar_channel_0 < 0.60*nb_augment)
        self.assertTrue(nb_similar_channel_1 > 0.40*nb_augment
                        and nb_similar_channel_1 < 0.60*nb_augment)
        self.assertTrue(nb_equally_transformed > 0.40*nb_augment
                        and nb_equally_transformed < 0.60*nb_augment)

    def test_no_blacks(self):
        """Test whether random augmentations can cause an image to turn
        completely black (cval=0.0), which should never happen."""
        image_before = data.camera()
        y_size, x_size = image_before.shape
        augmenter = ImageAugmenter(x_size, y_size,
                                   scale_to_percent=1.5,
                                   scale_axis_equally=False,
                                   rotation_deg=90,
                                   shear_deg=20,
                                   translation_x_px=10,
                                   translation_y_px=10)
        image_black = np.zeros(image_before.shape, dtype=np.float32)
        nb_augment = 100
        images = np.resize([image_before], (nb_augment, y_size, x_size))
        images_augmented = augmenter.augment_batch(images)
        nb_black = 0
        for image_after in images_augmented:
            if np.allclose(image_after, image_black):
                nb_black += 1
        self.assertEqual(nb_black, 0)

    def test_non_square_images(self):
        """Test whether transformation of images with unequal x and y axis sizes
        works as expected."""

        y_size = 11
        x_size = 4
        image_before = np.zeros((y_size, x_size), dtype=np.uint8)
        image_target = np.zeros((y_size, x_size), dtype=np.float32)

        # place a bright white line in the center (of the y-axis, so left to right)
        # Augmenter will move it up by 2 (translation on y by -2)
        y_line_pos = int(y_size/2) + 1
        for x_pos in range(x_size):
            image_before[y_line_pos][x_pos] = 255
            image_target[y_line_pos - 2][x_pos] = 1.0

        augmenter = ImageAugmenter(x_size, y_size, translation_y_px=(-2,-2))
        nb_augment = 100
        images = np.resize([image_before], (nb_augment, y_size, x_size))
        images_augmented = augmenter.augment_batch(images)
        nb_similar = 0
        for image_after in images_augmented:
            if np.allclose(image_after, image_target):
                nb_similar += 1
        self.assertEqual(nb_augment, nb_similar)

    def test_no_information_leaking(self):
        """Tests whether the image provided to augment_batch() is changed
        instead of only simply returned in the changed form (leaking
        information / hidden sideffects)."""
        image_before = [[255,   0, 255,   0, 255],
                        [  0, 255,   0, 255,   0],
                        [255, 255, 255, 255, 255],
                        [  0, 255,   0, 255,   0],
                        [255,   0, 255,   0, 255]]
        image_before = np.array(image_before, dtype=np.uint8)
        image_before_copy = np.copy(image_before)
        nb_augment = 100
        images = np.resize([image_before], (nb_augment, 5, 5))
        augmenter = ImageAugmenter(5, 5,
                                   hflip=True, vflip=True,
                                   scale_to_percent=1.5,
                                   rotation_deg=25, shear_deg=10,
                                   translation_x_px=5, translation_y_px=5)
        images_after = augmenter.augment_batch(images)
        self.assertTrue(np.array_equal(image_before, image_before_copy))

    def test_horizontal_flipping(self):
        """Tests horizontal flipping of images (mirror on y-axis)."""

        image_before = [[255,   0,   0],
                        [  0, 255, 255],
                        [  0,   0, 255]]
        image_before = np.array(image_before, dtype=np.uint8)
        image_target = [[  0,   0,  1.0],
                        [1.0, 1.0,    0],
                        [1.0,   0,    0]]
        image_target = np.array(image_target, dtype=np.float32)
        nb_augment = 1000
        images = np.resize([image_before], (nb_augment, 3, 3))

        # Test using just "False" for hflip (should be exactly 0%)
        augmenter = ImageAugmenter(3, 3, hflip=False)
        images_augmented = augmenter.augment_batch(images)
        nb_similar = 0
        for image_after in images_augmented:
            if np.allclose(image_after, image_target):
                nb_similar += 1
        self.assertEqual(nb_similar, 0)

        # Test using just "True" for hflip (should be ~50%)
        augmenter = ImageAugmenter(3, 3, hflip=True)
        images_augmented = augmenter.augment_batch(images)
        nb_similar = 0
        for image_after in images_augmented:
            if np.allclose(image_after, image_target):
                nb_similar += 1
        self.assertTrue(nb_similar > nb_augment*0.4 and nb_similar < nb_augment*0.6)

        # Test using a probability (float value) for hflip (hflip=0.9,
        # should be ~90%)
        augmenter = ImageAugmenter(3, 3, hflip=0.9)
        images_augmented = augmenter.augment_batch(images)
        nb_similar = 0
        for image_after in images_augmented:
            if np.allclose(image_after, image_target):
                nb_similar += 1
        self.assertTrue(nb_similar > nb_augment*0.8 and nb_similar <= nb_augment*1.0)

        # Test with multiple channels
        image_before = np.zeros((2, 3, 3), dtype=np.uint8)
        image_before[0] = [[255,   0,   0],
                           [255,   0,   0],
                           [  0,   0,   0]]
        image_before[1] = [[  0,   0,   0],
                           [255, 255,   0],
                           [  0,   0,   0]]
        image_target = np.zeros((2, 3, 3), dtype=np.float32)
        image_target[0] = [[  0,   0, 1.0],
                           [  0,   0, 1.0],
                           [  0,   0,   0]]
        image_target[1] = [[  0,   0,   0],
                           [  0, 1.0, 1.0],
                           [  0,   0,   0]]
        images = np.resize([image_before], (nb_augment, 2, 3, 3))
        augmenter = ImageAugmenter(3, 3, hflip=1.0, channel_is_first_axis=True)
        images_augmented = augmenter.augment_batch(images)
        nb_similar = 0
        for image_after in images_augmented:
            if np.allclose(image_after, image_target):
                nb_similar += 1
        self.assertTrue(nb_similar > nb_augment*0.9 and nb_similar <= nb_augment*1.0)

    def test_vertical_flipping(self):
        """Tests vertical flipping of images (mirror on x-axis)."""

        image_before = [[255,   0,   0],
                        [  0, 255, 255],
                        [  0,   0, 255]]
        image_before = np.array(image_before, dtype=np.uint8)
        image_target = [[  0,   0,  1.0],
                        [  0, 1.0,  1.0],
                        [1.0,   0,    0]]
        image_target = np.array(image_target, dtype=np.float32)
        nb_augment = 1000
        images = np.resize([image_before], (nb_augment, 3, 3))

        # Test using just "False" for vflip (should be exactly 0%)
        augmenter = ImageAugmenter(3, 3, vflip=False)
        images_augmented = augmenter.augment_batch(images)
        nb_similar = 0
        for image_after in images_augmented:
            if np.allclose(image_after, image_target):
                nb_similar += 1
        self.assertEqual(nb_similar, 0)

        # Test using just "True" for vflip (should be ~50%)
        augmenter = ImageAugmenter(3, 3, vflip=True)
        images_augmented = augmenter.augment_batch(images)
        nb_similar = 0
        for image_after in images_augmented:
            if np.allclose(image_after, image_target):
                nb_similar += 1
        self.assertTrue(nb_similar > nb_augment*0.4 and nb_similar < nb_augment*0.6)

        # Test using a probability (float value) for vflip (vflip=0.9,
        # should be ~90%)
        augmenter = ImageAugmenter(3, 3, vflip=0.9)
        images_augmented = augmenter.augment_batch(images)
        nb_similar = 0
        for image_after in images_augmented:
            if np.allclose(image_after, image_target):
                nb_similar += 1
        self.assertTrue(nb_similar > nb_augment*0.8 and nb_similar <= nb_augment*1.0)

        # Test with multiple channels
        image_before = np.zeros((2, 3, 3), dtype=np.uint8)
        image_before[0] = [[255, 255,   0],
                           [255,   0,   0],
                           [  0,   0,   0]]
        image_before[1] = [[  0, 255,   0],
                           [  0, 255,   0],
                           [  0,   0, 255]]
        image_target = np.zeros((2, 3, 3), dtype=np.float32)
        image_target[0] = [[  0,   0,   0],
                           [1.0,   0,   0],
                           [1.0, 1.0,   0]]
        image_target[1] = [[  0,   0, 1.0],
                           [  0, 1.0,   0],
                           [  0, 1.0,   0]]
        images = np.resize([image_before], (nb_augment, 2, 3, 3))
        augmenter = ImageAugmenter(3, 3, vflip=1.0, channel_is_first_axis=True)
        images_augmented = augmenter.augment_batch(images)
        nb_similar = 0
        for image_after in images_augmented:
            if np.allclose(image_after, image_target):
                nb_similar += 1
        self.assertTrue(nb_similar > nb_augment*0.9 and nb_similar <= nb_augment*1.0)

if __name__ == '__main__':
    unittest.main()
