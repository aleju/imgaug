import unittest
import numpy as np
from ImageAugmenter import ImageAugmenter
import random
from scipy import misc # to show images
from skimage import data

random.seed(123456789)

#def np_array_similar(arr1, arr2, **kwargs):
#    return np.allclose(arr1, arr2, kwargs)

class TestImageAugmenter(unittest.TestCase):
    def test_rotation_a(self):
        """Test rotation of -90 to 90 degrees on an image that should change
        upon rotation."""
        image_before = [[0, 255, 0],
                        [0, 255, 0],
                        [0, 255, 0]]
        image_target = [[0, 0, 0],
                        [1.0, 1.0, 1.0],
                        [0, 0, 0]]
        images = np.array([image_before]).astype(np.uint8)
        ia = ImageAugmenter(2, 2, rotation_deg=(90,90))
        
        image_after = ia.augment_batch(images)[0]
        self.assertTrue(np.allclose(image_target, image_after))
    
    def test_rotation_b(self):
        """Test rotation of -90 to 90 degrees on an rotation invariant image."""
        image_before = [[0, 0, 0],
                        [0, 255, 0],
                        [0, 0, 0]]
        image_target = [[0, 0, 0],
                        [0, 1.0, 0],
                        [0, 0, 0]]
        images = np.array([image_before]).astype(np.uint8)
        ia = ImageAugmenter(3, 3, rotation_deg=180)
        
        # all must be similar to target
        nb_similar = 0
        for _ in range(100):
            image_after = ia.augment_batch(images)[0]
            #print(image_after)
            if np.allclose(image_target, image_after, atol=0.1):
                nb_similar += 1
        self.assertEquals(nb_similar, 100)
    
    def test_scaling(self):
        """Test zooming/scaling on an image that should change upon zooming."""
        #image_before = [[0, 0, 0], [0, 255, 0], [0, 0, 0]]
        #image_target = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        
        """
        img = data.camera()
        ia = ImageAugmenter(img.shape[0], img.shape[1], scale_to_percent=(4.0, 4.0), scale_axis_equally=True)
        img_after = ia.augment_batch(np.array([img]))[0]
        misc.imshow(img)
        misc.imshow(img_after)
        """
        
        size_x = 4
        size_y = 4
        image_before = np.zeros((size_x,size_y))
        image_before[1:size_x-1, 1:size_y-1] = 255
        image_target = np.zeros((size_x,size_y))
        image_target[0:size_x, 0:size_y] = 1.0
        
        """image_before = [[0,   0,   0,   0, 0],
                        [0, 255, 255, 255, 0],
                        [0, 255, 255, 255, 0],
                        [0, 255, 255, 255, 0],
                        [0,   0,   0,   0, 0]]"""
        """
        image_target = [[1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0]]
        """
        
        images = np.array([image_before]).astype(np.uint8)
        ia = ImageAugmenter(size_x, size_y, scale_to_percent=(1.99, 1.99), scale_axis_equally=True)
        
        image_after = ia.augment_batch(images)[0]
        misc.imshow(image_after)
        self.assertTrue(np.sum(image_after) > np.sum(image_before)/255)
        
        # at least one should be similar to target
        """
        one_similar = False
        for _ in range(1000):
            image_after = ia.augment_batch(images)[0]
            misc.imshow(image_after)
            if np.allclose(image_target, image_after):
                one_similar = True
                break
        self.assertTrue(one_similar)
        """
    
    def test_shear(self):
        """Very rough test of shear: It simply measures whether an image tends
        to be significantly different after shear (any change)."""
        
        image_before = [[0, 255, 0], [0, 255, 0], [0, 255, 0]]
        image_target = [[0, 1.0, 0], [0, 1.0, 0], [0, 1.0, 0]]
        images = np.array([image_before]).astype(np.uint8)
        ia = ImageAugmenter(3, 3, shear_deg=20)
        
        # at least one should be different from target
        one_different = False
        for _ in range(1000):
            image_after = ia.augment_batch(images)[0]
            if not np.allclose(image_target, image_after):
                one_different = True
                break
        self.assertTrue(one_different)
    
    def test_translation_x(self):
        """Testing translation on the x-axis."""
        
        image_before = [[255, 0], [255, 0]]
        image_target = [[0, 1.0], [0, 1.0]]
        images = np.array([image_before]).astype(np.uint8)
        ia = ImageAugmenter(2, 2, translation_x_px=1)
        
        # at least one should be similar
        one_similar = False
        for _ in range(100):
            image_after = ia.augment_batch(images)[0]
            if np.allclose(image_target, image_after):
                one_similar = True
                break
        self.assertTrue(one_similar)
    
    def test_translation_y(self):
        """Testing translation on the y-axis."""
        image_before = [[0, 0], [255, 255]]
        image_target = [[1.0, 1.0], [0, 0]]
        images = np.array([image_before]).astype(np.uint8)
        ia = ImageAugmenter(2, 2, translation_y_px=1)
        
        # at least one should be similar
        one_similar = False
        for _ in range(100):
            image_after = ia.augment_batch(images)[0]
            if np.allclose(image_target, image_after):
                one_similar = True
                break
        self.assertTrue(one_similar)
    
    def test_single_channel(self):
        """Tests images with channels (e.g. RGB channels)."""
        # One single channel
        image_before = [[[255, 0], [255, 0]]]
        image_target = [[[0, 1.0], [0, 1.0]]]
        images = np.array([image_before]).astype(np.uint8)
        ia = ImageAugmenter(2, 2, translation_x_px=1)
        
        # at least one should be similar
        one_similar = False
        for _ in range(100):
            image_after = ia.augment_batch(images)[0]
            if np.allclose(image_target, image_after):
                one_similar = True
                break
        self.assertTrue(one_similar)
    
    def test_two_channels(self):
        """Tests augmentation of images with two channels (either first or last
        axis of each image). Tested using x-translation."""
        # two channels, channel is first axis of each image
        ia = ImageAugmenter(2, 2, translation_x_px=1)
        
        image_before = np.zeros((2, 2, 2)).astype(np.uint8)
        image_before[0][0][0] = 255
        image_before[0][0][1] = 255
        image_before[0][1][0] = 0
        image_before[0][1][1] = 0
        
        image_before[1][0][0] = 0
        image_before[1][0][1] = 255
        image_before[1][1][0] = 0
        image_before[1][1][1] = 0
        #            ^        channel
        #               ^     x
        #                  ^  y
        
        image_target = np.zeros((2, 2, 2)).astype(np.float32)
        image_target[0][0][0] = 0
        image_target[0][0][1] = 0
        image_target[0][1][0] = 1.0
        image_target[0][1][1] = 1.0
        
        image_target[1][0][0] = 0
        image_target[1][0][1] = 0
        image_target[1][1][0] = 0
        image_target[1][1][1] = 1.0
        
        images = np.array([image_before]).astype(np.uint8)
        nb_similar = 0
        for _ in range(100):
            image_after = ia.augment_batch(images)[0]
            if np.allclose(image_target, image_after):
                nb_similar += 1
        self.assertTrue(nb_similar > (33-10) and nb_similar < (33+10))
        
        
        # two channels, channel is last axis of each image
        ia = ImageAugmenter(2, 2, translation_x_px=1,
                            channel_is_first_axis=False)
        
        image_before = np.zeros((2, 2, 2)).astype(np.uint8)
        image_before[0][0][0] = 255
        image_before[0][1][0] = 255
        image_before[1][0][0] = 0
        image_before[1][1][0] = 0
        
        image_before[0][0][1] = 0
        image_before[0][1][1] = 255
        image_before[1][0][1] = 0
        image_before[1][1][1] = 0
        #            ^        x
        #               ^     y
        #                  ^  channel
        
        image_target = np.zeros((2, 2, 2)).astype(np.float32)
        image_target[0][0][0] = 0
        image_target[0][1][0] = 0
        image_target[1][0][0] = 1.0
        image_target[1][1][0] = 1.0
        
        image_target[0][0][1] = 0
        image_target[0][1][1] = 0
        image_target[1][0][1] = 0
        image_target[1][1][1] = 1.0
        
        images = np.array([image_before]).astype(np.uint8)
        nb_similar = 0
        for _ in range(100):
            image_after = ia.augment_batch(images)[0]
            if np.allclose(image_target, image_after):
                nb_similar += 1
        self.assertTrue(nb_similar > (33-10) and nb_similar < (33+10))

    def test_transform_channels_unequally(self):
        # two channels, channel is first axis of each image
        ia = ImageAugmenter(2, 2, translation_x_px=1,
                            transform_channels_equally=False)
        
        image_before = np.zeros((2, 2, 2)).astype(np.uint8)
        image_before[0][0][0] = 255
        image_before[0][0][1] = 255
        image_before[0][1][0] = 0
        image_before[0][1][1] = 0
        
        image_before[1][0][0] = 0
        image_before[1][0][1] = 255
        image_before[1][1][0] = 0
        image_before[1][1][1] = 0
        #            ^        channel
        #               ^     x
        #                  ^  y
        
        image_target = np.zeros((2, 2, 2)).astype(np.float32)
        image_target[0][0][0] = 0
        image_target[0][0][1] = 0
        image_target[0][1][0] = 1.0
        image_target[0][1][1] = 1.0
        
        image_target[1][0][0] = 0
        image_target[1][0][1] = 0
        image_target[1][1][0] = 0
        image_target[1][1][1] = 1.0
        
        images = np.array([image_before]).astype(np.uint8)
        nb_similar_channel_0 = 0
        nb_similar_channel_1 = 0
        nb_equally_transformed = 0
        nb_unequally_transformed = 0
        
        # augment 100 times and count how often the channels were transformed
        # in equal or unequal ways. Assumption: as equal transformation was
        # deactivated, the channels should be transformed in similar ways in
        # roughly 1/3 of all cases (as translation-x of 1 means a choice between
        # -1, 0 or +1)
        for _ in range(100):
            image_after = ia.augment_batch(images)[0]
            similar_channel_0 = np.allclose(image_target[0], image_after[0])
            similar_channel_1 = np.allclose(image_target[1], image_after[1])
            if similar_channel_0:
                nb_similar_channel_0 += 1
            if similar_channel_1:
                nb_similar_channel_1 += 1
            if similar_channel_0 == similar_channel_1:
                nb_equally_transformed += 1
            else:
                nb_unequally_transformed += 1
        self.assertTrue(nb_similar_channel_0 > (33-10) and nb_similar_channel_0 < (33+10))
        self.assertTrue(nb_similar_channel_1 > (33-10) and nb_similar_channel_1 < (33+10))
        self.assertTrue(nb_equally_transformed > (33-10) and nb_equally_transformed < (33+10))

if __name__ == '__main__':
    unittest.main()
