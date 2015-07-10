# -*- coding: utf-8 -*-
from __future__ import division
from skimage import transform
import numpy as np
import random
import warnings

def is_minmax_tuple(t, values_float=True):
    return type(t) is tuple and len(t) == 2

def create_aug_matrices(nb_matrices, img_width_px, img_height_px,
                        scale_to_percent=1.0, scale_axis_equally=False,
                        rotation_deg=0, shear_deg=0,
                        translation_x_px=0, translation_y_px=0,
                        seed=None):
    assert nb_matrices > 0
    assert img_width_px > 0
    assert img_height_px > 0
    assert is_minmax_tuple(scale_to_percent) or scale_to_percent >= 1.0
    assert is_minmax_tuple(rotation_deg) or rotation_deg >= 0
    assert is_minmax_tuple(shear_deg) or shear_deg >= 0
    assert is_minmax_tuple(translation_x_px) or translation_x_px >= 0
    assert is_minmax_tuple(translation_y_px) or translation_y_px >= 0

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    result = []
    
    shift_x = int(img_width_px / 2.0)
    shift_y = int(img_height_px / 2.0)
    
    # prepare min and max values for
    # scaling/zooming (min/max values)
    if is_minmax_tuple(scale_to_percent):
        scale_x_min = scale_to_percent[0]
        scale_x_max = scale_to_percent[1]
    else:
        scale_x_min = scale_to_percent
        scale_x_max = 1.0 - (scale_to_percent - 1.0)
    assert scale_x_min > 0.0
    if scale_x_max >= 2.0:
         warnings.warn("Scaling by more than 100 percent (%.2f)." % (scale_x_max,))
    scale_y_min = scale_x_min # scale_axis_equally affects the random value generation
    scale_y_max = scale_x_max
    
    # rotation (min/max values)
    if is_minmax_tuple(rotation_deg):
        rotation_deg_min = rotation_deg[0]
        rotation_deg_max = rotation_deg[1]
    else:
        rotation_deg_min = (-1) * int(rotation_deg)
        rotation_deg_max = int(rotation_deg)
    
    # shear (min/max values)
    if is_minmax_tuple(shear_deg):
        shear_deg_min = shear_deg[0]
        shear_deg_max = shear_deg[1]
    else:
        shear_deg_min = (-1) * int(shear_deg)
        shear_deg_max = int(shear_deg)
    
    # translation x-axis (min/max values)
    if is_minmax_tuple(translation_x_px):
        translation_x_px_min = translation_x_px[0]
        translation_x_px_max = translation_x_px[1]
    else:
        translation_x_px_min = (-1) * translation_x_px
        translation_x_px_max = translation_x_px
    
    # translation y-axis (min/max values)
    if is_minmax_tuple(translation_y_px):
        translation_y_px_min = translation_y_px[0]
        translation_y_px_max = translation_y_px[1]
    else:
        translation_y_px_min = (-1) * translation_y_px
        translation_y_px_max = translation_y_px
    
    # create nb_matrices randomized affine transformation matrices
    for _ in range(nb_matrices):
        # generate random values for scaling, rotation, shear, translation
        scale_x = random.uniform(scale_x_min, scale_x_max)
        scale_y = random.uniform(scale_y_min, scale_y_max)
        if not scale_axis_equally:
            scale_y = random.uniform(scale_y_min, scale_y_max)
        else:
            scale_y = scale_x
        rotation = np.deg2rad(random.randint(rotation_deg_min, rotation_deg_max))
        shear = np.deg2rad(random.randint(shear_deg_min, shear_deg_max))
        translation_x = random.randint(translation_x_px_min, translation_x_px_max)
        translation_y = random.randint(translation_y_px_min, translation_y_px_max)

        # create three affine transformation matrices
        # 1st one moves the image to top left, 2nd one transforms it, 3rd one
        # moves it back to the center.
        # The movement is neccessary, because rotation is applied to the top left
        # and not to the image's center.
        matrix_to_topleft = transform.SimilarityTransform(translation=[-shift_x, -shift_y])
        matrix_transforms = transform.AffineTransform(scale=(scale_x, scale_y), rotation=rotation, shear=shear, translation=(translation_x, translation_y))
        matrix_to_center = transform.SimilarityTransform(translation=[shift_x, shift_y])
        
        # Combine the three matrices to one affine transformation (one matrix)
        matrix = matrix_to_topleft + matrix_transforms + matrix_to_center
        
        result.append(matrix.inverse)

    return result

def apply_aug_matrices(images, matrices, transform_channels_equally=True, channel_is_first_axis=True, random_order=True):
    # images must be numpy array
    assert type(images).__module__ == np.__name__, "Expected numpy array for parameter 'images'."
    
    # images must have uint8 as dtype (0-255)
    assert images.dtype.name == "uint8", "Expected numpy.uint8 as image dtype."
    
    # 3 axis total (2 per image) for grayscale,
    # 4 axis total (3 per image) for RGB (usually)
    assert len(images.shape) in [3, 4], """Expected 'images' parameter to have either
        shape (image index, x, y) for greyscale
        or (image index, channel, x, y) / (image index, x, y, channel)
        for multi-channel (usually color) images."""
    
    nb_images = images.shape[0]
    
    # estimate number of channels, set to 1 if there is no axis channel,
    # otherwise it will usually be 3
    has_channels = False
    nb_channels = 1
    if len(images.shape) == 4:
        has_channels = True
        if channel_is_first_axis:
            nb_channels = images.shape[1] # first axis within each image
        else:
            nb_channels = images.shape[3] # last axis within each image
    
    # 0 to nb_images, but restart at 0 if index is beyond number of matrices
    len_indices = nb_images if transform_channels_equally else nb_images * nb_channels
    if random_order:
        order_indices = np.random.random_integers(0, len(matrices) - 1, len_indices)
    else:
        order_indices = np.arange(0, len_indices) % len(matrices)
    
    result = np.zeros(images.shape, dtype=np.float32)
    matrix_number = 0
    
    for img_idx, image in enumerate(images):
        if not has_channels:
            matrix = matrices[order_indices[matrix_number]]
            result[img_idx, ...] = transform.warp(image, matrix)
            matrix_number += 1
        else:
            for channel_idx in range(nb_channels):
                matrix = matrices[order_indices[matrix_number]]
                if channel_is_first_axis:
                    result[img_idx, channel_idx, ...] = transform.warp(image[channel_idx], matrix)
                else:
                    result[img_idx, ..., channel_idx] = transform.warp(image[..., channel_idx], matrix)
                
                if not transform_channels_equally:
                    matrix_number += 1
            if transform_channels_equally:
                matrix_number += 1

    return result

class BareboneImageAugmenter(object):
    def __init__(self):
        pass

    def augment_batch(self, images,
                        channel_is_first_axis=True,
                        scale_to_percent=1.0, scale_axis_equally=False,
                        rotation_deg=0, shear_deg=0,
                        translation_x_px=0, translation_y_px=0,
                        transform_channels_equally=True,
                        seed=None):
        if len(images.shape) == 3:
            # [image index, x, y]
            img_width_px = images.shape[1]
            img_height_px = images.shape[2]
        elif len(images.shape) == 4:
            if channel_is_first_axis:
                # [image index, channel, x, y]
                img_width_px = images.shape[2]
                img_height_px = images.shape[3]
            else:
                # [image index, x, y, channel]
                img_width_px = images.shape[1]
                img_height_px = images.shape[2]
        else:
            raise Exception("""Expected 'images' parameter to have 3
                dimensions (i.e. [image index, x, y])
                or 4 dimensions (either [image index, channel, x, y] or
                [image index, x, y, channel].""")
        
        # generate transformation matrices
        matrices = create_aug_matrices(images.shape[0],
                    img_width_px, img_height_px,
                    scale_to_percent=scale_to_percent,
                    scale_axis_equally=scale_axis_equally,
                    rotation_deg=rotation_deg,
                    shear_deg=shear_deg,
                    translation_x_px=translation_x_px,
                    translation_y_px=translation_y_px,
                    channel_is_first_axis=channel_is_first_axis,
                    seed=seed)
        
        # apply transformation matrices (i.e. augment images)
        return apply_aug_matrices(images, matrices,
                    transform_channels_equally=transform_channels_equally,
                    channel_is_first_axis=channel_is_first_axis)

class ImageAugmenter(object):
    def __init__(self, img_width_px, img_height_px,
                 channel_is_first_axis=True,
                 scale_to_percent=1.0, scale_axis_equally=False,
                 rotation_deg=0, shear_deg=0,
                 translation_x_px=0, translation_y_px=0,
                 transform_channels_equally=True):
        self.img_width_px = img_width_px
        self.img_height_px = img_height_px
        self.channel_is_first_axis = channel_is_first_axis
        self.scale_to_percent = scale_to_percent
        self.scale_axis_equally = scale_axis_equally
        self.rotation_deg = rotation_deg
        self.shear_deg = shear_deg
        self.translation_x_px = translation_x_px
        self.translation_y_px = translation_y_px
        self.transform_channels_equally = transform_channels_equally
    
    def augment_batch(self, images, seed=None):
        # generate transformation matrices
        matrices = create_aug_matrices(images.shape[0],
                    self.img_width_px,
                    self.img_height_px,
                    scale_to_percent=self.scale_to_percent,
                    scale_axis_equally=self.scale_axis_equally,
                    rotation_deg=self.rotation_deg,
                    shear_deg=self.shear_deg,
                    translation_x_px=self.translation_x_px,
                    translation_y_px=self.translation_y_px,
                    seed=seed)

        # apply transformation matrices (i.e. augment images)
        return apply_aug_matrices(images, matrices,
                    transform_channels_equally=self.transform_channels_equally,
                    channel_is_first_axis=self.channel_is_first_axis)
