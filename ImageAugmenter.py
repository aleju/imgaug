# -*- coding: utf-8 -*-
"""Wrapper functions and classes around scikit-images AffineTransformation.
Simplifies augmentation of images in machine learning.

Example usage:
        img_width = 32 # width of the images
        img_height = 32 # height of the images
        images = ... # e.g. load via scipy.misc.imload(filename)

        # For each image: randomly flip it horizontally (50% chance),
        # randomly rotate it between -20 and +20 degrees, randomly translate
        # it on the x-axis between -5 and +5 pixel.
        ia = ImageAugmenter(img_width, img_height, hlip=True, rotation_deg=20,
                            translation_x_px=5)
        augmented_images = ia.augment_batch(images)
"""
from __future__ import division
from skimage import transform as tf
import numpy as np
import random
#import warnings

def is_minmax_tuple(param):
    """Returns whether the parameter is a tuple containing two values.

    Used in create_aug_matrices() and probably useless everywhere else.

    Args:
        param: The parameter to check (whether it is a tuple of length 2).

    Returns:
        Boolean
    """
    return type(param) is tuple and len(param) == 2

def create_aug_matrices(nb_matrices, img_width_px, img_height_px,
                        scale_to_percent=1.0, scale_axis_equally=False,
                        rotation_deg=0, shear_deg=0,
                        translation_x_px=0, translation_y_px=0,
                        seed=None):
    """Creates the augmentation matrices that may later be used to transform
    images.

    This is a wrapper around scikit-image's transform.AffineTransform class.
    You can apply those matrices to images using the apply_aug_matrices()
    function.

    Args:
        nb_matrices: How many matrices to return, e.g. 100 returns 100 different
            random-generated matrices (= 100 different transformations).
        img_width_px: Width of the images that will be transformed later
            on (same as the width of each of the matrices).
        img_height_px: Height of the images that will be transformed later
            on (same as the height of each of the matrices).
        scale_to_percent: Same as in ImageAugmenter.__init__().
            Up to which percentage the images may be
            scaled/zoomed. The negative scaling is automatically derived
            from this value. A value of 1.1 allows scaling by any value
            between -10% and +10%. You may set min and max values yourself
            by using a tuple instead, like (1.1, 1.2) to scale between
            +10% and +20%. Default is 1.0 (no scaling).
        scale_axis_equally: Same as in ImageAugmenter.__init__().
            Whether to always scale both axis (x and y)
            in the same way. If set to False, then e.g. the Augmenter
            might scale the x-axis by 20% and the y-axis by -5%.
            Default is False.
        rotation_deg: Same as in ImageAugmenter.__init__().
            By how much the image may be rotated around its
            center (in degrees). The negative rotation will automatically
            be derived from this value. E.g. a value of 20 allows any
            rotation between -20 degrees and +20 degrees. You may set min
            and max values yourself by using a tuple instead, e.g. (5, 20)
            to rotate between +5 und +20 degrees. Default is 0 (no
            rotation).
        shear_deg: Same as in ImageAugmenter.__init__().
            By how much the image may be sheared (in degrees). The
            negative value will automatically be derived from this value.
            E.g. a value of 20 allows any shear between -20 degrees and
            +20 degrees. You may set min and max values yourself by using a
            tuple instead, e.g. (5, 20) to shear between +5 und +20
            degrees. Default is 0 (no shear).
        translation_x_px: Same as in ImageAugmenter.__init__().
            By up to how many pixels the image may be
            translated (moved) on the x-axis. The negative value will
            automatically be derived from this value. E.g. a value of +7
            allows any translation between -7 and +7 pixels on the x-axis.
            You may set min and max values yourself by using a tuple
            instead, e.g. (5, 20) to translate between +5 und +20 pixels.
            Default is 0 (no translation on the x-axis).
        translation_y_px: Same as in ImageAugmenter.__init__().
            See translation_x_px, just for the y-axis.
        seed: Seed to use for python's and numpy's random functions.

    Returns:
        List of augmentation matrices.
    """
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
    #if scale_x_max >= 2.0:
    #     warnings.warn("Scaling by more than 100 percent (%.2f)." % (scale_x_max,))
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
        # 1st one moves the image to the top left, 2nd one transforms it, 3rd one
        # moves it back to the center.
        # The movement is neccessary, because rotation is applied to the top left
        # and not to the image's center (same for scaling and shear).
        matrix_to_topleft = tf.SimilarityTransform(translation=[-shift_x, -shift_y])
        matrix_transforms = tf.AffineTransform(scale=(scale_x, scale_y),
                                               rotation=rotation, shear=shear,
                                               translation=(translation_x,
                                                            translation_y))
        matrix_to_center = tf.SimilarityTransform(translation=[shift_x, shift_y])

        # Combine the three matrices to one affine transformation (one matrix)
        matrix = matrix_to_topleft + matrix_transforms + matrix_to_center

        # one matrix is ready, add it to the result
        result.append(matrix.inverse)

    return result

def apply_aug_matrices(images, matrices, transform_channels_equally=True,
                       channel_is_first_axis=False, random_order=True,
                       mode="constant", cval=0.0, interpolation_order=1):
    """Augment the given images using the given augmentation matrices.

    This function is a wrapper around scikit-image's transform.warp().
    It is expected to be called by ImageAugmenter.augment_batch().
    The matrices may be generated by create_aug_matrices().

    Args:
        images: Same as in ImageAugmenter.augment_batch().
            Numpy array (dtype: uint8, i.e. values 0-255) with the images.
            Expected shape is either (image-index, height, width) for
            grayscale images or (image-index, channel, height, width) for
            images with channels (e.g. RGB) where the channel has the first
            index or (image-index, height, width, channel) for images with
            channels, where the channel is the last index.
            If your shape is (image-index, channel, width, height) then
            you must also set channel_is_first_axis=True in the constructor.
        matrices: A list of augmentation matrices as produced by
            create_aug_matrices().
        transform_channels_equally: Same as in ImageAugmenter.__init__().
            Whether to apply the exactly same
            transformations to each channel of an image (True). Setting
            it to False allows different transformations per channel,
            e.g. the red-channel might be rotated by +20 degrees, while
            the blue channel (of the same image) might be rotated
            by -5 degrees. If you don't have any channels (2D grayscale),
            you can simply ignore this setting.
            Default is True (transform all equally).
        channel_is_first_axis: Same as in ImageAugmenter.__init__().
            Whether the channel (e.g. RGB) is the first
            axis of each image (True) or the last axis (False).
            False matches the scipy and PIL implementation and is the
            default. If your images are 2D-grayscale then you can ignore
            this setting (as the augmenter will ignore it too).
        random_order: Whether to apply the augmentation matrices in a random
            order (True, e.g. the 2nd matrix might be applied to the
            5th image) or in the given order (False, e.g. the 2nd matrix might
            be applied to the 2nd image).
            Notice that for multi-channel images (e.g. RGB) this function
            will use a different matrix for each channel, unless
            transform_channels_equally is set to True.
        mode: Parameter used for the transform.warp-function of scikit-image.
            Can usually be ignored.
        cval: Parameter used for the transform.warp-function of scikit-image.
            Defines the fill color for "new" pixels, e.g. for empty areas
            after rotations. (0.0 is black, 1.0 is white.)
        interpolation_order: Parameter used for the transform.warp-function of
            scikit-image. Defines the order of all interpolations used to
            generate the new/augmented image. See their documentation for
            further details.
    """
    # images must be numpy array
    assert type(images).__module__ == np.__name__, "Expected numpy array for " \
                                                   "parameter 'images'."

    # images must have uint8 as dtype (0-255)
    assert images.dtype.name == "uint8", "Expected numpy.uint8 as image dtype."

    # 3 axis total (2 per image) for grayscale,
    # 4 axis total (3 per image) for RGB (usually)
    assert len(images.shape) in [3, 4], """Expected 'images' parameter to have
        either shape (image index, y, x) for greyscale
        or (image index, channel, y, x) / (image index, y, x, channel)
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


    # We generate here the order in which the matrices may be applied.
    # At the end, order_indices will contain the index of the matrix to use
    # for each image, e.g. [15, 2] would mean, that the 15th matrix will be
    # applied to the 0th image, the 2nd matrix to the 1st image.
    # If the images gave multiple channels (e.g. RGB) and
    # transform_channels_equally has been set to False, we will need one
    # matrix per channel instead of per image.

    # 0 to nb_images, but restart at 0 if index is beyond number of matrices
    len_indices = nb_images if transform_channels_equally else nb_images * nb_channels
    if random_order:
        # Notice: This way to choose random matrices is concise, but can create
        # problems if there is a low amount of images and matrices.
        # E.g. suppose that 2 images are ought to be transformed by either
        # 0px translation on the x-axis or 1px translation. So 50% of all
        # matrices translate by 0px and 50% by 1px. The following method
        # will randomly choose a combination of the two matrices for the
        # two images (matrix 0 for image 0 and matrix 0 for image 1,
        # matrix 0 for image 0 and matrix 1 for image 1, ...).
        # In 50% of these cases, a different matrix will be chosen for image 0
        # and image 1 (matrices 0, 1 or matrices 1, 0). But 50% of these
        # "different" matrices (different index) will be the same, as 50%
        # translate by 1px and 50% by 0px. As a result, 75% of all augmentations
        # will transform both images in the same way.
        # The effect decreases if more matrices or images are chosen.
        order_indices = np.random.random_integers(0, len(matrices) - 1, len_indices)
    else:
        # monotonously growing indexes (each by +1), but none of them may be
        # higher than or equal to the number of matrices
        order_indices = np.arange(0, len_indices) % len(matrices)

    result = np.zeros(images.shape, dtype=np.float32)
    matrix_number = 0

    # iterate over every image, find out which matrix to apply and then use
    # that matrix to augment the image
    for img_idx, image in enumerate(images):
        if not has_channels:
            matrix = matrices[order_indices[matrix_number]]
            result[img_idx, ...] = tf.warp(image, matrix, mode=mode, cval=cval,
                                           order=interpolation_order)
            matrix_number += 1
        else:
            for channel_idx in range(nb_channels):
                matrix = matrices[order_indices[matrix_number]]
                if channel_is_first_axis:
                    warped = tf.warp(image[channel_idx], matrix, mode=mode,
                                     cval=cval, order=interpolation_order)
                    result[img_idx, channel_idx, ...] = warped
                else:
                    warped = tf.warp(image[..., channel_idx], matrix, mode=mode,
                                     cval=cval, order=interpolation_order)
                    result[img_idx, ..., channel_idx] = warped

                if not transform_channels_equally:
                    matrix_number += 1
            if transform_channels_equally:
                matrix_number += 1

    return result

class ImageAugmenter(object):
    """Helper class to randomly augment images, usually for neural networks.

    Example usage:
        img_width = 32 # width of the images
        img_height = 32 # height of the images
        images = ... # e.g. load via scipy.misc.imload(filename)

        # For each image: randomly flip it horizontally (50% chance),
        # randomly rotate it between -20 and +20 degrees, randomly translate
        # it on the x-axis between -5 and +5 pixel.
        ia = ImageAugmenter(img_width, img_height, hlip=True, rotation_deg=20,
                            translation_x_px=5)
        augmented_images = ia.augment_batch(images)
    """
    def __init__(self, img_width_px, img_height_px, channel_is_first_axis=False,
                 hflip=False, vflip=False,
                 scale_to_percent=1.0, scale_axis_equally=False,
                 rotation_deg=0, shear_deg=0,
                 translation_x_px=0, translation_y_px=0,
                 transform_channels_equally=True):
        """
        Args:
            img_width_px: The intended width of each image in pixels.
            img_height_px: The intended height of each image in pixels.
            channel_is_first_axis: Whether the channel (e.g. RGB) is the first
                axis of each image (True) or the last axis (False).
                False matches the scipy and PIL implementation and is the
                default. If your images are 2D-grayscale then you can ignore
                this setting (as the augmenter will ignore it too).
            hflip: Whether to randomly flip images horizontally (on the y-axis).
                You may choose either False (no horizontal flipping),
                True (flip with probability 0.5) or use a float
                value (probability) between 0.0 and 1.0. Default is False.
            vflip: Whether to randomly flip images vertically (on the x-axis).
                You may choose either False (no vertical flipping),
                True (flip with probability 0.5) or use a float
                value (probability) between 0.0 and 1.0. Default is False.
            scale_to_percent: Up to which percentage the images may be
                scaled/zoomed. The negative scaling is automatically derived
                from this value. A value of 1.1 allows scaling by any value
                between -10% and +10%. You may set min and max values yourself
                by using a tuple instead, like (1.1, 1.2) to scale between
                +10% and +20%. Default is 1.0 (no scaling).
            scale_axis_equally: Whether to always scale both axis (x and y)
                in the same way. If set to False, then e.g. the Augmenter
                might scale the x-axis by 20% and the y-axis by -5%.
                Default is False.
            rotation_deg: By how much the image may be rotated around its
                center (in degrees). The negative rotation will automatically
                be derived from this value. E.g. a value of 20 allows any
                rotation between -20 degrees and +20 degrees. You may set min
                and max values yourself by using a tuple instead, e.g. (5, 20)
                to rotate between +5 und +20 degrees. Default is 0 (no
                rotation).
            shear_deg: By how much the image may be sheared (in degrees). The
                negative value will automatically be derived from this value.
                E.g. a value of 20 allows any shear between -20 degrees and
                +20 degrees. You may set min and max values yourself by using a
                tuple instead, e.g. (5, 20) to shear between +5 und +20
                degrees. Default is 0 (no shear).
            translation_x_px: By up to how many pixels the image may be
                translated (moved) on the x-axis. The negative value will
                automatically be derived from this value. E.g. a value of +7
                allows any translation between -7 and +7 pixels on the x-axis.
                You may set min and max values yourself by using a tuple
                instead, e.g. (5, 20) to translate between +5 und +20 pixels.
                Default is 0 (no translation on the x-axis).
            translation_y_px: See translation_x_px, just for the y-axis.
            transform_channels_equally: Whether to apply the exactly same
                transformations to each channel of an image (True). Setting
                it to False allows different transformations per channel,
                e.g. the red-channel might be rotated by +20 degrees, while
                the blue channel (of the same image) might be rotated
                by -5 degrees. If you don't have any channels (2D grayscale),
                you can simply ignore this setting.
                Default is True (transform all equally).
        """
        self.img_width_px = img_width_px
        self.img_height_px = img_height_px
        self.channel_is_first_axis = channel_is_first_axis

        self.hflip_prob = 0.0
        if hflip == True:
            self.hflip_prob = 0.5
        elif hflip == False:
            self.hflip_prob = 0.0
        elif isinstance(hflip, float):
            assert hflip >= 0.0 and hflip <= 1.0
            self.hflip_prob = hflip
        else:
            raise Exception("Unexpected value for parameter 'hflip'.")

        self.vflip_prob = 0.0
        if vflip == True:
            self.vflip_prob = 0.5
        elif vflip == False:
            self.vflip_prob = 0.0
        elif isinstance(vflip, float):
            assert vflip >= 0.0 and vflip <= 1.0
            self.vflip_prob = vflip
        else:
            raise Exception("Unexpected value for parameter 'vflip'.")

        self.scale_to_percent = scale_to_percent
        self.scale_axis_equally = scale_axis_equally
        self.rotation_deg = rotation_deg
        self.shear_deg = shear_deg
        self.translation_x_px = translation_x_px
        self.translation_y_px = translation_y_px
        self.transform_channels_equally = transform_channels_equally
        self.cval = 0.0
        self.interpolation_order = 1

    def augment_batch(self, images, seed=None):
        """Augments a batch of images.

        Applies all settings (rotation, shear, translation, ...) that
        have been chosen in the constructor.

        Args:
            images: Numpy array (dtype: uint8, i.e. values 0-255) with the images.
                Expected shape is either (image-index, height, width) for
                grayscale images or (image-index, channel, height, width) for
                images with channels (e.g. RGB) where the channel has the first
                index or (image-index, height, width, channel) for images with
                channels, where the channel is the last index.
                If your shape is (image-index, channel, width, height) then
                you must also set channel_is_first_axis=True in the constructor.
            seed: A seed to be used when calling the function to generate
                the augmentation matrices. Default is None (dont use a seed).

        Returns:
            Augmented images as numpy array of dtype float32 (i.e. values
            are between 0.0 and 1.0).
        """
        shape = images.shape
        nb_channels = 0
        if len(shape) == 3:
            # shape like (image_index, y-axis, x-axis)
            assert shape[1] == self.img_height_px
            assert shape[2] == self.img_width_px
            nb_channels = 1
        elif len(shape) == 4:
            if not self.channel_is_first_axis:
                # shape like (image-index, y-axis, x-axis, channel-index)
                assert shape[1] == self.img_height_px
                assert shape[2] == self.img_width_px
                nb_channels = shape[3]
            else:
                # shape like (image-index, channel-index, y-axis, x-axis)
                assert shape[2] == self.img_height_px
                assert shape[3] == self.img_width_px
                nb_channels = shape[1]
        else:
            msg = "Mismatch between images shape %s and " \
                  "predefined image width/height (%d/%d)."
            raise Exception(msg % (str(shape), self.img_width_px, self.img_height_px))

        # --------------------------------
        # horizontal and vertical flipping/mirroring
        # --------------------------------
        # This should be done before applying the affine matrices, as otherwise
        # contents of image might already be rotated/translated out of the image.
        # It is done with numpy instead of the affine matrices, because
        # scikit-image doesn't offer a nice interface to add mirroring/flipping
        # to affine transformations. The numpy operations are O(1), so they
        # shouldn't have a noticeable effect on runtimes. They also won't suffer
        # from interpolation problems.
        if self.hflip_prob > 0 or self.vflip_prob > 0:
            # TODO this currently ignores the setting in
            # transform_channels_equally and will instead always flip all
            # channels equally
            images_flipped = []
            y_p = self.hflip_prob
            x_p = self.vflip_prob
            for image in images:
                if random.random() < y_p:
                    images_flipped.append(np.fliplr(image))
                else:
                    images_flipped.append(image)
                # flipping vertically is a rather rare choice, so first check if
                # x-flip probability is non-zero before generating a random value,
                # should usually be faster
                if x_p > 0 and random.random() < x_p:
                    images_flipped.append(np.flipud(image))
                else:
                    images_flipped.append(image)
            images = np.array(images_flipped, dtype=np.uint8)

        # --------------------------------
        # generate transformation matrices
        # --------------------------------
        # estimate the number of matrices required
        if self.transform_channels_equally:
            nb_matrices = shape[0]
        else:
            nb_matrices = shape[0] * nb_channels

        # generate matrices
        matrices = create_aug_matrices(nb_matrices,
                                       self.img_width_px,
                                       self.img_height_px,
                                       scale_to_percent=self.scale_to_percent,
                                       scale_axis_equally=self.scale_axis_equally,
                                       rotation_deg=self.rotation_deg,
                                       shear_deg=self.shear_deg,
                                       translation_x_px=self.translation_x_px,
                                       translation_y_px=self.translation_y_px,
                                       seed=seed)

        # --------------------------------
        # apply transformation matrices (i.e. augment images)
        # --------------------------------
        return apply_aug_matrices(images, matrices,
                                  transform_channels_equally=self.transform_channels_equally,
                                  channel_is_first_axis=self.channel_is_first_axis,
                                  cval=self.cval, interpolation_order=self.interpolation_order)
