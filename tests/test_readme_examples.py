"""
Script to verify all examples in the readme.
Run from the project directory (i.e. parent) with
    python test_readme_examples.py
"""
from __future__ import print_function, division

#import sys
#import os
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy import misc

def main():
    example_standard_situation()
    example_heavy_augmentations()
    example_show()
    example_grayscale()
    example_determinism()
    example_keypoints()
    example_single_augmenters()
    example_unusual_distributions()
    example_hooks()

def example_standard_situation():
    print("Example: Standard Situation")
    # -------
    # dummy functions to make the example runnable here
    def load_batch(batch_idx):
        return np.random.randint(0, 255, (1, 16, 16, 3), dtype=np.uint8)

    def train_on_images(images):
        pass

    # -------

    from imgaug import augmenters as iaa

    seq = iaa.Sequential([
        iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
        iaa.Fliplr(0.5), # horizontally flip 50% of the images
        iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
    ])

    for batch_idx in range(1000):
        # 'images' should be either a 4D numpy array of shape (N, height, width, channels)
        # or a list of 3D numpy arrays, each having shape (height, width, channels).
        # Grayscale images must have shape (height, width, 1) each.
        # All images must have numpy's dtype uint8. Values are expected to be in
        # range 0-255.
        images = load_batch(batch_idx)
        images_aug = seq.augment_images(images)
        train_on_images(images_aug)


        # -----
        # Make sure that the example really does something
        if batch_idx == 0:
            assert not np.array_equal(images, images_aug)

def example_heavy_augmentations():
    print("Example: Heavy Augmentations")
    import imgaug as ia
    from imgaug import augmenters as iaa

    # random example images
    images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)

    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
    st = lambda aug: iaa.Sometimes(0.5, aug)

    # Define our sequence of augmentation steps that will be applied to every image
    # All augmenters with per_channel=0.5 will sample one value _per image_
    # in 50% of all cases. In all other cases they will sample new values
    # _per channel_.
    seq = iaa.Sequential([
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            iaa.Flipud(0.5), # vertically flip 50% of all images
            st(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
            st(iaa.GaussianBlur((0, 3.0))), # blur images with a sigma between 0 and 3.0
            st(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)), # add gaussian noise to images
            st(iaa.Dropout((0.0, 0.1), per_channel=0.5)), # randomly remove up to 10% of the pixels
            st(iaa.Add((-10, 10), per_channel=0.5)), # change brightness of images (by -10 to 10 of original value)
            st(iaa.Multiply((0.5, 1.5), per_channel=0.5)), # change brightness of images (50-150% of original value)
            st(iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)), # improve or worsen the contrast
            st(iaa.Grayscale((0.0, 1.0))), # blend with grayscale image
            st(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                translate_px={"x": (-16, 16), "y": (-16, 16)}, # translate by -16 to +16 pixels (per axis)
                rotate=(-45, 45), # rotate by -45 to +45 degrees
                shear=(-16, 16), # shear by -16 to +16 degrees
                order=[0, 1], # use scikit-image's interpolation orders 0 (nearest neighbour) and 1 (bilinear)
                cval=(0, 1.0), # if mode is constant, use a cval between 0 and 1.0
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            st(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)) # apply elastic transformations with random strengths
        ],
        random_order=True # do all of the above in random order
    )

    images_aug = seq.augment_images(images)

    # -----
    # Make sure that the example really does something
    assert not np.array_equal(images, images_aug)

def example_show():
    print("Example: Show")
    from imgaug import augmenters as iaa

    images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)
    seq = iaa.Sequential([iaa.Fliplr(0.5), iaa.GaussianBlur((0, 3.0))])

    # show an image with 8*8 augmented versions of image 0
    seq.show_grid(images[0], cols=8, rows=8)

    # Show an image with 8*8 augmented versions of image 0 and 8*8 augmented
    # versions of image 1. The identical augmentations will be applied to
    # image 0 and 1.
    seq.show_grid([images[0], images[1]], cols=8, rows=8)

def example_grayscale():
    print("Example: Grayscale")
    from imgaug import augmenters as iaa
    images = np.random.randint(0, 255, (16, 128, 128), dtype=np.uint8)
    seq = iaa.Sequential([iaa.Fliplr(0.5), iaa.GaussianBlur((0, 3.0))])
    # The library expects a list of images (3D inputs) or a single array (4D inputs).
    # So we add an axis to our grayscale array to convert it to shape (16, 128, 128, 1).
    images_aug = seq.augment_images(images[:, :, :, np.newaxis])

    # -----
    # Make sure that the example really does something
    assert not np.array_equal(images, images_aug)

def example_determinism():
    print("Example: Determinism")
    from imgaug import augmenters as iaa

    # Standard scenario: You have N RGB-images and additionally 21 heatmaps per image.
    # You want to augment each image and its heatmaps identically.
    images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)
    heatmaps = np.random.randint(0, 255, (16, 128, 128, 21), dtype=np.uint8)

    seq = iaa.Sequential([iaa.GaussianBlur((0, 3.0)), iaa.Affine(translate_px={"x": (-40, 40)})])

    # Convert the stochastic sequence of augmenters to a deterministic one.
    # The deterministic sequence will always apply the exactly same effects to the images.
    seq_det = seq.to_deterministic() # call this for each batch again, NOT only once at the start
    images_aug = seq_det.augment_images(images)
    heatmaps_aug = seq_det.augment_images(heatmaps)

    # -----
    # Make sure that the example really does something
    import imgaug as ia
    assert not np.array_equal(images, images_aug)
    assert not np.array_equal(heatmaps, heatmaps_aug)
    images_show = []
    for img_idx in range(len(images)):
        images_show.extend([images[img_idx], images_aug[img_idx], heatmaps[img_idx][..., 0:3], heatmaps_aug[img_idx][..., 0:3]])
    ia.show_grid(images_show, cols=4)

def example_keypoints():
    print("Example: Keypoints")
    import imgaug as ia
    from imgaug import augmenters as iaa
    from scipy import misc
    import random
    images = np.random.randint(0, 50, (4, 128, 128, 3), dtype=np.uint8)

    # Generate random keypoints.
    # The augmenters expect a list of imgaug.KeypointsOnImage.
    keypoints_on_images = []
    for image in images:
        height, width = image.shape[0:2]
        keypoints = []
        for _ in range(4):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            keypoints.append(ia.Keypoint(x=x, y=y))
        keypoints_on_images.append(ia.KeypointsOnImage(keypoints, shape=image.shape))

    seq = iaa.Sequential([iaa.GaussianBlur((0, 3.0)), iaa.Affine(scale=(0.5, 0.7))])
    seq_det = seq.to_deterministic() # call this for each batch again, NOT only once at the start

    # augment keypoints and images
    images_aug = seq_det.augment_images(images)
    keypoints_aug = seq_det.augment_keypoints(keypoints_on_images)

    # Example code to show each image and print the new keypoints coordinates
    for img_idx, (image_before, image_after, keypoints_before, keypoints_after) in enumerate(zip(images, images_aug, keypoints_on_images, keypoints_aug)):
        image_before = keypoints_before.draw_on_image(image_before)
        image_after = keypoints_after.draw_on_image(image_after)
        misc.imshow(np.concatenate((image_before, image_after), axis=1)) # before and after
        for kp_idx, keypoint in enumerate(keypoints_after.keypoints):
            keypoint_old = keypoints_on_images[img_idx].keypoints[kp_idx]
            x_old, y_old = keypoint_old.x, keypoint_old.y
            x_new, y_new = keypoint.x, keypoint.y
            print("[Keypoints for image #%d] before aug: x=%d y=%d | after aug: x=%d y=%d" % (img_idx, x_old, y_old, x_new, y_new))

def example_single_augmenters():
    print("Example: Single Augmenters")
    from imgaug import augmenters as iaa
    images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)

    flipper = iaa.Fliplr(1.0) # always horizontally flip each input image
    images[0] = flipper.augment_image(images[0]) # horizontally flip image 0

    vflipper = iaa.Flipud(0.9) # vertically flip each input image with 90% probability
    images[1] = vflipper.augment_image(images[1]) # probably vertically flip image 1

    blurer = iaa.GaussianBlur(3.0)
    images[2] = blurer.augment_image(images[2]) # blur image 2 by a sigma of 3.0
    images[3] = blurer.augment_image(images[3]) # blur image 3 by a sigma of 3.0 too

    translater = iaa.Affine(translate_px={"x": -16}) # move each input image by 16px to the left
    images[4] = translater.augment_image(images[4]) # move image 4 to the left

    scaler = iaa.Affine(scale={"y": (0.8, 1.2)}) # scale each input image to 80-120% on the y axis
    images[5] = scaler.augment_image(images[5]) # scale image 5 by 80-120% on the y axis

def example_unusual_distributions():
    print("Example: Unusual Distributions")
    from imgaug import augmenters as iaa
    from imgaug import parameters as iap
    images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)

    # Blur by a value sigma which is sampled from a uniform distribution
    # of range 0.1 <= x < 3.0.
    # The convenience shortcut for this is: iaa.GaussianBlur((0.1, 3.0))
    blurer = iaa.GaussianBlur(iap.Uniform(0.1, 3.0))
    images_aug = blurer.augment_images(images)

    # Blur by a value sigma which is sampled from a normal distribution N(1.0, 0.1),
    # i.e. sample a value that is usually around 1.0.
    # Clip the resulting value so that it never gets below 0.1 or above 3.0.
    blurer = iaa.GaussianBlur(iap.Clip(iap.Normal(1.0, 0.1), 0.1, 3.0))
    images_aug = blurer.augment_images(images)

    # Same again, but this time the mean of the normal distribution is not constant,
    # but comes itself from a uniform distribution between 0.5 and 1.5.
    blurer = iaa.GaussianBlur(iap.Clip(iap.Normal(iap.Uniform(0.5, 1.5), 0.1), 0.1, 3.0))
    images_aug = blurer.augment_images(images)

    # Use for sigma one of exactly three allowed values: 0.5, 1.0 or 1.5.
    blurer = iaa.GaussianBlur(iap.Choice([0.5, 1.0, 1.5]))
    images_aug = blurer.augment_images(images)

    # Sample sigma from a discrete uniform distribution of range 1 <= sigma <= 5,
    # i.e. sigma will have any of the following values: 1, 2, 3, 4, 5.
    blurer = iaa.GaussianBlur(iap.DiscreteUniform(1, 5))
    images_aug = blurer.augment_images(images)

def example_hooks():
    print("Example: Hooks")
    import imgaug as ia
    from imgaug import augmenters as iaa
    import numpy as np

    # images and heatmaps, just arrays filled with value 30
    images = np.ones((16, 128, 128, 3), dtype=np.uint8) * 30
    heatmaps = np.ones((16, 128, 128, 21), dtype=np.uint8) * 30

    # add vertical lines to see the effect of flip
    images[:, 16:128-16, 120:124, :] = 120
    heatmaps[:, 16:128-16, 120:124, :] = 120

    seq = iaa.Sequential([
      iaa.Fliplr(0.5, name="Flipper"),
      iaa.GaussianBlur((0, 3.0), name="GaussianBlur"),
      iaa.Dropout(0.02, name="Dropout"),
      iaa.AdditiveGaussianNoise(scale=0.01*255, name="MyLittleNoise"),
      iaa.AdditiveGaussianNoise(loc=32, scale=0.0001*255, name="SomeOtherNoise"),
      iaa.Affine(translate_px={"x": (-40, 40)}, name="Affine")
    ])

    # change the activated augmenters for heatmaps
    def activator_heatmaps(images, augmenter, parents, default):
        if augmenter.name in ["GaussianBlur", "Dropout", "MyLittleNoise"]:
            return False
        else:
            # default value for all other augmenters
            return default
    hooks_heatmaps = ia.HooksImages(activator=activator_heatmaps)

    seq_det = seq.to_deterministic() # call this for each batch again, NOT only once at the start
    images_aug = seq_det.augment_images(images)
    heatmaps_aug = seq_det.augment_images(heatmaps, hooks=hooks_heatmaps)

    # -----------
    ia.show_grid(images_aug)
    ia.show_grid(heatmaps_aug[..., 0:3])

if __name__ == "__main__":
    main()
