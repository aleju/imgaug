# ImageAugmenter

The ImageAugmenter is a class to augment your training images for neural networks (and other machine learning methods).
Possible augmentations are:
* Translation (moving the image)
* Scaling (zooming)
* Rotation
* Shear
* Horizontal flipping/mirroring
* Vertical flipping/mirroring

The class is build as a wrapper around scikit-image's `AffineTransform`.
Most augmentations (all except flipping) are combined into one affine transformation, making the augmentation process reasonably fast.

# Example

Lena, augmented 50 times with all augmentation options activated:

![50 augmented versions of Lena](lena_augmented.jpg?raw=true "50 augmented versions of Lena")

# Requirements

* numpy
* scikit-image
* scipy (optional, required for some of the tests)
* matplotlib (optional, required if you want to see a plot containing various example images showing the effects of your chosen augmentation settings)

# Usage

There is no pip-installer or setup.py for this class. Simply copy `ImageAugmenter.py` to your project.
Then import it, create a new `ImageAugmenter` object and use `ImageAugmenter.augment_batch()` to augment an array of images.
The function expects a numpy array (of dtype **numpy.uint8** with values between 0 and 255) of images.
The expected shape of that array is any of the following:
* `(image-index, y, x)`, automatically detected
* `(image-index, y, x, image channel)`, automatically detected
* `(image-index, channel, y, x)`, set `channel_is_first_axis=True` in the constructor if that is the case for you.
The return type is a numpy array of dtype **numpy.float32** with values between 0.0 and 1.0.

# Examples

Load an image and apply augmentations to it:

```python
from ImageAugmenter import ImageAugmenter
from scipy import misc
import numpy as np

image = misc.imread("example.png")
height = image.shape[0]
width = image.shape[1]
augmenter = ImageAugmenter(width, height, # width and height of the image (must be the same for all images in the batch)
                           hflip=True,    # flip horizontally with 50% probability
                           vflip=True,    # flip vertically with 50% probability
                           scale_to_percent=1.3, # scale the image to 70%-130% of its original size
                           scale_axis_equally=False, # allow the axis to be scaled unequally (e.g. x more than y)
                           rotation_deg=25,    # rotate between -25 and +25 degrees
                           shear_deg=10,       # shear between -10 and +10 degrees
                           translation_x_px=5, # translate between -5 and +5 px on the x-axis
                           translation_y_px=5  # translate between -5 and +5 px on the y-axis
                           )

# augment a batch containing only this image
# the input array must have dtype uint8 (ie. values 0-255), as is the case for scipy's imread()
# the output array will have dtype float32 (0.0-1.0) and can be fed directly into a neural network
augmented_images = augmenter.augment_batch(np.array([image], dtype=np.uint8))
```

You can set minimum and maximum values of the augmentation parameters by using tuples:

```python
image = misc.imread("example.png")
width = image.shape[1]
height = image.shape[0]
augmenter = ImageAugmenter(width, height,
                           scale_to_percent=(0.9, 1.2), # scale the image to 90%-120% of its original size
                           rotation_deg=(10, 25),       # rotate between +10 and +25 degrees
                           shear_deg=(-10, -5),         # shear between -10 and -5 degrees
                           translation_x_px=(0, 5),     # translate between 0 and +5 px on the x-axis
                           translation_y_px=(5, 10)     # translate between +5 and +10 px on the y-axis
                           )

augmented_images = augmenter.augment_batch(np.array([image], dtype=np.uint8))
```

Example with a synthetic image (grayscale, so no channels):

```python
image = [[0, 255, 0],
         [0, 255, 0],
         [0, 255, 0]]
images = np.array([image]).astype(np.uint8)

# set width to 3, height to 3, rotate always exactly by 45 degrees
augmenter = ImageAugmenter(3, 3, rotation_deg=(45, 45))
augmented_images = augmenter.augment_batch(np.array([image], dtype=np.uint8))
```

You can pregenerate some augmentation matrices, which will later on be applied to the images (in random order).
This will speed up the augmentation process a little bit, because the transformation matrices can be reused multiple times.

```python
augmenter = ImageAugmenter(height, width, rotation_deg=10)
# pregenerate 10,000 matrices
augmenter.pregenerate_matrices(10000)
augmented_images = augmenter.augment_batch(np.array([image], dtype=np.uint8))
```

# Plotting your augmentations

For debugging purposes you can show/plot examples of your augmentation settings (i.e. what images look like if you apply these settings to them).
Use either the method `ImageAugmenter.plot_image(numpy.array image)` for that or alternatively `ImageAugmenter.plot_images(numpy.array images, boolean augment)`. (For `plot_images` set `augment` to `False` if the images are already augmented, otherwise to `True`.)
Example for `plot_image()`:

```python
image = misc.imread("example.png")
width = image.shape[1]
height = image.shape[0]
augmenter = ImageAugmenter(width, height, rotation_deg=20)
# This will show the image 50 times, each one with a random augmentation as defined in the constructor,
# i.e. each image being rotated by any random value between -20 and +20 degrees.
augmenter.plot_image(image, nb_repeat=50)
```

# Special use cases

By default the class expects your images to have one of the following two shapes:
* `(y, x)` for grayscale images.
* `(y, x, channel)` for images with multiple channels (e.g. RGB).

If your images have their channel in the first axis instead of the last, i.e. `(channel, y, x)`, then you can use the parameter `channel_is_first_axis` in the `ImageAugmenter`'s `__init__` function:

```python
augmenter = ImageAugmenter(width, height, channel_is_first_axis=True)
```

The augmenter is able to augment every channel individually, e.g. rotating the red-channel by 20 degrees and rotating the blue-channel (of the same image) by -5 degrees.
To do that, simply set the flag `transform_channels_equally` to `False` in the constructor:

```python
augmenter = ImageAugmenter(width, height, transform_channels_equally=False)
```

Note that this setting currently does not affect vertical and horizontal flipping.
Those will always be applied to all channels of an image equally.

# Performance

Required time to augment 1 million images of size 32x32 (3 channels) is about 4 minutes when tested on an 3.5ghz i7 (haswell).
Larger images will require more time. The required time seems to grow linearly with the number of pixels in an image.

# Tests

The tests and checks are in the `tests/` directory.
You can run them using (from within that directory):

```python
python TestImageAugmenter.py
python CheckPerformance.py
python CheckPlotImages.py
```

where `CheckPerformance.py` measures the performance of the class on your machine and `CheckPlotImages.py` shows some plots with example augmentations.
