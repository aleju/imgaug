"""Rough measurements of the performance of the ImageAugmenter."""
from __future__ import print_function

# make sure that ImageAugmenter can be imported from parent directory
if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import numpy as np
from ImageAugmenter import ImageAugmenter, create_aug_matrices
from scipy import misc
from skimage import data
from skimage import transform as tf
import time

def main():
    """Measure time required to generate augmentations matrices and to apply
    them.
    """
    batch_size = 64
    nb_runs = 20

    # Measure time required to generate 100k augmentation matrices
    """
    print("Generating 100 times 1000 augmentation matrices of size 64x64...")
    start = time.time()
    for _ in range(100):
        create_aug_matrices(1000, 64, 64,
                            scale_to_percent=1.5, scale_axis_equally=False,
                            rotation_deg=20, shear_deg=20,
                            translation_x_px=5, translation_y_px=5)
    print("Done in %.8f" % (time.time() - start,))
    """

    # Test Performance on 64 images of size 512x512 pixels
    image = data.lena()
    images = np.resize(image, (batch_size, image.shape[0], image.shape[1], image.shape[2]))
    augmenter = ImageAugmenter(image.shape[0], image.shape[1],
                               hflip=True, vflip=True,
                               scale_to_percent=1.3, scale_axis_equally=False,
                               rotation_deg=25, shear_deg=10,
                               translation_x_px=5, translation_y_px=5)
    print("Running tests on %d images of shape %s" % (batch_size, str(image.shape)))
    run_tests(augmenter, images, nb_runs)
    print("")

    print("Running tests on %d images of shape %s" % (batch_size, str(image.shape)))
    print("(With 1000 pregenerated matrices)")
    augmenter.pregenerate_matrices(1000)
    run_tests(augmenter, images, nb_runs)
    print("")

    # Test Performance on 64 images of size 64x64 pixels
    image = data.lena()
    image = misc.imresize(image, (64, 64))
    images = np.resize(image, (batch_size, image.shape[0], image.shape[1], image.shape[2]))
    augmenter = ImageAugmenter(image.shape[0], image.shape[1],
                               hflip=True, vflip=True,
                               scale_to_percent=1.3, scale_axis_equally=False,
                               rotation_deg=25, shear_deg=10,
                               translation_x_px=5, translation_y_px=5)
    print("Running tests on %d images of shape %s" % (batch_size, str(image.shape)))
    run_tests(augmenter, images, nb_runs)

    print("Running tests on %d images of shape %s" % (batch_size, str(image.shape)))
    print("(With 1000 pregenerated matrices)")
    augmenter.pregenerate_matrices(1000)
    run_tests(augmenter, images, nb_runs)
    print("")

    # Time required to augment 1,000,000 images of size 32x32
    print("Augmenting 1000 batches of 1000 lena images (1 million total)" \
          ", each of size 32x32...")
    image = data.lena()
    image = misc.imresize(image, (32, 32))
    batch_size = 1000
    images = np.resize(image, (batch_size, image.shape[0], image.shape[1], image.shape[2]))
    augmenter = ImageAugmenter(image.shape[1], image.shape[0],
                               hflip=True, vflip=True,
                               scale_to_percent=1.3, scale_axis_equally=False,
                               rotation_deg=25, shear_deg=10,
                               translation_x_px=5, translation_y_px=5)
    augmenter.pregenerate_matrices(1000)

    start = time.time()
    for _ in range(1000):
        augmenter.augment_batch(images)
    print("Done in %.8fs" % (time.time() - start,))
    print("")

    # Time required to augment 1,000,000 images of size 32x32
    # but using only one matrix without the class (no library overhead from
    # ImageAugmenter)
    # Notice that this does not include horizontal and vertical flipping,
    # which is done via numpy in the ImageAugmenter class.
    print("Augmenting 1000 batches of 1000 lena images (1 million total)" \
          ", each of size 32x32, using one matrix directly (no ImageAugmenter " \
          "class)...")
    matrices = create_aug_matrices(1, image.shape[1], image.shape[0],
                                   scale_to_percent=1.3, scale_axis_equally=False,
                                   rotation_deg=25, shear_deg=10,
                                   translation_x_px=5, translation_y_px=5)
    matrix = matrices[0]

    start = time.time()
    for _ in range(1000):
        for image in images:
            augmented_image = tf.warp(image, matrix)
    print("Done in %.8fs" % (time.time() - start,))


def run_tests(augmenter, images, nb_runs):
    """Perform nb_runs augmentations of the provided images and measure the
    required time for that.
    """
    results = np.zeros((nb_runs,))
    for i in range(nb_runs):
        start = time.time()
        augmenter.augment_batch(images)
        results[i] = time.time() - start
        print("Run %d: %.8fs" % (i, results[i]))
    print("Mean: %.8fs" % (results.mean(),))
    print("Sum: %.8fs" % (results.sum(),))

if __name__ == "__main__":
    main()
