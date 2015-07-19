from __future__ import print_function
import numpy as np
from ImageAugmenter import ImageAugmenter, create_aug_matrices
import random
from scipy import misc
from skimage import data
import time

def main():
    batch_size = 64
    nb_runs = 20

    # Measure time required to generate 100k augmentation matrices
    print("Generating 100 times 1000 augmentation matrices of size 64x64...")
    start = time.time()
    for i in range(100):
        create_aug_matrices(1000, 64, 64,
                            scale_to_percent=1.5, scale_axis_equally=False,
                            rotation_deg=20, shear_deg=20,
                            translation_x_px=5, translation_y_px=5)
    print("Done in %.8f" % (time.time() - start,))

    # Test Performance on 64 images of size 512x512 pixels
    image = data.lena()
    images = np.resize(image, (batch_size, image.shape[0], image.shape[1], image.shape[2]))
    ia = ImageAugmenter(image.shape[0], image.shape[1],
                        hflip=True, vflip=True,
                        scale_to_percent=1.3, scale_axis_equally=False,
                        rotation_deg=25, shear_deg=10,
                        translation_x_px=5, translation_y_px=5)
    print("Running tests on %d images of shape %s" % (batch_size, str(image.shape)))
    run_tests(ia, images, nb_runs)
    print("")
    
    print("Running tests on %d images of shape %s" % (batch_size, str(image.shape)))
    print("(With 1000 pregenerated matrices)")
    ia.pregenerate_matrices(1000)
    run_tests(ia, images, nb_runs)
    print("")

    # Test Performance on 64 images of size 64x64 pixels
    image = data.lena()
    image = misc.imresize(image, (64,64))
    images = np.resize(image, (batch_size, image.shape[0], image.shape[1], image.shape[2]))
    ia = ImageAugmenter(image.shape[0], image.shape[1],
                        hflip=True, vflip=True,
                        scale_to_percent=1.3, scale_axis_equally=False,
                        rotation_deg=25, shear_deg=10,
                        translation_x_px=5, translation_y_px=5)
    print("Running tests on %d images of shape %s" % (batch_size, str(image.shape)))
    run_tests(ia, images, nb_runs)

    print("Running tests on %d images of shape %s" % (batch_size, str(image.shape)))
    print("(With 1000 pregenerated matrices)")
    ia.pregenerate_matrices(1000)
    run_tests(ia, images, nb_runs)
    print("")

def run_tests(ia, images, nb_runs):
    results = np.zeros((nb_runs,))
    for i in range(nb_runs):
        start = time.time()
        ia.augment_batch(images)
        results[i] = time.time() - start
        print("Run %d: %.8fs" % (i, results[i]))
    print("Mean: %.8fs" % (results.mean(),))
    print("Sum: %.8fs" % (results.sum(),))

if __name__ == "__main__":
    main()
