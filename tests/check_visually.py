"""
Tests to visually inspect the results of the library's functionality.
Run these checks from the project directory (i.e. parent directory) via
    python check_visually.py
"""
from __future__ import print_function, division

#import sys
#import os
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import numpy as np
from scipy import ndimage, misc
from skimage import data

def main():
    images = [
        misc.imresize(ndimage.imread("../quokka.jpg")[0:643, 0:643], (128, 128)),
        misc.imresize(data.astronaut(), (128, 128))
    ]

    augmenters = [
        iaa.Noop(name="Noop"),
        iaa.Crop(px=(0, 8), name="Crop-px"),
        iaa.Crop(percent=(0, 0.1), name="Crop-percent"),
        iaa.Fliplr(0.5, name="Fliplr"),
        iaa.Flipud(0.5, name="Flipud"),
        iaa.Grayscale(0.5, name="Grayscale0.5"),
        iaa.Grayscale(1.0, name="Grayscale1.0"),
        iaa.GaussianBlur((0, 3.0), name="GaussianBlur"),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1*255), name="AdditiveGaussianNoise"),
        iaa.Dropout((0.0, 0.1), name="Dropout"),
        iaa.Multiply((0.5, 1.5), name="Multiply"),
        iaa.ContrastNormalization(alpha=(0.5, 2.0), name="ContrastNormalization"),
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_px={"x": (-16, 16), "y": (-16, 16)},
            rotate=(-45, 45),
            shear=(-16, 16),
            order=ia.ALL,
            cval=(0, 1.0),
            mode=ia.ALL,
            name="Affine"
        ),
        iaa.ElasticTransformation(alpha=(0.5, 8.0), sigma=1.0, name="ElasticTransformation")
    ]

    #for i, aug in enumerate(augmenters):
        #print(i)
        #aug.deepcopy()
        #import copy
        #copy.deepcopy(aug)
    seq = iaa.Sequential([aug.copy() for aug in augmenters], name="Sequential")
    st = iaa.Sometimes(0.5, seq.copy(), name="Sometimes")
    augmenters.append(seq)
    augmenters.append(st)

    for augmenter in augmenters:
        print("Augmenter: %s" % (augmenter.name,))
        grid = augmenter.draw_grid(images, rows=1, cols=16)
        misc.imshow(grid)

if __name__ == "__main__":
    main()
