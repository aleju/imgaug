"""
Tests to visually inspect the results of the library's functionality.
Run checks via
    python check_visually.py
"""
from __future__ import print_function, division

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import numpy as np
from scipy import ndimage, misc
from skimage import data
import argparse

def main():
    parser = argparse.ArgumentParser(description="Check augmenters visually.")
    parser.add_argument('--only', default=None, help="If this is set, then only the results of an augmenter with this name will be shown.", required=False)
    args = parser.parse_args()

    images = [
        ia.quokka_square(size=(128, 128)),
        misc.imresize(data.astronaut(), (128, 128))
    ]

    augmenters = [
        iaa.Noop(name="Noop"),
        iaa.OneOf(children=[
            iaa.CoarseDropout(p=0.5, size_percent=0.05),
            iaa.AdditiveGaussianNoise(scale=0.1*255),
            iaa.Crop(percent=0.1)
        ], name="OneOf"),
        iaa.AddToHueAndSaturation((-20, 20), per_channel=True, name="AddToHueAndSaturation"),
        iaa.Crop(px=(0, 8), name="Crop-px"),
        iaa.Crop(percent=(0, 0.1), name="Crop-percent"),
        iaa.Fliplr(0.5, name="Fliplr"),
        iaa.Flipud(0.5, name="Flipud"),
        iaa.Superpixels(p_replace=0.75, n_segments=50, name="Superpixels"),
        iaa.Grayscale(0.5, name="Grayscale0.5"),
        iaa.Grayscale(1.0, name="Grayscale1.0"),
        iaa.AverageBlur(k=(3, 11), name="AverageBlur"),
        iaa.GaussianBlur((0, 3.0), name="GaussianBlur"),
        iaa.MedianBlur(k=(3, 11), name="MedianBlur"),
        iaa.Sharpen(alpha=(0.1, 1.0), lightness=(0, 2.0), name="Sharpen"),
        iaa.Emboss(alpha=(0.1, 1.0), strength=(0, 2.0), name="Emboss"),
        iaa.EdgeDetect(alpha=(0.1, 1.0), name="EdgeDetect"),
        iaa.DirectedEdgeDetect(alpha=(0.1, 1.0), direction=(0, 1.0), name="DirectedEdgeDetect"),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1*255), name="AdditiveGaussianNoise"),
        iaa.Dropout((0.0, 0.1), name="Dropout"),
        iaa.CoarseDropout(p=0.05, size_percent=(0.05, 0.5), name="CoarseDropout"),
        iaa.Invert(p=0.5, name="Invert"),
        iaa.Invert(p=0.5, per_channel=True, name="InvertPerChannel"),
        iaa.Add((-50, 50), name="Add"),
        iaa.Add((-50, 50), per_channel=True, name="AddPerChannel"),
        iaa.AddElementwise((-50, 50), name="AddElementwise"),
        iaa.Multiply((0.5, 1.5), name="Multiply"),
        iaa.Multiply((0.5, 1.5), per_channel=True, name="MultiplyPerChannel"),
        iaa.MultiplyElementwise((0.5, 1.5), name="MultiplyElementwise"),
        iaa.ContrastNormalization(alpha=(0.5, 2.0), name="ContrastNormalization"),
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_px={"x": (-16, 16), "y": (-16, 16)},
            rotate=(-45, 45),
            shear=(-16, 16),
            order=ia.ALL,
            cval=(0, 255),
            mode=ia.ALL,
            name="Affine"
        ),
        iaa.PiecewiseAffine(scale=0.03, nb_rows=(2, 6), nb_cols=(2, 6), name="PiecewiseAffine"),
        iaa.ElasticTransformation(alpha=(0.5, 8.0), sigma=1.0, name="ElasticTransformation"),
        iaa.PerspectiveTransform(scale=0.1, name="PerspectiveTransform"),
    ]

    augmenters.append(iaa.Sequential([iaa.Sometimes(0.2, aug.copy()) for aug in augmenters], name="Sequential"))
    augmenters.append(iaa.Sometimes(0.5, [aug.copy() for aug in augmenters], name="Sometimes"))

    for augmenter in augmenters:
        if args.only is None or augmenter.name == args.only:
            print("Augmenter: %s" % (augmenter.name,))
            grid = augmenter.draw_grid(images, rows=1, cols=16)
            misc.imshow(grid)

if __name__ == "__main__":
    main()
