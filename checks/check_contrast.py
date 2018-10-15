import os
import sys
#sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))
import imgaug as ia
from imgaug import augmenters as iaa
from skimage import data
import numpy as np
from scipy import misc
import argparse

def main():
    parser = argparse.ArgumentParser(description="Contrast check script")
    parser.add_argument("--per_channel", dest="per_channel", action="store_true")
    args = parser.parse_args()

    augs = []
    for p in [0.25, 0.5, 1.0, 2.0, (0.5, 1.5), [0.5, 1.0, 1.5]]:
        augs.append(("GammaContrast " + str(p), iaa.GammaContrast(p, per_channel=args.per_channel)))

    for cutoff in [0.25, 0.5, 0.75]:
        for gain in [5, 10, 15, 20, 25]:
            augs.append(("SigmoidContrast " + str(cutoff) + " " + str(gain), iaa.SigmoidContrast(gain, cutoff, per_channel=args.per_channel)))

    for gain in [-1.0, 0.0, 0.25, 0.5, 1.0, 2.0, (0.5, 1.5), [0.5, 1.0, 1.5]]:
        augs.append(("LogContrast " + str(gain), iaa.LogContrast(gain, per_channel=args.per_channel)))

    for alpha in [-1.0, 0.5, 0, 0.5, 1.0, 2.0, (0.5, 1.5), [0.5, 1.0, 1.5]]:
        augs.append(("LinearContrast " + str(alpha), iaa.LinearContrast(alpha, per_channel=args.per_channel)))

    images = [data.astronaut()] * 16
    images = ia.imresize_many_images(np.uint8(images), (128, 128))
    for name, aug in augs:
        print("-----------")
        print(name)
        print("-----------")
        images_aug = aug.augment_images(images)
        grid = ia.draw_grid(images_aug, rows=4, cols=4)
        ia.imshow(grid)

if __name__ == "__main__":
    main()
