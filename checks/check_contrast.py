from __future__ import print_function, division
import argparse

import numpy as np
from skimage import data

import imgaug as ia
from imgaug import augmenters as iaa


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

    for gain in [0.0, 0.25, 0.5, 1.0, 2.0, (0.5, 1.5), [0.5, 1.0, 1.5]]:
        augs.append(("LogContrast " + str(gain), iaa.LogContrast(gain, per_channel=args.per_channel)))

    for alpha in [-1.0, 0.5, 0, 0.5, 1.0, 2.0, (0.5, 1.5), [0.5, 1.0, 1.5]]:
        augs.append(("LinearContrast " + str(alpha), iaa.LinearContrast(alpha, per_channel=args.per_channel)))

    augs.append(("AllChannelsHistogramEqualization", iaa.AllChannelsHistogramEqualization()))
    augs.append(("HistogramEqualization (Lab)", iaa.HistogramEqualization(to_colorspace=iaa.HistogramEqualization.Lab)))
    augs.append(("HistogramEqualization (HSV)", iaa.HistogramEqualization(to_colorspace=iaa.HistogramEqualization.HSV)))
    augs.append(("HistogramEqualization (HLS)", iaa.HistogramEqualization(to_colorspace=iaa.HistogramEqualization.HLS)))

    for clip_limit in [0.1, 1, 5, 10]:
        for tile_grid_size_px in [3, 7]:
            augs.append(("AllChannelsCLAHE %d %dx%d" % (clip_limit, tile_grid_size_px, tile_grid_size_px),
                         iaa.AllChannelsCLAHE(clip_limit=clip_limit, tile_grid_size_px=tile_grid_size_px,
                                              per_channel=args.per_channel)))

    for clip_limit in [1, 5, 10, 100, 200]:
        for tile_grid_size_px in [3, 7, 15]:
            augs.append(("CLAHE %d %dx%d" % (clip_limit, tile_grid_size_px, tile_grid_size_px),
                         iaa.CLAHE(clip_limit=clip_limit, tile_grid_size_px=tile_grid_size_px)))

    images = [data.astronaut()] * 16
    images = ia.imresize_many_images(np.uint8(images), (128, 128))
    for name, aug in augs:
        print("-----------")
        print(name)
        print("-----------")
        images_aug = aug.augment_images(images)
        images_aug[0] = images[0]
        grid = ia.draw_grid(images_aug, rows=4, cols=4)
        ia.imshow(grid)


if __name__ == "__main__":
    main()
