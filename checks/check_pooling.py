from __future__ import print_function, division

import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa


def main():
    run(iaa.AveragePooling)
    run(iaa.MaxPooling)
    run(iaa.MinPooling)
    run(iaa.MedianPooling)


def run(clazz):
    image = ia.quokka_square((128, 128))
    aug = clazz(2)
    ia.imshow(
        ia.draw_grid(aug.augment_images([image] * (5*5)))
    )

    aug = clazz(2, keep_size=False)
    ia.imshow(
        ia.draw_grid(aug.augment_images([image] * (5*5)))
    )

    aug_pool = clazz(((0, 10), (0, 10)))
    aug_blur = clazz(((0, 10), (0, 10)))
    ia.imshow(
        np.hstack([
            ia.draw_grid(aug_pool.augment_images([image] * (4*5)), cols=4),
            ia.draw_grid(aug_blur.augment_images([image] * (4*5)), cols=4)
        ])
    )


if __name__ == "__main__":
    main()
