from __future__ import print_function, division

import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa


def main():
    image = ia.quokka_square((128, 128))
    aug = iaa.AveragePool(2)
    ia.imshow(
        ia.draw_grid(aug.augment_images([image] * (5*5)))
    )

    aug = iaa.AveragePool(2, keep_size=False)
    ia.imshow(
        ia.draw_grid(aug.augment_images([image] * (5*5)))
    )

    aug_pool = iaa.AveragePool(((0, 10), (0, 10)))
    aug_blur = iaa.AverageBlur(((0, 10), (0, 10)))
    ia.imshow(
        np.hstack([
            ia.draw_grid(aug_pool.augment_images([image] * (4*5)), cols=4),
            ia.draw_grid(aug_blur.augment_images([image] * (4*5)), cols=4)
        ])
    )


if __name__ == "__main__":
    main()
