from __future__ import print_function, division, absolute_import

import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa


def main():
    image = ia.quokka(0.25)

    for N in [1, 2]:
        print("N=%d" % (N,))

        images_aug = []
        for M in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
            images_aug.extend(
                iaa.RandAugment(n=N, m=M, random_state=1)(images=[image] * 10)
            )
        ia.imshow(ia.draw_grid(images_aug, cols=10))

    for M in [0, 1, 2, 4, 8, 10]:
        print("M=%d" % (M,))
        aug = iaa.RandAugment(m=M, random_state=1)

        images_aug = []
        for _ in np.arange(6):
            images_aug.extend(aug(images=[image] * 16))

        ia.imshow(ia.draw_grid(images_aug, cols=16, rows=6))


if __name__ == "__main__":
    main()
