from __future__ import print_function, division

import numpy as np
from skimage import data

import imgaug as ia
from imgaug import augmenters as iaa


def main():
    img = data.astronaut()
    img = ia.imresize_single_image(img, (64, 64))
    aug = iaa.Fliplr(0.5)
    unseeded1 = aug.draw_grid(img, cols=8, rows=1)
    unseeded2 = aug.draw_grid(img, cols=8, rows=1)

    ia.seed(1000)
    seeded1 = aug.draw_grid(img, cols=8, rows=1)
    seeded2 = aug.draw_grid(img, cols=8, rows=1)

    ia.seed(1000)
    reseeded1 = aug.draw_grid(img, cols=8, rows=1)
    reseeded2 = aug.draw_grid(img, cols=8, rows=1)

    ia.seed(1001)
    reseeded3 = aug.draw_grid(img, cols=8, rows=1)
    reseeded4 = aug.draw_grid(img, cols=8, rows=1)

    all_rows = np.vstack([unseeded1, unseeded2, seeded1, seeded2, reseeded1, reseeded2, reseeded3, reseeded4])
    ia.imshow(all_rows)


if __name__ == "__main__":
    main()
