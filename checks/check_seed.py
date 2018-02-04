from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa
from scipy import misc
import numpy as np
from skimage import data

def main():
    img = data.astronaut()
    img = misc.imresize(img, (64, 64))
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
    misc.imshow(all_rows)

if __name__ == "__main__":
    main()
