from __future__ import print_function, division
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa


def main():
    image = ia.quokka_square()
    images_aug = []
    for kelvin in np.linspace(1000, 10000, 64):
        images_aug.append(iaa.ChangeColorTemperature(kelvin)(image=image))

    ia.imshow(ia.draw_grid(images_aug))


if __name__ == "__main__":
    main()
