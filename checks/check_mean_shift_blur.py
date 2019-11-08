from __future__ import print_function, division, absolute_import
import imgaug as ia
import imgaug.augmenters as iaa


def main():
    image = ia.quokka_square((128, 128))
    aug = iaa.MeanShiftBlur()
    images_aug = aug(images=[image] * 16)
    ia.imshow(ia.draw_grid(images_aug))


if __name__ == "__main__":
    main()
