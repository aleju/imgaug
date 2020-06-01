from __future__ import print_function, division, absolute_import
import imgaug as ia
import imgaug.augmenters as iaa


def main():
    aug = iaa.Cutout(fill_mode=["gaussian", "constant"], cval=(0, 255),
                     fill_per_channel=0.5)
    image = ia.data.quokka()
    images_aug = aug(images=[image] * 16)
    ia.imshow(ia.draw_grid(images_aug, cols=4, rows=4))


if __name__ == "__main__":
    main()
