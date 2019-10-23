from __future__ import print_function, division
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa


def main():
    image = ia.quokka_square(size=(100, 100))

    for cspace in iaa.WithBrightnessChannels._VALID_COLORSPACES:
        print(cspace, "add")
        images_aug = []
        for add in np.linspace(-200, 200, 64):
            aug = iaa.MultiplyAndAddToBrightness(add=add, mul=1.0,
                                                 to_colorspace=cspace)
            images_aug.append(aug(image=image))

        ia.imshow(ia.draw_grid(images_aug))

    for cspace in iaa.WithBrightnessChannels._VALID_COLORSPACES:
        print(cspace, "mul")
        images_aug = []
        for mul in np.linspace(0.5, 1.5, 64):
            aug = iaa.MultiplyAndAddToBrightness(add=0, mul=mul,
                                                 to_colorspace=cspace)
            images_aug.append(aug(image=image))

        ia.imshow(ia.draw_grid(images_aug))

    for cspace in iaa.WithBrightnessChannels._VALID_COLORSPACES:
        print(cspace, "defaults")
        aug = iaa.MultiplyAndAddToBrightness(to_colorspace=cspace)
        images_aug = aug(images=[image] * 64)
        ia.imshow(ia.draw_grid(images_aug))


if __name__ == "__main__":
    main()
