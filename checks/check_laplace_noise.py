from __future__ import print_function, division

import imgaug as ia
from imgaug import augmenters as iaa


def main():
    img = ia.quokka(0.5)
    mul = 0.025
    augs = [
        ("iaa.AdditiveLaplaceNoise(255*(1*mul))", iaa.AdditiveLaplaceNoise(scale=255*(1*mul))),
        ("iaa.AdditiveLaplaceNoise(255*(2*mul))", iaa.AdditiveLaplaceNoise(scale=255*(2*mul))),
        ("iaa.AdditiveLaplaceNoise(255*(3*mul))", iaa.AdditiveLaplaceNoise(scale=255*(3*mul))),
        ("iaa.AdditiveLaplaceNoise(255*(4*mul))", iaa.AdditiveLaplaceNoise(scale=255*(4*mul))),
        ("iaa.AdditiveLaplaceNoise((255*(0*mul), 255*(4*mul)))",
         iaa.AdditiveLaplaceNoise(scale=(255*(0*mul), 255*(4*mul)))),
        ("iaa.AdditiveLaplaceNoise([255*(1*mul), 255*(2*mul), 255*(3*mul)])",
         iaa.AdditiveLaplaceNoise(scale=[255*(1*mul), 255*(2*mul), 255*(3*mul)])),
        ("iaa.AdditiveLaplaceNoise(255*(2*mul), per_channel=True)",
         iaa.AdditiveLaplaceNoise(scale=255*(2*mul), per_channel=True)),
    ]
    for descr, aug in augs:
        print(descr)
        imgs_aug = aug.augment_images([img] * 16)
        ia.imshow(ia.draw_grid(imgs_aug))


if __name__ == "__main__":
    main()
