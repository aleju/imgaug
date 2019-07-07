from __future__ import print_function, division
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa


def main():
    image = ia.quokka_square((256, 256))
    image_q2 = iaa.quantize_colors_kmeans(image, 2)
    image_q16 = iaa.quantize_colors_kmeans(image, 16)
    ia.imshow(np.hstack([image_q2, image_q16]))

    from_cs = "RGB"
    to_cs = ["RGB", "Lab"]
    kwargs = {"from_colorspace": from_cs, "to_colorspace": to_cs}
    augs = [
        iaa.KMeansColorQuantization(2, **kwargs),
        iaa.KMeansColorQuantization(4, **kwargs),
        iaa.KMeansColorQuantization(8, **kwargs),
        iaa.KMeansColorQuantization((2, 16), **kwargs),
    ]

    images_aug = []
    for aug in augs:
        images_aug.extend(aug(images=[image]*8))

    ia.imshow(ia.draw_grid(images_aug, cols=8))


if __name__ == "__main__":
    main()
