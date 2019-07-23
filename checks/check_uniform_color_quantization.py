from __future__ import print_function, division
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa


def main():
    image = ia.quokka_square((256, 256))
    ia.imshow(
        ia.draw_grid([
            iaa.quantize_colors_uniform(image, 2),
            iaa.quantize_colors_uniform(image, 4),
            iaa.quantize_colors_uniform(image, 8),
            iaa.quantize_colors_uniform(image, 16),
            iaa.quantize_colors_uniform(image, 32),
            iaa.quantize_colors_uniform(image, 64)
        ], cols=6)
    )

    aug = iaa.UniformColorQuantization((2, 16))
    ia.imshow(ia.draw_grid(aug(images=[image] * 16)))


if __name__ == "__main__":
    main()
