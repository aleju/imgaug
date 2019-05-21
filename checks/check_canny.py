from __future__ import print_function, division
import imgaug as ia
import imgaug.augmenters as iaa


def main():
    black_and_white = iaa.RandomColorsBinaryImageColorizer(
        color_true=255, color_false=0)

    print("alpha=1.0, black and white")
    image = ia.quokka_square((128, 128))
    aug = iaa.Canny(alpha=1.0, colorizer=black_and_white)
    ia.imshow(ia.draw_grid(aug(images=[image] * (5*5))))

    print("alpha=1.0, random color")
    image = ia.quokka_square((128, 128))
    aug = iaa.Canny(alpha=1.0)
    ia.imshow(ia.draw_grid(aug(images=[image] * (5*5))))

    print("alpha=1.0, sobel ksize=[3, 13], black and white")
    image = ia.quokka_square((128, 128))
    aug = iaa.Canny(alpha=1.0, sobel_kernel_size=[3, 7],
                    colorizer=black_and_white)
    ia.imshow(ia.draw_grid(aug(images=[image] * (5*5))))

    print("alpha=1.0, sobel ksize=3, black and white")
    image = ia.quokka_square((128, 128))
    aug = iaa.Canny(alpha=1.0, sobel_kernel_size=3,
                    colorizer=black_and_white)
    ia.imshow(ia.draw_grid(aug(images=[image] * (5*5))))

    print("fully random")
    image = ia.quokka_square((128, 128))
    aug = iaa.Canny()
    ia.imshow(ia.draw_grid(aug(images=[image] * (5*5))))


if __name__ == "__main__":
    main()
