import imgaug as ia
import imgaug.augmenters as iaa


def main():
    image = ia.quokka_square((128, 128))
    images_aug = iaa.Solarize(1.0)(images=[image] * (5*5))
    ia.imshow(ia.draw_grid(images_aug))


if __name__ == "__main__":
    main()
