from __future__ import print_function, division
import imgaug as ia
import imgaug.augmenters as iaa


def main():
    image = ia.quokka_square(size=(128, 128))
    images = []

    for i in range(15):
        aug = iaa.WithHueAndSaturation(iaa.WithChannels(0, iaa.Add(i*20)))
        images.append(aug.augment_image(image))

    for i in range(15):
        aug = iaa.WithHueAndSaturation(iaa.WithChannels(1, iaa.Add(i*20)))
        images.append(aug.augment_image(image))

    ia.imshow(ia.draw_grid(images, rows=2))


if __name__ == "__main__":
    main()
