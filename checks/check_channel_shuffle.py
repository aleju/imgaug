from imgaug import augmenters as iaa
import imgaug as ia
import imgaug.random as iarandom

iarandom.seed(1)


def main():
    img = ia.data.quokka(size=(128, 128), extract="square")

    aug = iaa.ChannelShuffle()
    imgs_aug = aug.augment_images([img] * 64)
    ia.imshow(ia.draw_grid(imgs_aug))

    aug = iaa.ChannelShuffle(p=0.1)
    imgs_aug = aug.augment_images([img] * 64)
    ia.imshow(ia.draw_grid(imgs_aug))

    aug = iaa.ChannelShuffle(p=1.0, channels=[0, 1])
    imgs_aug = aug.augment_images([img] * 64)
    ia.imshow(ia.draw_grid(imgs_aug))

    aug = iaa.ChannelShuffle(p=1.0, channels=[1, 2])
    imgs_aug = aug.augment_images([img] * 64)
    ia.imshow(ia.draw_grid(imgs_aug))

    aug = iaa.ChannelShuffle(p=1.0, channels=[1, 1, 2])
    imgs_aug = aug.augment_images([img] * 64)
    ia.imshow(ia.draw_grid(imgs_aug))

    aug = iaa.ChannelShuffle(p=1.0, channels=ia.ALL)
    imgs_aug = aug.augment_images([img] * 64)
    ia.imshow(ia.draw_grid(imgs_aug))


if __name__ == "__main__":
    main()
