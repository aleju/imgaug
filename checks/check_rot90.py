from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa


def main():
    augs = [
        ("iaa.Rot90(-1, keep_size=False)", iaa.Rot90(-1, keep_size=False)),
        ("iaa.Rot90(0, keep_size=False)", iaa.Rot90(0, keep_size=False)),
        ("iaa.Rot90(1, keep_size=False)", iaa.Rot90(1, keep_size=False)),
        ("iaa.Rot90(2, keep_size=False)", iaa.Rot90(2, keep_size=False)),
        ("iaa.Rot90(3, keep_size=False)", iaa.Rot90(3, keep_size=False)),
        ("iaa.Rot90(4, keep_size=False)", iaa.Rot90(4, keep_size=False)),
        ("iaa.Rot90(-1, keep_size=True)", iaa.Rot90(-1, keep_size=True)),
        ("iaa.Rot90(0, keep_size=True)", iaa.Rot90(0, keep_size=True)),
        ("iaa.Rot90(1, keep_size=True)", iaa.Rot90(1, keep_size=True)),
        ("iaa.Rot90(2, keep_size=True)", iaa.Rot90(2, keep_size=True)),
        ("iaa.Rot90(3, keep_size=True)", iaa.Rot90(3, keep_size=True)),
        ("iaa.Rot90(4, keep_size=True)", iaa.Rot90(4, keep_size=True)),
        ("iaa.Rot90([0, 1, 2, 3, 4], keep_size=False)", iaa.Rot90([0, 1, 2, 3, 4], keep_size=False)),
        ("iaa.Rot90([0, 1, 2, 3, 4], keep_size=True)", iaa.Rot90([0, 1, 2, 3, 4], keep_size=True)),
        ("iaa.Rot90((0, 4), keep_size=False)", iaa.Rot90((0, 4), keep_size=False)),
        ("iaa.Rot90((0, 4), keep_size=True)", iaa.Rot90((0, 4), keep_size=True)),
        ("iaa.Rot90((1, 3), keep_size=False)", iaa.Rot90((1, 3), keep_size=False)),
        ("iaa.Rot90((1, 3), keep_size=True)", iaa.Rot90((1, 3), keep_size=True))
    ]

    image = ia.quokka(0.25)

    print("--------")
    print("Image + Keypoints")
    print("--------")
    kps = ia.quokka_keypoints(0.25)
    for name, aug in augs:
        print(name, "...")
        aug_det = aug.to_deterministic()
        images_aug = aug_det.augment_images([image] * 16)
        kps_aug = aug_det.augment_keypoints([kps] * 16)
        images_aug = [kps_aug_i.draw_on_image(image_aug_i, size=5)
                      for image_aug_i, kps_aug_i in zip(images_aug, kps_aug)]
        ia.imshow(ia.draw_grid(images_aug))

    print("--------")
    print("Image + Heatmaps (low res)")
    print("--------")
    hms = ia.quokka_heatmap(0.10)
    for name, aug in augs:
        print(name, "...")
        aug_det = aug.to_deterministic()
        images_aug = aug_det.augment_images([image] * 16)
        hms_aug = aug_det.augment_heatmaps([hms] * 16)
        images_aug = [hms_aug_i.draw_on_image(image_aug_i)[0]
                      for image_aug_i, hms_aug_i in zip(images_aug, hms_aug)]
        ia.imshow(ia.draw_grid(images_aug))


if __name__ == "__main__":
    main()
