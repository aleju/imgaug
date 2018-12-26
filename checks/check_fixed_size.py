from __future__ import print_function, division

import numpy as np

import imgaug as ia
from imgaug import augmenters as iaa


def main():
    image = ia.quokka(size=0.5)
    kps = [ia.KeypointsOnImage(
        [ia.Keypoint(x=245, y=203), ia.Keypoint(x=365, y=195), ia.Keypoint(x=313, y=269)],
        shape=(image.shape[0]*2, image.shape[1]*2)
    )]
    kps[0] = kps[0].on(image.shape)
    print("image shape:", image.shape)
    
    augs_many = [
        iaa.PadToFixedSize(200, 200, name="pad-width200-height200"),
        iaa.PadToFixedSize(200, 322, name="pad-width200-height322"),
        iaa.PadToFixedSize(200, 400, name="pad-width200-height400"),
        iaa.PadToFixedSize(480, 200, name="pad-width480-height200"),
        iaa.PadToFixedSize(480, 322, name="pad-width480-height322"),  # input size == output size
        iaa.PadToFixedSize(480, 400, name="pad-width480-height400"),
        iaa.PadToFixedSize(600, 200, name="pad-width600-height200"),
        iaa.PadToFixedSize(600, 322, name="pad-width600-height322"),
        iaa.PadToFixedSize(600, 400, name="pad-width600-height400"),

        iaa.CropToFixedSize(200, 200, name="crop-width200-height200"),
        iaa.CropToFixedSize(200, 322, name="crop-width200-height322"),
        iaa.CropToFixedSize(200, 400, name="crop-width200-height400"),
        iaa.CropToFixedSize(480, 200, name="crop-width480-height200"),
        iaa.CropToFixedSize(480, 322, name="crop-width480-height322"),  # input size == output size
        iaa.CropToFixedSize(480, 400, name="crop-width480-height400"),
        iaa.CropToFixedSize(600, 200, name="crop-width600-height200"),
        iaa.CropToFixedSize(600, 322, name="crop-width600-height322"),
        iaa.CropToFixedSize(600, 400, name="crop-width600-height400"),

        iaa.Sequential([
            iaa.PadToFixedSize(200, 200),
            iaa.CropToFixedSize(200, 200)
        ], name="pad-crop-width200-height200"),
        iaa.Sequential([
            iaa.PadToFixedSize(400, 400),
            iaa.CropToFixedSize(400, 400)
        ], name="pad-crop-width400-height400"),
        iaa.Sequential([
            iaa.PadToFixedSize(600, 600),
            iaa.CropToFixedSize(600, 600)
        ], name="pad-crop-width600-height600"),

        iaa.Sequential([
            iaa.CropToFixedSize(200, 200),
            iaa.PadToFixedSize(200, 200)
        ], name="crop-pad-width200-height200"),
        iaa.Sequential([
            iaa.CropToFixedSize(400, 400),
            iaa.PadToFixedSize(400, 400)
        ], name="crop-pad-width400-height400"),
        iaa.Sequential([
            iaa.CropToFixedSize(600, 600),
            iaa.PadToFixedSize(600, 600)
        ], name="crop-pad-width600-height600"),

    ]

    print("original", image.shape)
    ia.imshow(kps[0].draw_on_image(image))

    print("-----------------")
    print("Random aug per image")
    print("-----------------")
    for aug in augs_many:
        images_aug = []
        for _ in range(36):
            aug_det = aug.to_deterministic()
            img_aug = aug_det.augment_image(image)
            kps_aug = aug_det.augment_keypoints(kps)[0]
            img_aug_kps = kps_aug.draw_on_image(img_aug)
            img_aug_kps = np.pad(img_aug_kps, ((1, 1), (1, 1), (0, 0)), mode="constant", constant_values=255)
            images_aug.append(img_aug_kps)
        print(aug.name)
        ia.imshow(ia.draw_grid(images_aug))


if __name__ == "__main__":
    main()
