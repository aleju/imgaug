from __future__ import print_function, division

import cv2
import numpy as np
from skimage import data

import imgaug as ia
from imgaug import augmenters as iaa


def main():
    # test 2d image
    ia.imshow(iaa.Resize(64).augment_image(data.camera()))

    # test many images
    images = [ia.quokka(size=0.5), ia.quokka(size=0.5)]
    images_aug = iaa.Resize(64).augment_images(images)
    ia.imshow(np.hstack(images_aug))

    image = ia.quokka(size=0.5)
    kps = [ia.KeypointsOnImage(
        [ia.Keypoint(x=245, y=203), ia.Keypoint(x=365, y=195), ia.Keypoint(x=313, y=269)],
        shape=(image.shape[0]*2, image.shape[1]*2)
    )]
    kps[0] = kps[0].on(image.shape)
    print("image shape:", image.shape)

    augs = [
        iaa.Resize("keep", name="keep"),
        iaa.Resize(32, name="i32"),
        iaa.Resize(0.5, name="f05"),

        iaa.Resize({"height": 32}, name="height32"),
        iaa.Resize({"width": 32}, name="width32"),
        iaa.Resize({"height": "keep", "width": 32}, name="keep-width32"),
        iaa.Resize({"height": 32, "width": "keep"}, name="height32-keep"),
        iaa.Resize({"height": "keep", "width": "keep"}, name="keep-keep"),
        iaa.Resize({"height": 32, "width": 64}, name="height32width64"),
        iaa.Resize({"height": 64, "width": "keep-aspect-ratio"}, name="height64width-kar"),
        iaa.Resize({"height": "keep-aspect-ratio", "width": 64}, name="height-kar_width64")
    ]

    augs_many = [
        iaa.Resize((32, 128), name="tuple-32-128"),
        iaa.Resize([32, 64, 128], name="list-32-64-128"),
        iaa.Resize({"height": (32, 128), "width": "keep"}, name="height-32-64_width-keep"),
        iaa.Resize({"height": (32, 128), "width": "keep-aspect-ratio"}, name="height-32-128_width-kar"),
        iaa.Resize({"height": (32, 128), "width": (32, 128)}, name="height-32-128_width-32-128")
    ]

    print("original", image.shape)
    ia.imshow(kps[0].draw_on_image(image))

    print("-----------------")
    print("Same size per image")
    print("-----------------")
    for aug in augs:
        img_aug = aug.augment_image(image)
        kps_aug = aug.augment_keypoints(kps)[0]
        img_aug_kps = kps_aug.draw_on_image(img_aug)
        print(aug.name, img_aug_kps.shape, img_aug_kps.shape[1]/img_aug_kps.shape[0])
        ia.imshow(img_aug_kps)

    print("-----------------")
    print("Random per image")
    print("-----------------")
    for aug in augs_many:
        images_aug = []
        for _ in range(64):
            aug_det = aug.to_deterministic()
            img_aug = aug_det.augment_image(image)
            kps_aug = aug_det.augment_keypoints(kps)[0]
            img_aug_kps = kps_aug.draw_on_image(img_aug)
            images_aug.append(img_aug_kps)
        print(aug.name)
        ia.imshow(ia.draw_grid(images_aug))

    print("nearest/cv2.INTER_NEAREST/cubic")
    ia.imshow(np.hstack([
        iaa.Resize(64, interpolation="nearest").augment_image(image),
        iaa.Resize(64, interpolation=cv2.INTER_NEAREST).augment_image(image),
        iaa.Resize(64, interpolation="cubic").augment_image(image)
    ]))

    print("random nearest/cubic")
    iaa.Resize(64, interpolation=["nearest", "cubic"]).show_grid([image], 8, 8)


if __name__ == "__main__":
    main()
