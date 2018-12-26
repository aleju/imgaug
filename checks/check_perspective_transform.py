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

    augs = [
        iaa.PerspectiveTransform(scale=0.01, name="pt001", keep_size=True),
        iaa.PerspectiveTransform(scale=0.1, name="pt01", keep_size=True),
        iaa.PerspectiveTransform(scale=0.2, name="pt02", keep_size=True),
        iaa.PerspectiveTransform(scale=0.3, name="pt03", keep_size=True),
        iaa.PerspectiveTransform(scale=(0, 0.3), name="pt00to03", keep_size=True)
    ]

    print("original", image.shape)
    ia.imshow(kps[0].draw_on_image(image))

    print("-----------------")
    print("Random aug per image")
    print("-----------------")
    for aug in augs:
        images_aug = []
        for _ in range(16):
            aug_det = aug.to_deterministic()
            img_aug = aug_det.augment_image(image)
            kps_aug = aug_det.augment_keypoints(kps)[0]
            img_aug_kps = kps_aug.draw_on_image(img_aug)
            img_aug_kps = np.pad(img_aug_kps, ((1, 1), (1, 1), (0, 0)), mode="constant", constant_values=255)
            images_aug.append(img_aug_kps)
        print(aug.name)
        ia.imshow(ia.draw_grid(images_aug))

    print("----------------")
    print("6 channels")
    print("----------------")
    image6 = np.dstack([image, image])
    image6_aug = augs[1].augment_image(image6)
    ia.imshow(
        np.hstack([image6_aug[..., 0:3], image6_aug[..., 3:6]])
    )


if __name__ == "__main__":
    main()
