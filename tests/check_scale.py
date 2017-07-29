from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa
from scipy import misc
import numpy as np
from skimage import data
import cv2

def main():
    image = data.astronaut()
    kps = [ia.KeypointsOnImage(
        [ia.Keypoint(x=32, y=64), ia.Keypoint(48, 70), ia.Keypoint(256, 256)],
        shape=image.shape
    )]
    print("image shape:", image.shape)

    augs = [
        iaa.Scale(32, name="i32"),
        iaa.Scale((32, 256), name="i32i256"),
        iaa.Scale(0.5, name="f05"),
        iaa.Scale((0.9, 0.1), name="f09f01"),
    ]

    print("original", image.shape)
    misc.imshow(kps[0].draw_on_image(image))
    for aug in augs:
        img_aug = aug.augment_image(image)
        kps_aug = aug.augment_keypoints(kps)[0]
        img_aug_kps = kps_aug.draw_on_image(img_aug)
        print(aug.name, img_aug_kps.shape)
        misc.imshow(img_aug_kps)

    print("nearest/cv2.INTER_NEAREST/cubic")
    misc.imshow(np.hstack([
        iaa.Scale(64, interpolation="nearest").augment_image(image),
        iaa.Scale(64, interpolation=cv2.INTER_NEAREST).augment_image(image),
        iaa.Scale(64, interpolation="cubic").augment_image(image)
    ]))

    print("random nearest/cubic")
    iaa.Scale(64, interpolation=["nearest", "cubic"]).show_grid([image], 8, 8)

if __name__ == "__main__":
    main()
