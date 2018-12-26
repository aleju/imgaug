from __future__ import print_function, division

import numpy as np
from skimage import data
import cv2

import imgaug as ia
from imgaug import augmenters as iaa

VAL_PER_STEP = 1
TIME_PER_STEP = 10


def main():
    image = data.astronaut()

    cv2.namedWindow("aug", cv2.WINDOW_NORMAL)
    cv2.imshow("aug", image)
    cv2.waitKey(TIME_PER_STEP)

    # for value in cycle(np.arange(-255, 255, VAL_PER_STEP)):
    for value in np.arange(-255, 255, VAL_PER_STEP):
        aug = iaa.AddToHueAndSaturation(value=value)
        img_aug = aug.augment_image(image)
        img_aug = ia.pad(img_aug, bottom=40)
        img_aug = ia.draw_text(img_aug, x=0, y=img_aug.shape[0]-38, text="value=%d" % (value,), size=30)

        cv2.imshow("aug", img_aug)
        cv2.waitKey(TIME_PER_STEP)

    images_aug = iaa.AddToHueAndSaturation(value=(-255, 255), per_channel=True).augment_images([image] * 64)
    ia.imshow(ia.draw_grid(images_aug))


if __name__ == "__main__":
    main()
