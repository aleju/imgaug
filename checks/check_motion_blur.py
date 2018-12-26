from __future__ import print_function, division
from itertools import cycle

import numpy as np
import cv2

import imgaug as ia
from imgaug import augmenters as iaa

POINT_SIZE = 5
DEG_PER_STEP = 2
TIME_PER_STEP = 1


def main():
    image = ia.quokka(0.5)
    height, width = image.shape[0], image.shape[1]
    center_x = width // 2
    center_y = height // 2
    r = int(min(image.shape[0], image.shape[1]) / 3)

    cv2.namedWindow("aug", cv2.WINDOW_NORMAL)
    cv2.imshow("aug", image[:, :, ::-1])
    cv2.waitKey(TIME_PER_STEP)

    for angle in cycle(np.arange(0, 360, DEG_PER_STEP)):
        rad = np.deg2rad(angle-90)
        point_x = int(center_x + r * np.cos(rad))
        point_y = int(center_y + r * np.sin(rad))

        aug = iaa.MotionBlur(k=35, angle=angle, direction=-1.0)
        img_aug = aug.augment_image(image)
        img_aug[
            point_y-POINT_SIZE:point_y+POINT_SIZE+1,
            point_x-POINT_SIZE:point_x+POINT_SIZE+1,
            :] = np.array([0, 255, 0])

        aug_inv = iaa.MotionBlur(k=35, angle=angle, direction=1.0)
        img_aug_inv = aug_inv.augment_image(image)
        img_aug_inv[
            point_y - POINT_SIZE:point_y + POINT_SIZE + 1,
            point_x - POINT_SIZE:point_x + POINT_SIZE + 1,
            :] = np.array([0, 255, 0])

        cv2.imshow("aug", np.hstack([img_aug[:, :, ::-1], img_aug_inv[:, :, ::-1]]))
        cv2.waitKey(TIME_PER_STEP)


if __name__ == "__main__":
    main()
