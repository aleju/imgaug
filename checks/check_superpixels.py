from __future__ import print_function, division

import time
from itertools import cycle

import cv2
import numpy as np
from skimage import data

import imgaug as ia
from imgaug import augmenters as iaa

POINT_SIZE = 5
SEGMENTS_PER_STEP = 1
TIME_PER_STEP = 10


def main():
    image = data.astronaut()[..., ::-1]  # rgb2bgr
    print(image.shape)

    cv2.namedWindow("aug", cv2.WINDOW_NORMAL)
    cv2.imshow("aug", image)
    cv2.waitKey(TIME_PER_STEP)

    for n_segments in cycle(reversed(np.arange(1, 200, SEGMENTS_PER_STEP))):
        aug = iaa.Superpixels(p_replace=0.75, n_segments=n_segments)
        time_start = time.time()
        img_aug = aug.augment_image(image)
        print("augmented %d in %.4fs" % (n_segments, time.time() - time_start))
        img_aug = ia.draw_text(img_aug, x=5, y=5, text="%d" % (n_segments,))

        cv2.imshow("aug", img_aug)
        cv2.waitKey(TIME_PER_STEP)


if __name__ == "__main__":
    main()
