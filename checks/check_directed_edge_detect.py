from __future__ import print_function, division
from imgaug import augmenters as iaa
import numpy as np
from skimage import data
import cv2
from itertools import cycle

POINT_SIZE = 5
DEG_PER_STEP = 1
TIME_PER_STEP = 10

def main():
    image = data.astronaut()

    cv2.namedWindow("aug", cv2.WINDOW_NORMAL)
    cv2.imshow("aug", image)
    cv2.waitKey(TIME_PER_STEP)

    height, width = image.shape[0], image.shape[1]
    center_x = width // 2
    center_y = height // 2
    r = int(min(image.shape[0], image.shape[1]) / 3)

    for deg in cycle(np.arange(0, 360, DEG_PER_STEP)):
        rad = np.deg2rad(deg-90)
        #print(deg, rad)
        point_x = int(center_x + r * np.cos(rad))
        point_y = int(center_y + r * np.sin(rad))

        direction = deg / 360
        aug = iaa.DirectedEdgeDetect(alpha=1.0, direction=direction)
        img_aug = aug.augment_image(image)
        img_aug[point_y-POINT_SIZE:point_y+POINT_SIZE+1, point_x-POINT_SIZE:point_x+POINT_SIZE+1, :] = np.array([0, 255, 0])
        #print(point_x, point_y)

        cv2.imshow("aug", img_aug)
        cv2.waitKey(TIME_PER_STEP)

if __name__ == "__main__":
    main()
