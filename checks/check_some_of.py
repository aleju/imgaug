from __future__ import print_function, division

import cv2
import numpy as np
from skimage import data

import imgaug as ia
from imgaug import augmenters as iaa

TIME_PER_STEP = 20000
NB_AUGS_PER_IMAGE = 10


def main():
    image = data.astronaut()
    image = ia.imresize_single_image(image, (64, 64))
    keypoints_on_image = ia.KeypointsOnImage([ia.Keypoint(x=10, y=10)], shape=image.shape)
    images_arr = np.array([image for _ in range(NB_AUGS_PER_IMAGE)])
    images_list = [image for _ in range(NB_AUGS_PER_IMAGE)]
    keypoints_on_images = [keypoints_on_image.deepcopy() for _ in range(NB_AUGS_PER_IMAGE)]
    print("image shape:", image.shape)
    print("Press ENTER or wait %d ms to proceed to the next image." % (TIME_PER_STEP,))

    children = [
        iaa.CoarseDropout(p=0.5, size_percent=0.05),
        iaa.AdditiveGaussianNoise(scale=0.1*255)
    ]

    n = [
        None,
        0,
        1,
        len(children),
        len(children)+1,
        (1, 1),
        (1, len(children)),
        (1, len(children)+1),
        (1, None)
    ]

    cv2.namedWindow("aug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("aug", 64*NB_AUGS_PER_IMAGE, (2+4+4)*64)

    rows = []
    for ni in n:
        aug = iaa.SomeOf(ni, children, random_order=False)
        rows.append(aug.augment_images(images_arr))
    grid = to_grid(rows, None)
    cv2.imshow("aug", grid[..., ::-1])  # here with rgb2bgr
    cv2.waitKey(TIME_PER_STEP)

    for ni in n:
        print("------------------------")
        print("-- %s" % (str(ni),))
        print("------------------------")
        aug = iaa.SomeOf(ni, children, random_order=False)
        aug_ro = iaa.SomeOf(ni, children, random_order=True)

        aug_det = aug.to_deterministic()
        aug_ro_det = aug_ro.to_deterministic()

        aug_kps = []
        aug_kps.extend([aug_det.augment_keypoints(keypoints_on_images)] * 4)
        aug_kps.extend([aug_ro_det.augment_keypoints(keypoints_on_images)] * 4)

        aug_rows = []
        aug_rows.append(images_arr)
        aug_rows.append(images_list)

        aug_rows.append(aug_det.augment_images(images_arr))
        aug_rows.append(aug_det.augment_images(images_arr))
        aug_rows.append(aug_det.augment_images(images_list))
        aug_rows.append(aug_det.augment_images(images_list))

        aug_rows.append(aug_ro_det.augment_images(images_arr))
        aug_rows.append(aug_ro_det.augment_images(images_arr))
        aug_rows.append(aug_ro_det.augment_images(images_list))
        aug_rows.append(aug_ro_det.augment_images(images_list))

        grid = to_grid(aug_rows, aug_kps)

        title = "n=%s" % (str(ni),)
        grid = ia.draw_text(grid, x=5, y=5, text=title)

        cv2.imshow("aug", grid[..., ::-1]) # here with rgb2bgr
        cv2.waitKey(TIME_PER_STEP)


# TODO could be replaced by imgaug.draw_grid()?
def to_grid(rows, rows_kps):
    if rows_kps is None:
        rows = [np.hstack(list(row)) for row in rows]
        return np.vstack(rows)
    else:
        rows_rendered = []
        for row, row_kps in zip(rows, rows_kps):
            row_with_kps = []
            for i in range(len(row)):
                img = row[i]
                img_kps = row_kps[i].draw_on_image(img)
                row_with_kps.append(img_kps)
            rows_rendered.append(np.hstack(row_with_kps))

        return np.vstack(rows_rendered)


if __name__ == "__main__":
    main()
