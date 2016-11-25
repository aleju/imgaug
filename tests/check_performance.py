"""
Tests to measure the performance of each augmenter.
Run these checks from the project directory (i.e. parent directory) via
    python -m tests/check_performance
"""
from __future__ import print_function, division
import imgaug as ia
import augmenters as iaa
#import parameters as iap
import numpy as np
#from scipy import ndimage, misc
#from skimage import data
import time

"""
[Augmenter: Noop]
(16, 4, 4, 3)        | SUM 0.02978s | PER ITER avg 0.00003s, min 0.00002s, max 0.00015s
(16, 32, 32, 3)      | SUM 0.03937s | PER ITER avg 0.00004s, min 0.00003s, max 0.00008s
(16, 256, 256, 3)    | SUM 0.64828s | PER ITER avg 0.00065s, min 0.00047s, max 0.00396s
[Augmenter: Crop-px]
(16, 4, 4, 3)        | SUM 4.15772s | PER ITER avg 0.00416s, min 0.00237s, max 0.00962s
(16, 32, 32, 3)      | SUM 3.95888s | PER ITER avg 0.00396s, min 0.00247s, max 0.00908s
(16, 256, 256, 3)    | SUM 13.43704s | PER ITER avg 0.01344s, min 0.00822s, max 0.02683s
[Augmenter: Crop-percent]
(16, 4, 4, 3)        | SUM 2.78791s | PER ITER avg 0.00279s, min 0.00186s, max 0.00714s
(16, 32, 32, 3)      | SUM 3.94477s | PER ITER avg 0.00394s, min 0.00218s, max 0.01315s
(16, 256, 256, 3)    | SUM 12.71237s | PER ITER avg 0.01271s, min 0.00812s, max 0.02040s
[Augmenter: Fliplr]
(16, 4, 4, 3)        | SUM 0.08755s | PER ITER avg 0.00009s, min 0.00004s, max 0.00021s
(16, 32, 32, 3)      | SUM 0.17233s | PER ITER avg 0.00017s, min 0.00008s, max 0.00055s
(16, 256, 256, 3)    | SUM 4.50344s | PER ITER avg 0.00450s, min 0.00178s, max 0.01106s
[Augmenter: Flipud]
(16, 4, 4, 3)        | SUM 0.10705s | PER ITER avg 0.00011s, min 0.00008s, max 0.00026s
(16, 32, 32, 3)      | SUM 0.11831s | PER ITER avg 0.00012s, min 0.00006s, max 0.00021s
(16, 256, 256, 3)    | SUM 1.57488s | PER ITER avg 0.00157s, min 0.00084s, max 0.00454s
[Augmenter: GaussianBlur]
(16, 4, 4, 3)        | SUM 2.31715s | PER ITER avg 0.00232s, min 0.00139s, max 0.00571s
(16, 32, 32, 3)      | SUM 4.48496s | PER ITER avg 0.00448s, min 0.00220s, max 0.00788s
(16, 256, 256, 3)    | SUM 71.18983s | PER ITER avg 0.07119s, min 0.04511s, max 0.11663s
[Augmenter: AdditiveGaussianNoise]
(16, 4, 4, 3)        | SUM 1.70817s | PER ITER avg 0.00171s, min 0.00090s, max 0.00715s
(16, 32, 32, 3)      | SUM 3.05533s | PER ITER avg 0.00306s, min 0.00205s, max 0.00712s
(16, 256, 256, 3)    | SUM 93.89743s | PER ITER avg 0.09390s, min 0.07205s, max 0.12113s
[Augmenter: Dropout]
(16, 4, 4, 3)        | SUM 1.50206s | PER ITER avg 0.00150s, min 0.00086s, max 0.00316s
(16, 32, 32, 3)      | SUM 2.45660s | PER ITER avg 0.00246s, min 0.00165s, max 0.01205s
(16, 256, 256, 3)    | SUM 66.59406s | PER ITER avg 0.06659s, min 0.04842s, max 0.09186s
[Augmenter: Multiply]
(16, 4, 4, 3)        | SUM 0.87121s | PER ITER avg 0.00087s, min 0.00046s, max 0.00211s
(16, 32, 32, 3)      | SUM 1.20261s | PER ITER avg 0.00120s, min 0.00061s, max 0.00456s
(16, 256, 256, 3)    | SUM 11.87721s | PER ITER avg 0.01188s, min 0.00888s, max 0.02220s
[Augmenter: UnnamedContrastNormalization]
(16, 4, 4, 3)        | SUM 0.94920s | PER ITER avg 0.00095s, min 0.00052s, max 0.00225s
(16, 32, 32, 3)      | SUM 1.37388s | PER ITER avg 0.00137s, min 0.00074s, max 0.00281s
(16, 256, 256, 3)    | SUM 19.87741s | PER ITER avg 0.01988s, min 0.01431s, max 0.03306s
[Augmenter: Affine]
(16, 4, 4, 3)        | SUM 8.62488s | PER ITER avg 0.00862s, min 0.00456s, max 0.01680s
(16, 32, 32, 3)      | SUM 48.71082s | PER ITER avg 0.04871s, min 0.01350s, max 0.11260s
(16, 256, 256, 3)    | SUM 2727.93633s | PER ITER avg 2.72794s, min 0.60814s, max 5.74334s
[Augmenter: ElasticTransformation]
(16, 4, 4, 3)        | SUM 4.51120s | PER ITER avg 0.00451s, min 0.00283s, max 0.00762s
(16, 32, 32, 3)      | SUM 10.91999s | PER ITER avg 0.01092s, min 0.00670s, max 0.02178s
(16, 256, 256, 3)    | SUM 368.11991s | PER ITER avg 0.36812s, min 0.26031s, max 0.49898s
"""
def main():
    augmenters = [
        iaa.Noop(name="Noop"),
        iaa.Crop(px=(0, 8), name="Crop-px"),
        iaa.Crop(percent=(0, 0.1), name="Crop-percent"),
        iaa.Fliplr(0.5, name="Fliplr"),
        iaa.Flipud(0.5, name="Flipud"),
        iaa.GaussianBlur((0, 3.0), name="GaussianBlur"),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1), name="AdditiveGaussianNoise"),
        iaa.Dropout((0.0, 0.1), name="Dropout"),
        iaa.Multiply((0.5, 1.5), name="Multiply"),
        iaa.ContrastNormalization(alpha=(0.5, 2.0)),
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_px={"x": (-16, 16), "y": (-16, 16)},
            rotate=(-45, 45),
            shear=(-16, 16),
            order=ia.ALL,
            cval=(0, 1.0),
            mode=ia.ALL,
            name="Affine"
        ),
        iaa.ElasticTransformation(alpha=(0.5, 8.0), sigma=1.0, name="ElasticTransformation")
    ]

    small_images = np.random.randint(0, 255, (16, 4, 4, 3)).astype(np.uint8)
    medium_images = np.random.randint(0, 255, (16, 32, 32, 3)).astype(np.uint8)
    large_images = np.random.randint(0, 255, (16, 256, 256, 3)).astype(np.uint8)

    for augmenter in augmenters:
        print("[Augmenter: %s]" % (augmenter.name,))
        for images in [small_images, medium_images, large_images]:
            times = []
            for i in range(1000):
                time_start = time.time()
                img_aug = augmenter.augment_images(images)
                time_end = time.time()
                times.append(time_end - time_start)
            times = np.array(times)
            img_str = "{:20s}".format(images.shape)
            print("%s | SUM %.5fs | PER ITER avg %.5fs, min %.5fs, max %.5fs" % (img_str, np.sum(times), np.average(times), np.min(times), np.max(times)))

if __name__ == "__main__":
    main()
