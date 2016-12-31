"""
Tests to measure the performance of each augmenter.
Run these checks from the project directory (i.e. parent directory) via
    python check_performance.py
"""
from __future__ import print_function, division

#import sys
#import os
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
#from scipy import ndimage, misc
#from skimage import data
import time
import random
import six
import six.moves as sm

"""
---------------------------
Keypoints
---------------------------
[Augmenter: Noop]
(4, 4, 3)            | SUM 0.01990s | PER ITER avg 0.00020s, min 0.00017s, max 0.00043s
(32, 32, 3)          | SUM 0.01863s | PER ITER avg 0.00019s, min 0.00017s, max 0.00033s
(256, 256, 3)        | SUM 0.01879s | PER ITER avg 0.00019s, min 0.00017s, max 0.00029s
[Augmenter: Crop-px]
(4, 4, 3)            | SUM 0.20215s | PER ITER avg 0.00202s, min 0.00168s, max 0.01908s
(32, 32, 3)          | SUM 0.19844s | PER ITER avg 0.00198s, min 0.00164s, max 0.01933s
(256, 256, 3)        | SUM 0.17918s | PER ITER avg 0.00179s, min 0.00166s, max 0.00214s
[Augmenter: Crop-percent]
(4, 4, 3)            | SUM 0.14201s | PER ITER avg 0.00142s, min 0.00114s, max 0.02041s
(32, 32, 3)          | SUM 0.16912s | PER ITER avg 0.00169s, min 0.00137s, max 0.02023s
(256, 256, 3)        | SUM 0.15548s | PER ITER avg 0.00155s, min 0.00142s, max 0.00193s
[Augmenter: Fliplr]
(4, 4, 3)            | SUM 0.02303s | PER ITER avg 0.00023s, min 0.00021s, max 0.00034s
(32, 32, 3)          | SUM 0.02477s | PER ITER avg 0.00025s, min 0.00021s, max 0.00038s
(256, 256, 3)        | SUM 0.02383s | PER ITER avg 0.00024s, min 0.00022s, max 0.00036s
[Augmenter: Flipud]
(4, 4, 3)            | SUM 0.02362s | PER ITER avg 0.00024s, min 0.00021s, max 0.00035s
(32, 32, 3)          | SUM 0.02356s | PER ITER avg 0.00024s, min 0.00021s, max 0.00032s
(256, 256, 3)        | SUM 0.02415s | PER ITER avg 0.00024s, min 0.00021s, max 0.00037s
[Augmenter: Grayscale]
(4, 4, 3)            | SUM 0.01908s | PER ITER avg 0.00019s, min 0.00017s, max 0.00030s
(32, 32, 3)          | SUM 0.01903s | PER ITER avg 0.00019s, min 0.00017s, max 0.00030s
(256, 256, 3)        | SUM 0.01876s | PER ITER avg 0.00019s, min 0.00017s, max 0.00027s
[Augmenter: GaussianBlur]
(4, 4, 3)            | SUM 0.01904s | PER ITER avg 0.00019s, min 0.00017s, max 0.00029s
(32, 32, 3)          | SUM 0.01851s | PER ITER avg 0.00019s, min 0.00017s, max 0.00033s
(256, 256, 3)        | SUM 0.01894s | PER ITER avg 0.00019s, min 0.00017s, max 0.00025s
[Augmenter: AdditiveGaussianNoise]
(4, 4, 3)            | SUM 0.01902s | PER ITER avg 0.00019s, min 0.00017s, max 0.00029s
(32, 32, 3)          | SUM 0.01905s | PER ITER avg 0.00019s, min 0.00017s, max 0.00028s
(256, 256, 3)        | SUM 0.01971s | PER ITER avg 0.00020s, min 0.00017s, max 0.00046s
[Augmenter: Dropout]
(4, 4, 3)            | SUM 0.01887s | PER ITER avg 0.00019s, min 0.00017s, max 0.00027s
(32, 32, 3)          | SUM 0.01913s | PER ITER avg 0.00019s, min 0.00017s, max 0.00030s
(256, 256, 3)        | SUM 0.01922s | PER ITER avg 0.00019s, min 0.00017s, max 0.00029s
[Augmenter: Multiply]
(4, 4, 3)            | SUM 0.01942s | PER ITER avg 0.00019s, min 0.00017s, max 0.00028s
(32, 32, 3)          | SUM 0.01922s | PER ITER avg 0.00019s, min 0.00017s, max 0.00032s
(256, 256, 3)        | SUM 0.01875s | PER ITER avg 0.00019s, min 0.00017s, max 0.00030s
[Augmenter: ContrastNormalization]
(4, 4, 3)            | SUM 0.01852s | PER ITER avg 0.00019s, min 0.00017s, max 0.00029s
(32, 32, 3)          | SUM 0.01869s | PER ITER avg 0.00019s, min 0.00017s, max 0.00026s
(256, 256, 3)        | SUM 0.01875s | PER ITER avg 0.00019s, min 0.00017s, max 0.00028s
[Augmenter: Grayscale]
(4, 4, 3)            | SUM 0.01919s | PER ITER avg 0.00019s, min 0.00017s, max 0.00030s
(32, 32, 3)          | SUM 0.01923s | PER ITER avg 0.00019s, min 0.00017s, max 0.00033s
(256, 256, 3)        | SUM 0.01888s | PER ITER avg 0.00019s, min 0.00017s, max 0.00028s
[Augmenter: ElasticTransformation]
(4, 4, 3)            | SUM 0.01882s | PER ITER avg 0.00019s, min 0.00017s, max 0.00024s
(32, 32, 3)          | SUM 0.01883s | PER ITER avg 0.00019s, min 0.00017s, max 0.00029s
(256, 256, 3)        | SUM 0.01869s | PER ITER avg 0.00019s, min 0.00017s, max 0.00029s
[Augmenter: AffineOrder0ModeConstant]
(4, 4, 3)            | SUM 0.28146s | PER ITER avg 0.00281s, min 0.00243s, max 0.02199s
(32, 32, 3)          | SUM 0.28047s | PER ITER avg 0.00280s, min 0.00243s, max 0.02083s
(256, 256, 3)        | SUM 0.28715s | PER ITER avg 0.00287s, min 0.00243s, max 0.02088s
[Augmenter: AffineOrder0]
(4, 4, 3)            | SUM 0.27242s | PER ITER avg 0.00272s, min 0.00246s, max 0.00362s
(32, 32, 3)          | SUM 0.29675s | PER ITER avg 0.00297s, min 0.00247s, max 0.02220s
(256, 256, 3)        | SUM 0.28988s | PER ITER avg 0.00290s, min 0.00247s, max 0.02128s
[Augmenter: AffineOrder1]
(4, 4, 3)            | SUM 0.26750s | PER ITER avg 0.00267s, min 0.00246s, max 0.00321s
(32, 32, 3)          | SUM 0.28361s | PER ITER avg 0.00284s, min 0.00245s, max 0.02144s
(256, 256, 3)        | SUM 0.28973s | PER ITER avg 0.00290s, min 0.00246s, max 0.02070s
[Augmenter: AffineAll]
(4, 4, 3)            | SUM 0.27070s | PER ITER avg 0.00271s, min 0.00246s, max 0.00367s
(32, 32, 3)          | SUM 0.28405s | PER ITER avg 0.00284s, min 0.00247s, max 0.02120s
(256, 256, 3)        | SUM 0.28895s | PER ITER avg 0.00289s, min 0.00247s, max 0.02144s
---------------------------
Images
---------------------------
[Augmenter: Noop]
(16, 4, 4, 3)        | SUM 0.00135s | PER ITER avg 0.00001s, min 0.00001s, max 0.00008s
(16, 32, 32, 3)      | SUM 0.00203s | PER ITER avg 0.00002s, min 0.00002s, max 0.00005s
(16, 256, 256, 3)    | SUM 0.05284s | PER ITER avg 0.00053s, min 0.00044s, max 0.00194s
[Augmenter: Crop-px]
(16, 4, 4, 3)        | SUM 0.09324s | PER ITER avg 0.00093s, min 0.00084s, max 0.00315s
(16, 32, 32, 3)      | SUM 0.10302s | PER ITER avg 0.00103s, min 0.00094s, max 0.00162s
(16, 256, 256, 3)    | SUM 0.81943s | PER ITER avg 0.00819s, min 0.00767s, max 0.00934s
[Augmenter: Crop-percent]
(16, 4, 4, 3)        | SUM 0.06562s | PER ITER avg 0.00066s, min 0.00057s, max 0.00099s
(16, 32, 32, 3)      | SUM 0.09784s | PER ITER avg 0.00098s, min 0.00089s, max 0.00131s
(16, 256, 256, 3)    | SUM 0.80779s | PER ITER avg 0.00808s, min 0.00732s, max 0.01008s
[Augmenter: Fliplr]
(16, 4, 4, 3)        | SUM 0.00525s | PER ITER avg 0.00005s, min 0.00004s, max 0.00017s
(16, 32, 32, 3)      | SUM 0.01025s | PER ITER avg 0.00010s, min 0.00007s, max 0.00015s
(16, 256, 256, 3)    | SUM 0.36918s | PER ITER avg 0.00369s, min 0.00181s, max 0.00553s
[Augmenter: Flipud]
(16, 4, 4, 3)        | SUM 0.00512s | PER ITER avg 0.00005s, min 0.00004s, max 0.00009s
(16, 32, 32, 3)      | SUM 0.00665s | PER ITER avg 0.00007s, min 0.00006s, max 0.00011s
(16, 256, 256, 3)    | SUM 0.12664s | PER ITER avg 0.00127s, min 0.00092s, max 0.00167s
[Augmenter: Grayscale]
(16, 4, 4, 3)        | SUM 0.05943s | PER ITER avg 0.00059s, min 0.00050s, max 0.00125s
(16, 32, 32, 3)      | SUM 0.12247s | PER ITER avg 0.00122s, min 0.00106s, max 0.00205s
(16, 256, 256, 3)    | SUM 3.62785s | PER ITER avg 0.03628s, min 0.03508s, max 0.03963s
[Augmenter: GaussianBlur]
(16, 4, 4, 3)        | SUM 0.15514s | PER ITER avg 0.00155s, min 0.00136s, max 0.00188s
(16, 32, 32, 3)      | SUM 0.25121s | PER ITER avg 0.00251s, min 0.00221s, max 0.00298s
(16, 256, 256, 3)    | SUM 5.51685s | PER ITER avg 0.05517s, min 0.04923s, max 0.06026s
[Augmenter: AdditiveGaussianNoise]
(16, 4, 4, 3)        | SUM 0.09606s | PER ITER avg 0.00096s, min 0.00085s, max 0.00150s
(16, 32, 32, 3)      | SUM 0.21302s | PER ITER avg 0.00213s, min 0.00196s, max 0.00254s
(16, 256, 256, 3)    | SUM 7.22374s | PER ITER avg 0.07224s, min 0.07017s, max 0.07558s
[Augmenter: Dropout]
(16, 4, 4, 3)        | SUM 0.09362s | PER ITER avg 0.00094s, min 0.00084s, max 0.00118s
(16, 32, 32, 3)      | SUM 0.17472s | PER ITER avg 0.00175s, min 0.00161s, max 0.00230s
(16, 256, 256, 3)    | SUM 5.04969s | PER ITER avg 0.05050s, min 0.04839s, max 0.05631s
[Augmenter: Multiply]
(16, 4, 4, 3)        | SUM 0.05442s | PER ITER avg 0.00054s, min 0.00046s, max 0.00089s
(16, 32, 32, 3)      | SUM 0.06895s | PER ITER avg 0.00069s, min 0.00060s, max 0.00109s
(16, 256, 256, 3)    | SUM 0.87311s | PER ITER avg 0.00873s, min 0.00799s, max 0.00993s
[Augmenter: ContrastNormalization]
(16, 4, 4, 3)        | SUM 0.05746s | PER ITER avg 0.00057s, min 0.00050s, max 0.00094s
(16, 32, 32, 3)      | SUM 0.08083s | PER ITER avg 0.00081s, min 0.00071s, max 0.00133s
(16, 256, 256, 3)    | SUM 1.57577s | PER ITER avg 0.01576s, min 0.01443s, max 0.01831s
[Augmenter: Grayscale]
(16, 4, 4, 3)        | SUM 0.05464s | PER ITER avg 0.00055s, min 0.00049s, max 0.00069s
(16, 32, 32, 3)      | SUM 0.12058s | PER ITER avg 0.00121s, min 0.00104s, max 0.00223s
(16, 256, 256, 3)    | SUM 3.57037s | PER ITER avg 0.03570s, min 0.03461s, max 0.03780s
[Augmenter: ElasticTransformation]
(16, 4, 4, 3)        | SUM 0.29551s | PER ITER avg 0.00296s, min 0.00272s, max 0.00336s
(16, 32, 32, 3)      | SUM 0.68591s | PER ITER avg 0.00686s, min 0.00642s, max 0.00764s
(16, 256, 256, 3)    | SUM 26.30515s | PER ITER avg 0.26305s, min 0.25754s, max 0.26912s
[Augmenter: AffineOrder0ModeConstant]
(16, 4, 4, 3)        | SUM 0.35887s | PER ITER avg 0.00359s, min 0.00333s, max 0.00424s
(16, 32, 32, 3)      | SUM 0.47889s | PER ITER avg 0.00479s, min 0.00451s, max 0.00535s
(16, 256, 256, 3)    | SUM 9.83738s | PER ITER avg 0.09837s, min 0.09417s, max 0.10458s
[Augmenter: AffineOrder0]
(16, 4, 4, 3)        | SUM 0.37980s | PER ITER avg 0.00380s, min 0.00340s, max 0.00517s
(16, 32, 32, 3)      | SUM 0.53106s | PER ITER avg 0.00531s, min 0.00472s, max 0.00630s
(16, 256, 256, 3)    | SUM 10.69961s | PER ITER avg 0.10700s, min 0.10223s, max 0.11325s
[Augmenter: AffineOrder1]
(16, 4, 4, 3)        | SUM 0.39431s | PER ITER avg 0.00394s, min 0.00363s, max 0.00511s
(16, 32, 32, 3)      | SUM 0.62730s | PER ITER avg 0.00627s, min 0.00576s, max 0.00711s
(16, 256, 256, 3)    | SUM 14.50003s | PER ITER avg 0.14500s, min 0.13785s, max 0.15291s
[Augmenter: AffineAll]
(16, 4, 4, 3)        | SUM 0.58742s | PER ITER avg 0.00587s, min 0.00429s, max 0.00724s
(16, 32, 32, 3)      | SUM 3.69956s | PER ITER avg 0.03700s, min 0.01358s, max 0.06233s
(16, 256, 256, 3)    | SUM 212.91776s | PER ITER avg 2.12918s, min 0.57114s, max 3.95389s
"""
def main():
    augmenters = [
        iaa.Noop(name="Noop"),
        iaa.Crop(px=(0, 8), name="Crop-px"),
        iaa.Crop(percent=(0, 0.1), name="Crop-percent"),
        iaa.Fliplr(0.5, name="Fliplr"),
        iaa.Flipud(0.5, name="Flipud"),
        iaa.Grayscale((0.0, 1.0), name="Grayscale"),
        iaa.GaussianBlur((0, 3.0), name="GaussianBlur"),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1), name="AdditiveGaussianNoise"),
        iaa.Dropout((0.0, 0.1), name="Dropout"),
        iaa.Multiply((0.5, 1.5), name="Multiply"),
        iaa.ContrastNormalization(alpha=(0.5, 2.0), name="ContrastNormalization"),
        iaa.Grayscale(alpha=(0.0, 1.0), name="Grayscale"),
        iaa.ElasticTransformation(alpha=(0.5, 8.0), sigma=1.0, name="ElasticTransformation"),
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_px={"x": (-16, 16), "y": (-16, 16)},
            rotate=(-45, 45),
            shear=(-16, 16),
            order=0,
            cval=(0, 1.0),
            mode="constant",
            name="AffineOrder0ModeConstant"
        )
    ]

    for order in [0, 1]:
        augmenters.append(
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_px={"x": (-16, 16), "y": (-16, 16)},
                rotate=(-45, 45),
                shear=(-16, 16),
                order=order,
                cval=(0, 1.0),
                mode=ia.ALL,
                name="AffineOrder%d" % (order,)
            )
        )

    augmenters.append(
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_px={"x": (-16, 16), "y": (-16, 16)},
            rotate=(-45, 45),
            shear=(-16, 16),
            order=ia.ALL,
            cval=(0, 1.0),
            mode=ia.ALL,
            name="AffineAll"
        )
    )

    kps = []
    for _ in sm.xrange(20):
        x = random.randint(0, 31)
        y = random.randint(0, 31)
        kps.append(ia.Keypoint(x=x, y=y))
    kps = ia.KeypointsOnImage(kps, shape=(32, 32, 3))
    #small_keypoints_one = ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=1, y=1), ia.Keypoint(x=3, y=3)], shape=(4, 4, 3))
    #medium_keypoints_one = ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=16, y=16), ia.Keypoint(x=31, y=31)], shape=(32, 32, 3))
    #large_keypoints_one = ia.KeypointsOnImage([ia.Keypoint(x=0, y=0), ia.Keypoint(x=128, y=128), ia.Keypoint(x=255, y=255)], shape=(256, 256, 3))
    small_keypoints_one = kps.on((4, 4, 3))
    medium_keypoints_one = kps.on((32, 32, 3))
    large_keypoints_one = kps.on((256, 256, 3))
    small_keypoints = [small_keypoints_one.deepcopy() for _ in sm.xrange(16)]
    medium_keypoints = [medium_keypoints_one.deepcopy() for _ in sm.xrange(16)]
    large_keypoints = [large_keypoints_one.deepcopy() for _ in sm.xrange(16)]

    small_images = np.random.randint(0, 255, (16, 4, 4, 3)).astype(np.uint8)
    medium_images = np.random.randint(0, 255, (16, 32, 32, 3)).astype(np.uint8)
    large_images = np.random.randint(0, 255, (16, 256, 256, 3)).astype(np.uint8)

    print("---------------------------")
    print("Keypoints")
    print("---------------------------")
    for augmenter in augmenters:
        print("[Augmenter: %s]" % (augmenter.name,))
        for keypoints in [small_keypoints, medium_keypoints, large_keypoints]:
            times = []
            for i in sm.xrange(100):
                time_start = time.time()
                img_aug = augmenter.augment_keypoints(keypoints)
                time_end = time.time()
                times.append(time_end - time_start)
            times = np.array(times)
            img_str = "{:20s}".format(keypoints[0].shape)
            print("%s | SUM %.5fs | PER ITER avg %.5fs, min %.5fs, max %.5fs" % (img_str, np.sum(times), np.average(times), np.min(times), np.max(times)))

    print("---------------------------")
    print("Images")
    print("---------------------------")
    for augmenter in augmenters:
        print("[Augmenter: %s]" % (augmenter.name,))
        for images in [small_images, medium_images, large_images]:
            times = []
            for i in sm.xrange(100):
                time_start = time.time()
                img_aug = augmenter.augment_images(images)
                time_end = time.time()
                times.append(time_end - time_start)
            times = np.array(times)
            img_str = "{:20s}".format(images.shape)
            print("%s | SUM %.5fs | PER ITER avg %.5fs, min %.5fs, max %.5fs" % (img_str, np.sum(times), np.average(times), np.min(times), np.max(times)))

if __name__ == "__main__":
    main()
