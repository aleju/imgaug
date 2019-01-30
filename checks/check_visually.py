"""
Tests to visually inspect the results of the library's functionality.
Run checks via
    python check_visually.py
"""
from __future__ import print_function, division

import argparse

import numpy as np
from skimage import data

import imgaug as ia
from imgaug import augmenters as iaa


def main():
    parser = argparse.ArgumentParser(description="Check augmenters visually.")
    parser.add_argument(
        "--only", default=None,
        help="If this is set, then only the results of an augmenter with this name will be shown. "
             "Optionally, comma-separated list.",
        required=False)
    args = parser.parse_args()

    images = [
        ia.quokka_square(size=(128, 128)),
        ia.imresize_single_image(data.astronaut(), (128, 128))
    ]

    keypoints = [
        ia.KeypointsOnImage([
                ia.Keypoint(x=50, y=40),
                ia.Keypoint(x=70, y=38),
                ia.Keypoint(x=62, y=52)
            ],
            shape=images[0].shape
        ),
        ia.KeypointsOnImage([
                ia.Keypoint(x=55, y=32),
                ia.Keypoint(x=42, y=95),
                ia.Keypoint(x=75, y=89)
            ],
            shape=images[1].shape
        )
    ]

    bounding_boxes = [
        ia.BoundingBoxesOnImage([
                ia.BoundingBox(x1=10, y1=10, x2=20, y2=20),
                ia.BoundingBox(x1=40, y1=50, x2=70, y2=60)
            ],
            shape=images[0].shape
        ),
        ia.BoundingBoxesOnImage([
                ia.BoundingBox(x1=10, y1=10, x2=20, y2=20),
                ia.BoundingBox(x1=40, y1=50, x2=70, y2=60)
            ],
            shape=images[1].shape
        )
    ]

    augmenters = [
        iaa.Sequential([
            iaa.CoarseDropout(p=0.5, size_percent=0.05),
            iaa.AdditiveGaussianNoise(scale=0.1*255),
            iaa.Crop(percent=0.1)
        ], name="Sequential"),
        iaa.SomeOf(2, children=[
            iaa.CoarseDropout(p=0.5, size_percent=0.05),
            iaa.AdditiveGaussianNoise(scale=0.1*255),
            iaa.Crop(percent=0.1)
        ], name="SomeOf"),
        iaa.OneOf(children=[
            iaa.CoarseDropout(p=0.5, size_percent=0.05),
            iaa.AdditiveGaussianNoise(scale=0.1*255),
            iaa.Crop(percent=0.1)
        ], name="OneOf"),
        iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=0.1*255), name="Sometimes"),
        iaa.WithColorspace("HSV", children=[iaa.Add(20)], name="WithColorspace"),
        iaa.WithChannels([0], children=[iaa.Add(20)], name="WithChannels"),
        iaa.AddToHueAndSaturation((-20, 20), per_channel=True, name="AddToHueAndSaturation"),
        iaa.Noop(name="Noop"),
        iaa.Resize({"width": 64, "height": 64}, name="Resize"),
        iaa.CropAndPad(px=(-8, 8), name="CropAndPad-px"),
        iaa.Pad(px=(0, 8), name="Pad-px"),
        iaa.Crop(px=(0, 8), name="Crop-px"),
        iaa.Crop(percent=(0, 0.1), name="Crop-percent"),
        iaa.Fliplr(0.5, name="Fliplr"),
        iaa.Flipud(0.5, name="Flipud"),
        iaa.Superpixels(p_replace=0.75, n_segments=50, name="Superpixels"),
        iaa.Grayscale(0.5, name="Grayscale0.5"),
        iaa.Grayscale(1.0, name="Grayscale1.0"),
        iaa.GaussianBlur((0, 3.0), name="GaussianBlur"),
        iaa.AverageBlur(k=(3, 11), name="AverageBlur"),
        iaa.MedianBlur(k=(3, 11), name="MedianBlur"),
        iaa.BilateralBlur(d=10, name="BilateralBlur"),
        iaa.Sharpen(alpha=(0.1, 1.0), lightness=(0, 2.0), name="Sharpen"),
        iaa.Emboss(alpha=(0.1, 1.0), strength=(0, 2.0), name="Emboss"),
        iaa.EdgeDetect(alpha=(0.1, 1.0), name="EdgeDetect"),
        iaa.DirectedEdgeDetect(alpha=(0.1, 1.0), direction=(0, 1.0), name="DirectedEdgeDetect"),
        iaa.Add((-50, 50), name="Add"),
        iaa.Add((-50, 50), per_channel=True, name="AddPerChannel"),
        iaa.AddElementwise((-50, 50), name="AddElementwise"),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1*255), name="AdditiveGaussianNoise"),
        iaa.Multiply((0.5, 1.5), name="Multiply"),
        iaa.Multiply((0.5, 1.5), per_channel=True, name="MultiplyPerChannel"),
        iaa.MultiplyElementwise((0.5, 1.5), name="MultiplyElementwise"),
        iaa.Dropout((0.0, 0.1), name="Dropout"),
        iaa.CoarseDropout(p=0.05, size_percent=(0.05, 0.5), name="CoarseDropout"),
        iaa.Invert(p=0.5, name="Invert"),
        iaa.Invert(p=0.5, per_channel=True, name="InvertPerChannel"),
        iaa.ContrastNormalization(alpha=(0.5, 2.0), name="ContrastNormalization"),
        iaa.SaltAndPepper(p=0.05, name="SaltAndPepper"),
        iaa.Salt(p=0.05, name="Salt"),
        iaa.Pepper(p=0.05, name="Pepper"),
        iaa.CoarseSaltAndPepper(p=0.05, size_percent=(0.01, 0.1), name="CoarseSaltAndPepper"),
        iaa.CoarseSalt(p=0.05, size_percent=(0.01, 0.1), name="CoarseSalt"),
        iaa.CoarsePepper(p=0.05, size_percent=(0.01, 0.1), name="CoarsePepper"),
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_px={"x": (-16, 16), "y": (-16, 16)},
            rotate=(-45, 45),
            shear=(-16, 16),
            order=ia.ALL,
            cval=(0, 255),
            mode=ia.ALL,
            name="Affine"
        ),
        iaa.PiecewiseAffine(scale=0.03, nb_rows=(2, 6), nb_cols=(2, 6), name="PiecewiseAffine"),
        iaa.PerspectiveTransform(scale=0.1, name="PerspectiveTransform"),
        iaa.ElasticTransformation(alpha=(0.5, 8.0), sigma=1.0, name="ElasticTransformation"),
        iaa.Alpha(
            factor=(0.0, 1.0),
            first=iaa.Add(100),
            second=iaa.Dropout(0.5),
            per_channel=False,
            name="Alpha"
        ),
        iaa.Alpha(
            factor=(0.0, 1.0),
            first=iaa.Add(100),
            second=iaa.Dropout(0.5),
            per_channel=True,
            name="AlphaPerChannel"
        ),
        iaa.Alpha(
            factor=(0.0, 1.0),
            first=iaa.Affine(rotate=(-45, 45)),
            per_channel=True,
            name="AlphaAffine"
        ),
        iaa.AlphaElementwise(
            factor=(0.0, 1.0),
            first=iaa.Add(50),
            second=iaa.ContrastNormalization(2.0),
            per_channel=False,
            name="AlphaElementwise"
        ),
        iaa.AlphaElementwise(
            factor=(0.0, 1.0),
            first=iaa.Add(50),
            second=iaa.ContrastNormalization(2.0),
            per_channel=True,
            name="AlphaElementwisePerChannel"
        ),
        iaa.AlphaElementwise(
            factor=(0.0, 1.0),
            first=iaa.Affine(rotate=(-45, 45)),
            per_channel=True,
            name="AlphaElementwiseAffine"
        ),
        iaa.SimplexNoiseAlpha(
            first=iaa.EdgeDetect(1.0),
            per_channel=False,
            name="SimplexNoiseAlpha"
        ),
        iaa.FrequencyNoiseAlpha(
            first=iaa.EdgeDetect(1.0),
            per_channel=False,
            name="FrequencyNoiseAlpha"
        )
    ]

    augmenters.append(iaa.Sequential([iaa.Sometimes(0.2, aug.copy()) for aug in augmenters], name="Sequential"))
    augmenters.append(iaa.Sometimes(0.5, [aug.copy() for aug in augmenters], name="Sometimes"))

    for augmenter in augmenters:
        if args.only is None or augmenter.name in [v.strip() for v in args.only.split(",")]:
            print("Augmenter: %s" % (augmenter.name,))
            grid = []
            for image, kps, bbs in zip(images, keypoints, bounding_boxes):
                aug_det = augmenter.to_deterministic()
                imgs_aug = aug_det.augment_images(np.tile(image[np.newaxis, ...], (16, 1, 1, 1)))
                kps_aug = aug_det.augment_keypoints([kps] * 16)
                bbs_aug = aug_det.augment_bounding_boxes([bbs] * 16)
                imgs_aug_drawn = [kps_aug_one.draw_on_image(img_aug) for img_aug, kps_aug_one in zip(imgs_aug, kps_aug)]
                imgs_aug_drawn = [bbs_aug_one.draw_on_image(img_aug) for img_aug, bbs_aug_one in zip(imgs_aug_drawn, bbs_aug)]
                grid.append(np.hstack(imgs_aug_drawn))
            ia.imshow(np.vstack(grid))


if __name__ == "__main__":
    main()
