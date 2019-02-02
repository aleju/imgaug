from __future__ import print_function, division, absolute_import

import time

import matplotlib
matplotlib.use('Agg')  # fix execution of tests involving matplotlib on travis
import numpy as np
import skimage
import skimage.data

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.testutils import create_random_images, array_equal_lists, keypoints_equal, reseed


def main():
    time_start = time.time()

    test_determinism()
    test_keypoint_augmentation()
    test_unusual_channel_numbers()
    test_dtype_preservation()

    time_end = time.time()
    print("<%s> Finished without errors in %.4fs." % (__file__, time_end - time_start,))


def test_determinism():
    reseed()

    images = [
        ia.quokka(size=(128, 128)),
        ia.quokka(size=(64, 64)),
        ia.imresize_single_image(skimage.data.astronaut(), (128, 256))
    ]
    images.extend([ia.quokka(size=(16, 16))] * 20)

    keypoints = [
        ia.KeypointsOnImage([
            ia.Keypoint(x=20, y=10), ia.Keypoint(x=5, y=5),
            ia.Keypoint(x=10, y=43)], shape=(50, 60, 3))
    ] * 20

    augs = [
        iaa.Sequential([iaa.Fliplr(0.5), iaa.Flipud(0.5)]),
        iaa.SomeOf(1, [iaa.Fliplr(0.5), iaa.Flipud(0.5)]),
        iaa.OneOf([iaa.Fliplr(0.5), iaa.Flipud(0.5)]),
        iaa.Sometimes(0.5, iaa.Fliplr(1.0)),
        iaa.WithColorspace("HSV", children=iaa.Add((-50, 50))),
        # iaa.WithChannels([0], iaa.Add((-50, 50))),
        # iaa.Noop(name="Noop-nochange"),
        # iaa.Lambda(
        #     func_images=lambda images, random_state, parents, hooks: images,
        #     func_keypoints=lambda keypoints_on_images, random_state, parents, hooks: keypoints_on_images,
        #     name="Lambda-nochange"
        # ),
        # iaa.AssertLambda(
        #     func_images=lambda images, random_state, parents, hooks: True,
        #     func_keypoints=lambda keypoints_on_images, random_state, parents, hooks: True,
        #     name="AssertLambda-nochange"
        # ),
        # iaa.AssertShape(
        #     (None, None, None, 3),
        #     check_keypoints=False,
        #     name="AssertShape-nochange"
        # ),
        iaa.Resize((0.5, 0.9)),
        iaa.CropAndPad(px=(-50, 50)),
        iaa.Pad(px=(1, 50)),
        iaa.Crop(px=(1, 50)),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Superpixels(p_replace=(0.25, 1.0), n_segments=(16, 128)),
        # iaa.ChangeColorspace(to_colorspace="GRAY"),
        iaa.Grayscale(alpha=(0.1, 1.0)),
        iaa.GaussianBlur((0.1, 3.0)),
        iaa.AverageBlur((3, 11)),
        iaa.MedianBlur((3, 11)),
        # iaa.Convolve(np.array([[0, 1, 0],
        #                       [1, -4, 1],
        #                       [0, 1, 0]])),
        iaa.Sharpen(alpha=(0.1, 1.0), lightness=(0.8, 1.2)),
        iaa.Emboss(alpha=(0.1, 1.0), strength=(0.8, 1.2)),
        iaa.EdgeDetect(alpha=(0.1, 1.0)),
        iaa.DirectedEdgeDetect(alpha=(0.1, 1.0), direction=(0.0, 1.0)),
        iaa.Add((-50, 50)),
        iaa.AddElementwise((-50, 50)),
        iaa.AdditiveGaussianNoise(scale=(0.1, 1.0)),
        iaa.Multiply((0.6, 1.4)),
        iaa.MultiplyElementwise((0.6, 1.4)),
        iaa.Dropout((0.3, 0.5)),
        iaa.CoarseDropout((0.3, 0.5), size_percent=(0.05, 0.2)),
        iaa.Invert(0.5),
        iaa.ContrastNormalization((0.6, 1.4)),
        iaa.Affine(scale=(0.7, 1.3), translate_percent=(-0.1, 0.1),
                   rotate=(-20, 20), shear=(-20, 20), order=ia.ALL,
                   mode=ia.ALL, cval=(0, 255)),
        iaa.PiecewiseAffine(scale=(0.1, 0.3)),
        iaa.ElasticTransformation(alpha=0.5)
    ]

    augs_affect_geometry = [
        iaa.Sequential([iaa.Fliplr(0.5), iaa.Flipud(0.5)]),
        iaa.SomeOf(1, [iaa.Fliplr(0.5), iaa.Flipud(0.5)]),
        iaa.OneOf([iaa.Fliplr(0.5), iaa.Flipud(0.5)]),
        iaa.Sometimes(0.5, iaa.Fliplr(1.0)),
        iaa.Resize((0.5, 0.9)),
        iaa.CropAndPad(px=(-50, 50)),
        iaa.Pad(px=(1, 50)),
        iaa.Crop(px=(1, 50)),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Affine(scale=(0.7, 1.3), translate_percent=(-0.1, 0.1),
                   rotate=(-20, 20), shear=(-20, 20), order=ia.ALL,
                   mode=ia.ALL, cval=(0, 255)),
        iaa.PiecewiseAffine(scale=(0.1, 0.3)),
        iaa.ElasticTransformation(alpha=(5, 100), sigma=(3, 5))
    ]

    for aug in augs:
        aug_det = aug.to_deterministic()
        images_aug1 = aug_det.augment_images(images)
        images_aug2 = aug_det.augment_images(images)

        aug_det = aug.to_deterministic()
        images_aug3 = aug_det.augment_images(images)
        images_aug4 = aug_det.augment_images(images)

        assert array_equal_lists(images_aug1, images_aug2), \
            "Images (1, 2) expected to be identical for %s" % (aug.name,)

        assert array_equal_lists(images_aug3, images_aug4), \
            "Images (3, 4) expected to be identical for %s" % (aug.name,)

        assert not array_equal_lists(images_aug1, images_aug3), \
            "Images (1, 3) expected to be different for %s" % (aug.name,)

    for aug in augs_affect_geometry:
        aug_det = aug.to_deterministic()
        kps_aug1 = aug_det.augment_keypoints(keypoints)
        kps_aug2 = aug_det.augment_keypoints(keypoints)

        aug_det = aug.to_deterministic()
        kps_aug3 = aug_det.augment_keypoints(keypoints)
        kps_aug4 = aug_det.augment_keypoints(keypoints)

        assert keypoints_equal(kps_aug1, kps_aug2), \
            "Keypoints (1, 2) expected to be identical for %s" % (aug.name,)

        assert keypoints_equal(kps_aug3, kps_aug4), \
            "Keypoints (3, 4) expected to be identical for %s" % (aug.name,)

        assert not keypoints_equal(kps_aug1, kps_aug3), \
            "Keypoints (1, 3) expected to be different for %s" % (aug.name,)


def test_keypoint_augmentation():
    reseed()

    keypoints = []
    for y in range(40//5):
        for x in range(60//5):
            keypoints.append(ia.Keypoint(y=y*5, x=x*5))

    keypoints_oi = ia.KeypointsOnImage(keypoints, shape=(40, 60, 3))
    keypoints_oi_empty = ia.KeypointsOnImage([], shape=(40, 60, 3))

    augs = [
        iaa.Add((-5, 5), name="Add"),
        iaa.AddElementwise((-5, 5), name="AddElementwise"),
        iaa.AdditiveGaussianNoise(0.01*255, name="AdditiveGaussianNoise"),
        iaa.Multiply((0.95, 1.05), name="Multiply"),
        iaa.Dropout(0.01, name="Dropout"),
        iaa.CoarseDropout(0.01, size_px=6, name="CoarseDropout"),
        iaa.Invert(0.01, per_channel=True, name="Invert"),
        iaa.ContrastNormalization((0.95, 1.05), name="ContrastNormalization"),
        iaa.GaussianBlur(sigma=(0.95, 1.05), name="GaussianBlur"),
        iaa.AverageBlur((3, 5), name="AverageBlur"),
        iaa.MedianBlur((3, 5), name="MedianBlur"),
        # iaa.BilateralBlur((3, 5), name="BilateralBlur"),
        # WithColorspace ?
        # iaa.AddToHueAndSaturation((-5, 5), name="AddToHueAndSaturation"),
        # ChangeColorspace ?
        # Grayscale cannot be tested, input not RGB
        # Convolve ?
        iaa.Sharpen((0.0, 0.1), lightness=(1.0, 1.2), name="Sharpen"),
        iaa.Emboss(alpha=(0.0, 0.1), strength=(0.5, 1.5), name="Emboss"),
        iaa.EdgeDetect(alpha=(0.0, 0.1), name="EdgeDetect"),
        iaa.DirectedEdgeDetect(alpha=(0.0, 0.1), direction=0, name="DirectedEdgeDetect"),
        iaa.Fliplr(0.5, name="Fliplr"),
        iaa.Flipud(0.5, name="Flipud"),
        iaa.Affine(translate_px=(-5, 5), name="Affine-translate-px"),
        iaa.Affine(translate_percent=(-0.05, 0.05), name="Affine-translate-percent"),
        iaa.Affine(rotate=(-20, 20), name="Affine-rotate"),
        iaa.Affine(shear=(-20, 20), name="Affine-shear"),
        iaa.Affine(scale=(0.9, 1.1), name="Affine-scale"),
        iaa.PiecewiseAffine(scale=(0.001, 0.005), name="PiecewiseAffine"),
        # iaa.PerspectiveTransform(scale=(0.01, 0.10), name="PerspectiveTransform"),
        iaa.ElasticTransformation(alpha=(0.1, 0.2), sigma=(0.1, 0.2), name="ElasticTransformation"),
        # Sequential
        # SomeOf
        # OneOf
        # Sometimes
        # WithChannels
        # Noop
        # Lambda
        # AssertLambda
        # AssertShape
        iaa.Alpha((0.0, 0.1), iaa.Add(10), name="Alpha"),
        iaa.AlphaElementwise((0.0, 0.1), iaa.Add(10), name="AlphaElementwise"),
        iaa.SimplexNoiseAlpha(iaa.Add(10), name="SimplexNoiseAlpha"),
        iaa.FrequencyNoiseAlpha(exponent=(-2, 2), first=iaa.Add(10),
                                name="SimplexNoiseAlpha"),
        iaa.Superpixels(p_replace=0.01, n_segments=64),
        iaa.Resize(0.5, name="Resize"),
        iaa.CropAndPad(px=(-10, 10), name="CropAndPad"),
        iaa.Pad(px=(0, 10), name="Pad"),
        iaa.Crop(px=(0, 10), name="Crop")
    ]

    for aug in augs:
        dss = []
        for i in range(10):
            aug_det = aug.to_deterministic()

            kp_fully_empty_aug = aug_det.augment_keypoints([])
            assert kp_fully_empty_aug == []

            kp_first_empty_aug = aug_det.augment_keypoints([keypoints_oi_empty])[0]
            assert len(kp_first_empty_aug.keypoints) == 0

            kp_image = keypoints_oi.to_keypoint_image(size=5)
            kp_image_aug = aug_det.augment_image(kp_image)
            kp_image_aug_rev = ia.KeypointsOnImage.from_keypoint_image(
                kp_image_aug,
                if_not_found_coords={"x": -9999, "y": -9999},
                nb_channels=1
            )
            kp_aug = aug_det.augment_keypoints([keypoints_oi])[0]
            ds = []
            assert len(kp_image_aug_rev.keypoints) == len(kp_aug.keypoints),\
                "Lost keypoints for '%s' (%d vs expected %d)" \
                % (aug.name, len(kp_aug.keypoints), len(kp_image_aug_rev.keypoints))
            for kp_pred, kp_pred_img in zip(kp_aug.keypoints, kp_image_aug_rev.keypoints):
                kp_pred_lost = (kp_pred.x == -9999 and kp_pred.y == -9999)
                kp_pred_img_lost = (kp_pred_img.x == -9999 and kp_pred_img.y == -9999)

                if not kp_pred_lost and not kp_pred_img_lost:
                    d = np.sqrt((kp_pred.x - kp_pred_img.x) ** 2
                                + (kp_pred.y - kp_pred_img.y) ** 2)
                    ds.append(d)
            dss.extend(ds)
            if len(ds) == 0:
                print("[INFO] No valid keypoints found for '%s' "
                      "in test_keypoint_augmentation()" % (str(aug),))
        assert np.average(dss) < 5.0, \
            "Average distance too high (%.2f, with ds: %s)" \
            % (np.average(dss), str(dss))


def test_unusual_channel_numbers():
    reseed()

    images = [
        (0, create_random_images((4, 16, 16))),
        (1, create_random_images((4, 16, 16, 1))),
        (2, create_random_images((4, 16, 16, 2))),
        (4, create_random_images((4, 16, 16, 4))),
        (5, create_random_images((4, 16, 16, 5))),
        (10, create_random_images((4, 16, 16, 10))),
        (20, create_random_images((4, 16, 16, 20)))
    ]

    augs = [
        iaa.Add((-5, 5), name="Add"),
        iaa.AddElementwise((-5, 5), name="AddElementwise"),
        iaa.AdditiveGaussianNoise(0.01*255, name="AdditiveGaussianNoise"),
        iaa.Multiply((0.95, 1.05), name="Multiply"),
        iaa.Dropout(0.01, name="Dropout"),
        iaa.CoarseDropout(0.01, size_px=6, name="CoarseDropout"),
        iaa.Invert(0.01, per_channel=True, name="Invert"),
        iaa.ContrastNormalization((0.95, 1.05), name="ContrastNormalization"),
        iaa.GaussianBlur(sigma=(0.95, 1.05), name="GaussianBlur"),
        iaa.AverageBlur((3, 5), name="AverageBlur"),
        iaa.MedianBlur((3, 5), name="MedianBlur"),
        # iaa.BilateralBlur((3, 5), name="BilateralBlur"), # works only with 3/RGB channels
        # WithColorspace ?
        # iaa.AddToHueAndSaturation((-5, 5), name="AddToHueAndSaturation"), # works only with 3/RGB channels
        # ChangeColorspace ?
        # iaa.Grayscale((0.0, 0.1), name="Grayscale"), # works only with 3 channels
        # Convolve ?
        iaa.Sharpen((0.0, 0.1), lightness=(1.0, 1.2), name="Sharpen"),
        iaa.Emboss(alpha=(0.0, 0.1), strength=(0.5, 1.5), name="Emboss"),
        iaa.EdgeDetect(alpha=(0.0, 0.1), name="EdgeDetect"),
        iaa.DirectedEdgeDetect(alpha=(0.0, 0.1), direction=0,
                               name="DirectedEdgeDetect"),
        iaa.Fliplr(0.5, name="Fliplr"),
        iaa.Flipud(0.5, name="Flipud"),
        iaa.Affine(translate_px=(-5, 5), name="Affine-translate-px"),
        iaa.Affine(translate_percent=(-0.05, 0.05), name="Affine-translate-percent"),
        iaa.Affine(rotate=(-20, 20), name="Affine-rotate"),
        iaa.Affine(shear=(-20, 20), name="Affine-shear"),
        iaa.Affine(scale=(0.9, 1.1), name="Affine-scale"),
        iaa.PiecewiseAffine(scale=(0.001, 0.005), name="PiecewiseAffine"),
        iaa.PerspectiveTransform(scale=(0.01, 0.10), name="PerspectiveTransform"),
        iaa.ElasticTransformation(alpha=(0.1, 0.2), sigma=(0.1, 0.2),
                                  name="ElasticTransformation"),
        iaa.Sequential([iaa.Add((-5, 5)), iaa.AddElementwise((-5, 5))]),
        iaa.SomeOf(1, [iaa.Add((-5, 5)), iaa.AddElementwise((-5, 5))]),
        iaa.OneOf([iaa.Add((-5, 5)), iaa.AddElementwise((-5, 5))]),
        iaa.Sometimes(0.5, iaa.Add((-5, 5)), name="Sometimes"),
        # WithChannels
        iaa.Noop(name="Noop"),
        # Lambda
        # AssertLambda
        # AssertShape
        iaa.Alpha((0.0, 0.1), iaa.Add(10), name="Alpha"),
        iaa.AlphaElementwise((0.0, 0.1), iaa.Add(10), name="AlphaElementwise"),
        iaa.SimplexNoiseAlpha(iaa.Add(10), name="SimplexNoiseAlpha"),
        iaa.FrequencyNoiseAlpha(exponent=(-2, 2), first=iaa.Add(10),
                                name="SimplexNoiseAlpha"),
        iaa.Superpixels(p_replace=0.01, n_segments=64),
        iaa.Resize({"height": 4, "width": 4}, name="Resize"),
        iaa.CropAndPad(px=(-10, 10), name="CropAndPad"),
        iaa.Pad(px=(0, 10), name="Pad"),
        iaa.Crop(px=(0, 10), name="Crop")
    ]

    for aug in augs:
        for (nb_channels, images_c) in images:
            if aug.name != "Resize":
                images_aug = aug.augment_images(images_c)
                assert images_aug.shape == images_c.shape
                image_aug = aug.augment_image(images_c[0])
                assert image_aug.shape == images_c[0].shape
            else:
                images_aug = aug.augment_images(images_c)
                image_aug = aug.augment_image(images_c[0])
                if images_c.ndim == 3:
                    assert images_aug.shape == (4, 4, 4)
                    assert image_aug.shape == (4, 4)
                else:
                    assert images_aug.shape == (4, 4, 4, images_c.shape[3])
                    assert image_aug.shape == (4, 4, images_c.shape[3])


def test_dtype_preservation():
    reseed()

    size = (4, 16, 16, 3)
    images = [
        np.random.uniform(0, 255, size).astype(np.uint8),
        np.random.uniform(0, 65535, size).astype(np.uint16),
        np.random.uniform(0, 4294967295, size).astype(np.uint32),
        np.random.uniform(-128, 127, size).astype(np.int16),
        np.random.uniform(-32768, 32767, size).astype(np.int32),
        np.random.uniform(0.0, 1.0, size).astype(np.float32),
        np.random.uniform(-1000.0, 1000.0, size).astype(np.float16),
        np.random.uniform(-1000.0, 1000.0, size).astype(np.float32),
        np.random.uniform(-1000.0, 1000.0, size).astype(np.float64)
    ]

    default_dtypes = set([arr.dtype for arr in images])
    # Some dtypes are here removed per augmenter, because the respective
    # augmenter does not support them. This test currently only checks whether
    # dtypes are preserved from in- to output for all dtypes that are supported
    # per augmenter.
    # dtypes are here removed via list comprehension instead of
    # `default_dtypes - set([dtype])`, because the latter one simply never
    # removed the dtype(s) for some reason?!

    def _not_dts(dts):
        return [dt for dt in default_dtypes if dt not in dts]

    augs = [
        (iaa.Add((-5, 5), name="Add"), _not_dts([np.uint32, np.int32, np.float64])),
        (iaa.AddElementwise((-5, 5), name="AddElementwise"), _not_dts([np.uint32, np.int32, np.float64])),
        (iaa.AdditiveGaussianNoise(0.01*255, name="AdditiveGaussianNoise"), _not_dts([np.uint32, np.int32, np.float64])),
        (iaa.Multiply((0.95, 1.05), name="Multiply"), _not_dts([np.uint32, np.int32, np.float64])),
        (iaa.Dropout(0.01, name="Dropout"), _not_dts([np.uint32, np.int32, np.float64])),
        (iaa.CoarseDropout(0.01, size_px=6, name="CoarseDropout"), _not_dts([np.uint32, np.int32, np.float64])),
        (iaa.Invert(0.01, per_channel=True, name="Invert"), default_dtypes),
        (iaa.ContrastNormalization((0.95, 1.05), name="ContrastNormalization"), default_dtypes),
        (iaa.GaussianBlur(sigma=(0.95, 1.05), name="GaussianBlur"), _not_dts([np.float16])),
        (iaa.AverageBlur((3, 5), name="AverageBlur"), _not_dts([np.uint32, np.int32, np.float16])),
        (iaa.MedianBlur((3, 5), name="MedianBlur"), _not_dts([np.uint32, np.int32, np.float16, np.float64])),
        (iaa.BilateralBlur((3, 5), name="BilateralBlur"),
         _not_dts([np.uint16, np.uint32, np.int16, np.int32, np.float16, np.float64])),
        # WithColorspace ?
        # iaa.AddToHueAndSaturation((-5, 5), name="AddToHueAndSaturation"), # works only with RGB/uint8
        # ChangeColorspace ?
        # iaa.Grayscale((0.0, 0.1), name="Grayscale"), # works only with RGB/uint8
        # Convolve ?
        (iaa.Sharpen((0.0, 0.1), lightness=(1.0, 1.2), name="Sharpen"),
         _not_dts([np.uint32, np.int32, np.float16, np.uint32])),
        (iaa.Emboss(alpha=(0.0, 0.1), strength=(0.5, 1.5), name="Emboss"),
         _not_dts([np.uint32, np.int32, np.float16, np.uint32])),
        (iaa.EdgeDetect(alpha=(0.0, 0.1), name="EdgeDetect"),
         _not_dts([np.uint32, np.int32, np.float16, np.uint32])),
        (iaa.DirectedEdgeDetect(alpha=(0.0, 0.1), direction=0, name="DirectedEdgeDetect"),
         _not_dts([np.uint32, np.int32, np.float16, np.uint32])),
        (iaa.Fliplr(0.5, name="Fliplr"), default_dtypes),
        (iaa.Flipud(0.5, name="Flipud"), default_dtypes),
        (iaa.Affine(translate_px=(-5, 5), name="Affine-translate-px"), _not_dts([np.uint32, np.int32])),
        (iaa.Affine(translate_percent=(-0.05, 0.05), name="Affine-translate-percent"), _not_dts([np.uint32, np.int32])),
        (iaa.Affine(rotate=(-20, 20), name="Affine-rotate"), _not_dts([np.uint32, np.int32])),
        (iaa.Affine(shear=(-20, 20), name="Affine-shear"), _not_dts([np.uint32, np.int32])),
        (iaa.Affine(scale=(0.9, 1.1), name="Affine-scale"), _not_dts([np.uint32, np.int32])),
        (iaa.PiecewiseAffine(scale=(0.001, 0.005), name="PiecewiseAffine"), default_dtypes),
        # (iaa.PerspectiveTransform(scale=(0.01, 0.10), name="PerspectiveTransform"), not_dts([np.uint32])),
        (iaa.ElasticTransformation(alpha=(0.1, 0.2), sigma=(0.1, 0.2), name="ElasticTransformation"),
         _not_dts([np.float16])),
        (iaa.Sequential([iaa.Noop(), iaa.Noop()], name="SequentialNoop"), default_dtypes),
        (iaa.SomeOf(1, [iaa.Noop(), iaa.Noop()], name="SomeOfNoop"), default_dtypes),
        (iaa.OneOf([iaa.Noop(), iaa.Noop()], name="OneOfNoop"), default_dtypes),
        (iaa.Sometimes(0.5, iaa.Noop(), name="SometimesNoop"), default_dtypes),
        (iaa.Sequential([iaa.Add((-5, 5)), iaa.AddElementwise((-5, 5))], name="Sequential"), _not_dts([np.uint32, np.int32, np.float64])),
        (iaa.SomeOf(1, [iaa.Add((-5, 5)), iaa.AddElementwise((-5, 5))], name="SomeOf"), _not_dts([np.uint32, np.int32, np.float64])),
        (iaa.OneOf([iaa.Add((-5, 5)), iaa.AddElementwise((-5, 5))], name="OneOf"), _not_dts([np.uint32, np.int32, np.float64])),
        (iaa.Sometimes(0.5, iaa.Add((-5, 5)), name="Sometimes"), _not_dts([np.uint32, np.int32, np.float64])),
        # WithChannels
        (iaa.Noop(name="Noop"), default_dtypes),
        # Lambda
        # AssertLambda
        # AssertShape
        (iaa.Alpha((0.0, 0.1), iaa.Noop(), name="AlphaNoop"), default_dtypes),
        (iaa.AlphaElementwise((0.0, 0.1), iaa.Noop(), name="AlphaElementwiseNoop"), default_dtypes),
        (iaa.SimplexNoiseAlpha(iaa.Noop(), name="SimplexNoiseAlphaNoop"), default_dtypes),
        (iaa.FrequencyNoiseAlpha(exponent=(-2, 2), first=iaa.Noop(), name="SimplexNoiseAlphaNoop"), default_dtypes),
        (iaa.Alpha((0.0, 0.1), iaa.Add(10), name="Alpha"), _not_dts([np.uint32, np.int32, np.float64])),
        (iaa.AlphaElementwise((0.0, 0.1), iaa.Add(10), name="AlphaElementwise"), _not_dts([np.uint32, np.int32, np.float64])),
        (iaa.SimplexNoiseAlpha(iaa.Add(10), name="SimplexNoiseAlpha"), _not_dts([np.uint32, np.int32, np.float64])),
        (iaa.FrequencyNoiseAlpha(exponent=(-2, 2), first=iaa.Add(10), name="SimplexNoiseAlpha"), _not_dts([np.uint32, np.int32, np.float64])),
        (iaa.Superpixels(p_replace=0.01, n_segments=64), _not_dts([np.float16, np.float32, np.float64])),
        (iaa.Resize({"height": 4, "width": 4}, name="Resize"),
         _not_dts([np.uint16, np.uint32, np.int16, np.int32, np.float32, np.float16, np.float64])),
        (iaa.CropAndPad(px=(-10, 10), name="CropAndPad"),
         _not_dts([np.uint16, np.uint32, np.int16, np.int32, np.float32, np.float16, np.float64])),
        (iaa.Pad(px=(0, 10), name="Pad"),
         _not_dts([np.uint16, np.uint32, np.int16, np.int32, np.float32, np.float16, np.float64])),
        (iaa.Crop(px=(0, 10), name="Crop"),
         _not_dts([np.uint16, np.uint32, np.int16, np.int32, np.float32, np.float16, np.float64]))
    ]

    for (aug, allowed_dtypes) in augs:
        # print("aug", aug.name)
        # print("allowed_dtypes", allowed_dtypes)
        for images_i in images:
            if images_i.dtype in allowed_dtypes:
                # print("image dt", images_i.dtype)
                images_aug = aug.augment_images(images_i)
                assert images_aug.dtype == images_i.dtype
            else:
                # print("image dt", images_i.dtype, "[SKIPPED]")
                pass


if __name__ == "__main__":
    main()
