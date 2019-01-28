from __future__ import print_function, division

import imageio
import numpy as np
import six.moves as sm
from skimage import data
from scipy import ndimage
import cv2

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmenters import meta


def main():
    image = data.astronaut()
    image = ia.imresize_single_image(image, (128, 128))

    # check if scipy and cv2 remap similarly
    rs = ia.new_random_state(1)
    aug_scipy = ElasticTransformationScipy(alpha=30, sigma=3, random_state=rs, deterministic=True)
    aug_cv2 = ElasticTransformationCv2(alpha=30, sigma=3, random_state=rs, deterministic=True)
    augs_scipy = aug_scipy.augment_images([image] * 8)
    augs_cv2 = aug_cv2.augment_images([image] * 8)
    ia.imshow(ia.draw_grid(augs_scipy + augs_cv2, rows=2))

    print("alpha=vary, sigma=0.25")
    augs = [iaa.ElasticTransformation(alpha=alpha, sigma=0.25) for alpha in np.arange(0.0, 50.0, 0.1)]
    images_aug = [aug.augment_image(image) for aug in augs]
    ia.imshow(ia.draw_grid(images_aug, cols=10))

    print("alpha=vary, sigma=1.0")
    augs = [iaa.ElasticTransformation(alpha=alpha, sigma=1.0) for alpha in np.arange(0.0, 50.0, 0.1)]
    images_aug = [aug.augment_image(image) for aug in augs]
    ia.imshow(ia.draw_grid(images_aug, cols=10))

    print("alpha=vary, sigma=3.0")
    augs = [iaa.ElasticTransformation(alpha=alpha, sigma=3.0) for alpha in np.arange(0.0, 50.0, 0.1)]
    images_aug = [aug.augment_image(image) for aug in augs]
    ia.imshow(ia.draw_grid(images_aug, cols=10))

    print("alpha=vary, sigma=5.0")
    augs = [iaa.ElasticTransformation(alpha=alpha, sigma=5.0) for alpha in np.arange(0.0, 50.0, 0.1)]
    images_aug = [aug.augment_image(image) for aug in augs]
    ia.imshow(ia.draw_grid(images_aug, cols=10))

    print("alpha=1.0, sigma=vary")
    augs = [iaa.ElasticTransformation(alpha=1.0, sigma=sigma) for sigma in np.arange(0.0, 50.0, 0.1)]
    images_aug = [aug.augment_image(image) for aug in augs]
    ia.imshow(ia.draw_grid(images_aug, cols=10))

    print("alpha=10.0, sigma=vary")
    augs = [iaa.ElasticTransformation(alpha=10.0, sigma=sigma) for sigma in np.arange(0.0, 50.0, 0.1)]
    images_aug = [aug.augment_image(image) for aug in augs]
    ia.imshow(ia.draw_grid(images_aug, cols=10))

    kps = ia.KeypointsOnImage(
        [ia.Keypoint(x=1, y=1),
         ia.Keypoint(x=50, y=24), ia.Keypoint(x=42, y=96), ia.Keypoint(x=88, y=106), ia.Keypoint(x=88, y=53),
         ia.Keypoint(x=0, y=0), ia.Keypoint(x=128, y=128), ia.Keypoint(x=-20, y=30), ia.Keypoint(x=20, y=-30),
         ia.Keypoint(x=-20, y=-30)],
        shape=image.shape
    )

    images = []
    params = [
        (0.0, 0.0),
        (0.2, 0.2),
        (2.0, 0.25),
        (0.25, 3.0),
        (2.0, 3.0),
        (6.0, 3.0),
        (12.0, 3.0),
        (50.0, 5.0),
        (100.0, 5.0),
        (100.0, 10.0)
    ]

    for (alpha, sigma) in params:
        images_row = []
        seqs_row = [
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="constant", cval=0, order=0),
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="constant", cval=128, order=0),
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="constant", cval=255, order=0),
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="constant", cval=0, order=1),
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="constant", cval=128, order=1),
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="constant", cval=255, order=1),
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="constant", cval=0, order=3),
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="constant", cval=128, order=3),
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="constant", cval=255, order=3),
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="nearest", order=0),
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="nearest", order=1),
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="nearest", order=2),
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="nearest", order=3),
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="reflect", order=0),
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="reflect", order=1),
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="reflect", order=2),
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="reflect", order=3),
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="wrap", order=0),
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="wrap", order=1),
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="wrap", order=2),
            iaa.ElasticTransformation(alpha=alpha, sigma=sigma, mode="wrap", order=3)
        ]

        for seq in seqs_row:
            seq_det = seq.to_deterministic()
            image_aug = seq_det.augment_image(image)
            kps_aug = seq_det.augment_keypoints([kps])[0]
            image_aug_kp = np.copy(image_aug)
            image_aug_kp = kps_aug.draw_on_image(image_aug_kp, size=3)
            images_row.append(image_aug_kp)

        images.append(np.hstack(images_row))

    ia.imshow(np.vstack(images))
    imageio.imwrite("elastic_transformations.jpg", np.vstack(images))


#
# These classes are copies of ElasticTransformation and either always remap via scipy or always via cv2
# This way, the remapping can be compared to make sure that scipy and cv2 lead to similar remapping behaviours
# (important if two arrays with different dtypes are supposed to be remapped similarly)
#
class ElasticTransformationScipy(iaa.ElasticTransformation):
    @classmethod
    def map_coordinates(cls, image, dx, dy, order=1, cval=0, mode="constant"):
        # small debug message to be sure that the right function is executed
        print("map_coordinates() with scipy")

        if order == 0 and image.dtype.name in ["uint64", "int64"]:
            raise Exception(("dtypes uint64 and int64 are only supported in ElasticTransformation for order=0, "
                             + "got order=%d with dtype=%s.") % (order, image.dtype.name))

        input_dtype = image.dtype
        if image.dtype.name == "bool":
            image = image.astype(np.float32)
        elif order == 1 and image.dtype.name in ["int8", "int16", "int32"]:
            image = image.astype(np.float64)
        elif order >= 2 and image.dtype.name == "int8":
            image = image.astype(np.int16)
        elif order >= 2 and image.dtype.name == "int32":
            image = image.astype(np.float64)

        shrt_max = 32767
        backend = "cv2"
        if order == 0:
            bad_dtype_cv2 = (image.dtype.name in ["uint32", "uint64", "int64", "float128", "bool"])
        elif order == 1:
            bad_dtype_cv2 = (image.dtype.name in ["uint32", "uint64", "int8", "int16", "int32", "int64", "float128",
                                                  "bool"])
        else:
            bad_dtype_cv2 = (image.dtype.name in ["uint32", "uint64", "int8", "int32", "int64", "float128", "bool"])

        bad_dx_shape_cv2 = (dx.shape[0] >= shrt_max or dx.shape[1] >= shrt_max)
        bad_dy_shape_cv2 = (dy.shape[0] >= shrt_max or dy.shape[1] >= shrt_max)
        if bad_dtype_cv2 or bad_dx_shape_cv2 or bad_dy_shape_cv2:
            backend = "scipy"

        ia.do_assert(image.ndim == 3)
        result = np.copy(image)
        height, width = image.shape[0:2]
        # True was added here, only difference to usual code
        if True or backend == "scipy":
            h, w = image.shape[0:2]
            y, x = np.meshgrid(np.arange(h).astype(np.float32), np.arange(w).astype(np.float32), indexing='ij')
            x_shifted = x + (-1) * dx
            y_shifted = y + (-1) * dy

            for c in sm.xrange(image.shape[2]):
                remapped_flat = ndimage.interpolation.map_coordinates(
                    image[..., c],
                    (y_shifted.flatten(), x_shifted.flatten()),
                    order=order,
                    cval=cval,
                    mode=mode
                )
                remapped = remapped_flat.reshape((height, width))
                result[..., c] = remapped
        else:
            h, w, nb_channels = image.shape

            y, x = np.meshgrid(np.arange(h).astype(np.float32), np.arange(w).astype(np.float32), indexing='ij')
            x_shifted = x + (-1) * dx
            y_shifted = y + (-1) * dy

            if image.dtype.kind == "f":
                cval = float(cval)
            else:
                cval = int(cval)
            border_mode = cls._MAPPING_MODE_SCIPY_CV2[mode]
            interpolation = cls._MAPPING_ORDER_SCIPY_CV2[order]

            is_nearest_neighbour = (interpolation == cv2.INTER_NEAREST)
            map1, map2 = cv2.convertMaps(x_shifted, y_shifted, cv2.CV_16SC2, nninterpolation=is_nearest_neighbour)
            # remap only supports up to 4 channels
            if nb_channels <= 4:
                result = cv2.remap(image, map1, map2, interpolation=interpolation, borderMode=border_mode, borderValue=cval)
                if image.ndim == 3 and result.ndim == 2:
                    result = result[..., np.newaxis]
            else:
                current_chan_idx = 0
                result = []
                while current_chan_idx < nb_channels:
                    channels = image[..., current_chan_idx:current_chan_idx+4]
                    result_c =  cv2.remap(channels, map1, map2, interpolation=interpolation, borderMode=border_mode,
                                          borderValue=cval)
                    if result_c.ndim == 2:
                        result_c = result_c[..., np.newaxis]
                    result.append(result_c)
                    current_chan_idx += 4
                result = np.concatenate(result, axis=2)

        if result.dtype.name != input_dtype.name:
            result = meta.restore_dtypes_(result, input_dtype)

        return result


class ElasticTransformationCv2(iaa.ElasticTransformation):
    @classmethod
    def map_coordinates(cls, image, dx, dy, order=1, cval=0, mode="constant"):
        # small debug message to be sure that the right function is executed
        print("map_coordinates() with cv2")

        if order == 0 and image.dtype.name in ["uint64", "int64"]:
            raise Exception(("dtypes uint64 and int64 are only supported in ElasticTransformation for order=0, "
                             + "got order=%d with dtype=%s.") % (order, image.dtype.name))

        input_dtype = image.dtype
        if image.dtype.name == "bool":
            image = image.astype(np.float32)
        elif order == 1 and image.dtype.name in ["int8", "int16", "int32"]:
            image = image.astype(np.float64)
        elif order >= 2 and image.dtype.name == "int8":
            image = image.astype(np.int16)
        elif order >= 2 and image.dtype.name == "int32":
            image = image.astype(np.float64)

        shrt_max = 32767
        backend = "cv2"
        if order == 0:
            bad_dtype_cv2 = (image.dtype.name in ["uint32", "uint64", "int64", "float128", "bool"])
        elif order == 1:
            bad_dtype_cv2 = (image.dtype.name in ["uint32", "uint64", "int8", "int16", "int32", "int64", "float128",
                                                  "bool"])
        else:
            bad_dtype_cv2 = (image.dtype.name in ["uint32", "uint64", "int8", "int32", "int64", "float128", "bool"])

        bad_dx_shape_cv2 = (dx.shape[0] >= shrt_max or dx.shape[1] >= shrt_max)
        bad_dy_shape_cv2 = (dy.shape[0] >= shrt_max or dy.shape[1] >= shrt_max)
        if bad_dtype_cv2 or bad_dx_shape_cv2 or bad_dy_shape_cv2:
            backend = "scipy"

        ia.do_assert(image.ndim == 3)
        result = np.copy(image)
        height, width = image.shape[0:2]
        # False was added here, only difference to usual code
        if False or backend == "scipy":
            h, w = image.shape[0:2]
            y, x = np.meshgrid(np.arange(h).astype(np.float32), np.arange(w).astype(np.float32), indexing='ij')
            x_shifted = x + (-1) * dx
            y_shifted = y + (-1) * dy

            for c in sm.xrange(image.shape[2]):
                remapped_flat = ndimage.interpolation.map_coordinates(
                    image[..., c],
                    (x_shifted.flatten(), y_shifted.flatten()),
                    order=order,
                    cval=cval,
                    mode=mode
                )
                remapped = remapped_flat.reshape((height, width))
                result[..., c] = remapped
        else:
            h, w, nb_channels = image.shape

            y, x = np.meshgrid(np.arange(h).astype(np.float32), np.arange(w).astype(np.float32), indexing='ij')
            x_shifted = x + (-1) * dx
            y_shifted = y + (-1) * dy

            if image.dtype.kind == "f":
                cval = float(cval)
            else:
                cval = int(cval)
            border_mode = cls._MAPPING_MODE_SCIPY_CV2[mode]
            interpolation = cls._MAPPING_ORDER_SCIPY_CV2[order]

            is_nearest_neighbour = (interpolation == cv2.INTER_NEAREST)
            map1, map2 = cv2.convertMaps(x_shifted, y_shifted, cv2.CV_16SC2, nninterpolation=is_nearest_neighbour)
            # remap only supports up to 4 channels
            if nb_channels <= 4:
                result = cv2.remap(image, map1, map2, interpolation=interpolation, borderMode=border_mode, borderValue=cval)
                if image.ndim == 3 and result.ndim == 2:
                    result = result[..., np.newaxis]
            else:
                current_chan_idx = 0
                result = []
                while current_chan_idx < nb_channels:
                    channels = image[..., current_chan_idx:current_chan_idx+4]
                    result_c =  cv2.remap(channels, map1, map2, interpolation=interpolation, borderMode=border_mode,
                                          borderValue=cval)
                    if result_c.ndim == 2:
                        result_c = result_c[..., np.newaxis]
                    result.append(result_c)
                    current_chan_idx += 4
                result = np.concatenate(result, axis=2)

        if result.dtype.name != input_dtype.name:
            result = meta.restore_dtypes_(result, input_dtype)

        return result


if __name__ == "__main__":
    main()
