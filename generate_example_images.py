from __future__ import print_function, division
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from scipy import ndimage, misc
from skimage import data
import matplotlib.pyplot as plt
import six.moves as sm
import re
import os
from collections import defaultdict
import PIL.Image
try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO

np.random.seed(44)
ia.seed(44)

IMAGES_DIR = "images"

def main():
    draw_single_sequential_images()
    draw_per_augmenter_images()

def draw_single_sequential_images():
    ia.seed(44)

    #image = misc.imresize(ndimage.imread("quokka.jpg")[0:643, 0:643], (128, 128))
    image = ia.quokka_square(size=(128, 128))

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            iaa.Flipud(0.2), # vertically flip 20% of all images
            # crop images by -5% to 10% of their height/width
            sometimes(iaa.CropAndPad(
                percent=(-0.05, 0.1),
                pad_mode=ia.ALL,
                pad_cval=(0, 255)
            )),
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45), # rotate by -45 to +45 degrees
                shear=(-16, 16), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                [
                    sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                    # search either for all edges or for directed edges,
                    # blend the result with the original image using a blobby mask
                    iaa.SimplexNoiseAlpha(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0.5, 1.0)),
                        iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                    ])),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                        iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                    ]),
                    iaa.Invert(0.05, per_channel=True), # invert color channels
                    iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                    iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                    # either change the brightness of the whole image (sometimes
                    # per channel) or change the brightness of subareas
                    iaa.OneOf([
                        iaa.Multiply((0.5, 1.5), per_channel=0.5),
                        iaa.FrequencyNoiseAlpha(
                            exponent=(-4, 0),
                            first=iaa.Multiply((0.5, 1.5), per_channel=True),
                            second=iaa.ContrastNormalization((0.5, 2.0))
                        )
                    ]),
                    iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                    sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                ],
                random_order=True
            )
        ],
        random_order=True
    )

    grid = seq.draw_grid(image, cols=8, rows=8)
    misc.imsave("examples_grid.jpg", grid)

def draw_per_augmenter_images():
    print("[draw_per_augmenter_images] Loading image...")
    #image = misc.imresize(ndimage.imread("quokka.jpg")[0:643, 0:643], (128, 128))
    image = ia.quokka_square(size=(128, 128))

    keypoints = [ia.Keypoint(x=34, y=15), ia.Keypoint(x=85, y=13), ia.Keypoint(x=63, y=73)] # left ear, right ear, mouth
    keypoints = [ia.KeypointsOnImage(keypoints, shape=image.shape)]

    print("[draw_per_augmenter_images] Initializing...")
    rows_augmenters = [
        (0, "Noop", [("", iaa.Noop()) for _ in sm.xrange(5)]),
        (0, "Crop\n(top, right,\nbottom, left)", [(str(vals), iaa.Crop(px=vals)) for vals in [(2, 0, 0, 0), (0, 8, 8, 0), (4, 0, 16, 4), (8, 0, 0, 32), (32, 64, 0, 0)]]),
        (0, "Pad\n(top, right,\nbottom, left)", [(str(vals), iaa.Pad(px=vals)) for vals in [(2, 0, 0, 0), (0, 8, 8, 0), (4, 0, 16, 4), (8, 0, 0, 32), (32, 64, 0, 0)]]),
        (0, "Fliplr", [(str(p), iaa.Fliplr(p)) for p in [0, 0, 1, 1, 1]]),
        (0, "Flipud", [(str(p), iaa.Flipud(p)) for p in [0, 0, 1, 1, 1]]),
        (0, "Superpixels\np_replace=1", [("n_segments=%d" % (n_segments,), iaa.Superpixels(p_replace=1.0, n_segments=n_segments)) for n_segments in [25, 50, 75, 100, 125]]),
        (0, "Superpixels\nn_segments=100", [("p_replace=%.2f" % (p_replace,), iaa.Superpixels(p_replace=p_replace, n_segments=100)) for p_replace in [0, 0.25, 0.5, 0.75, 1.0]]),
        (0, "Invert", [("p=%d" % (p,), iaa.Invert(p=p)) for p in [0, 0, 1, 1, 1]]),
        (0, "Invert\n(per_channel)", [("p=%.2f" % (p,), iaa.Invert(p=p, per_channel=True)) for p in [0.5, 0.5, 0.5, 0.5, 0.5]]),
        (0, "Add", [("value=%d" % (val,), iaa.Add(val)) for val in [-45, -25, 0, 25, 45]]),
        (0, "Add\n(per channel)", [("value=(%d, %d)" % (vals[0], vals[1],), iaa.Add(vals, per_channel=True)) for vals in [(-55, -35), (-35, -15), (-10, 10), (15, 35), (35, 55)]]),
        (0, "AddToHueAndSaturation", [("value=%d" % (val,), iaa.AddToHueAndSaturation(val)) for val in [-45, -25, 0, 25, 45]]),
        (0, "Multiply", [("value=%.2f" % (val,), iaa.Multiply(val)) for val in [0.25, 0.5, 1.0, 1.25, 1.5]]),
        (1, "Multiply\n(per channel)", [("value=(%.2f, %.2f)" % (vals[0], vals[1],), iaa.Multiply(vals, per_channel=True)) for vals in [(0.15, 0.35), (0.4, 0.6), (0.9, 1.1), (1.15, 1.35), (1.4, 1.6)]]),
        (0, "GaussianBlur", [("sigma=%.2f" % (sigma,), iaa.GaussianBlur(sigma=sigma)) for sigma in [0.25, 0.50, 1.0, 2.0, 4.0]]),
        (0, "AverageBlur", [("k=%d" % (k,), iaa.AverageBlur(k=k)) for k in [1, 3, 5, 7, 9]]),
        (0, "MedianBlur", [("k=%d" % (k,), iaa.MedianBlur(k=k)) for k in [1, 3, 5, 7, 9]]),
        (0, "BilateralBlur\nsigma_color=250,\nsigma_space=250", [("d=%d" % (d,), iaa.BilateralBlur(d=d, sigma_color=250, sigma_space=250)) for d in [1, 3, 5, 7, 9]]),
        (0, "Sharpen\n(alpha=1)", [("lightness=%.2f" % (lightness,), iaa.Sharpen(alpha=1, lightness=lightness)) for lightness in [0, 0.5, 1.0, 1.5, 2.0]]),
        (0, "Emboss\n(alpha=1)", [("strength=%.2f" % (strength,), iaa.Emboss(alpha=1, strength=strength)) for strength in [0, 0.5, 1.0, 1.5, 2.0]]),
        (0, "EdgeDetect", [("alpha=%.2f" % (alpha,), iaa.EdgeDetect(alpha=alpha)) for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]]),
        (0, "DirectedEdgeDetect\n(alpha=1)", [("direction=%.2f" % (direction,), iaa.DirectedEdgeDetect(alpha=1, direction=direction)) for direction in [0.0, 1*(360/5)/360, 2*(360/5)/360, 3*(360/5)/360, 4*(360/5)/360]]),
        (0, "AdditiveGaussianNoise", [("scale=%.2f*255" % (scale,), iaa.AdditiveGaussianNoise(scale=scale * 255)) for scale in [0.025, 0.05, 0.1, 0.2, 0.3]]),
        (0, "AdditiveGaussianNoise\n(per channel)", [("scale=%.2f*255" % (scale,), iaa.AdditiveGaussianNoise(scale=scale * 255, per_channel=True)) for scale in [0.025, 0.05, 0.1, 0.2, 0.3]]),
        (0, "Dropout", [("p=%.2f" % (p,), iaa.Dropout(p=p)) for p in [0.025, 0.05, 0.1, 0.2, 0.4]]),
        (0, "Dropout\n(per channel)", [("p=%.2f" % (p,), iaa.Dropout(p=p, per_channel=True)) for p in [0.025, 0.05, 0.1, 0.2, 0.4]]),
        (3, "CoarseDropout\n(p=0.2)", [("size_percent=%.2f" % (size_percent,), iaa.CoarseDropout(p=0.2, size_percent=size_percent, min_size=2)) for size_percent in [0.3, 0.2, 0.1, 0.05, 0.02]]),
        (0, "CoarseDropout\n(p=0.2, per channel)", [("size_percent=%.2f" % (size_percent,), iaa.CoarseDropout(p=0.2, size_percent=size_percent, per_channel=True, min_size=2)) for size_percent in [0.3, 0.2, 0.1, 0.05, 0.02]]),
        (0, "SaltAndPepper", [("p=%.2f" % (p,), iaa.SaltAndPepper(p=p)) for p in [0.025, 0.05, 0.1, 0.2, 0.4]]),
        (0, "Salt", [("p=%.2f" % (p,), iaa.Salt(p=p)) for p in [0.025, 0.05, 0.1, 0.2, 0.4]]),
        (0, "Pepper", [("p=%.2f" % (p,), iaa.Pepper(p=p)) for p in [0.025, 0.05, 0.1, 0.2, 0.4]]),
        (0, "CoarseSaltAndPepper\n(p=0.2)", [("size_percent=%.2f" % (size_percent,), iaa.CoarseSaltAndPepper(p=0.2, size_percent=size_percent, min_size=2)) for size_percent in [0.3, 0.2, 0.1, 0.05, 0.02]]),
        (0, "CoarseSalt\n(p=0.2)", [("size_percent=%.2f" % (size_percent,), iaa.CoarseSalt(p=0.2, size_percent=size_percent, min_size=2)) for size_percent in [0.3, 0.2, 0.1, 0.05, 0.02]]),
        (0, "CoarsePepper\n(p=0.2)", [("size_percent=%.2f" % (size_percent,), iaa.CoarsePepper(p=0.2, size_percent=size_percent, min_size=2)) for size_percent in [0.3, 0.2, 0.1, 0.05, 0.02]]),
        (0, "ContrastNormalization", [("alpha=%.1f" % (alpha,), iaa.ContrastNormalization(alpha=alpha)) for alpha in [0.5, 0.75, 1.0, 1.25, 1.50]]),
        (0, "ContrastNormalization\n(per channel)", [("alpha=(%.2f, %.2f)" % (alphas[0], alphas[1],), iaa.ContrastNormalization(alpha=alphas, per_channel=True)) for alphas in [(0.4, 0.6), (0.65, 0.85), (0.9, 1.1), (1.15, 1.35), (1.4, 1.6)]]),
        (0, "Grayscale", [("alpha=%.1f" % (alpha,), iaa.Grayscale(alpha=alpha)) for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]]),
        (6, "PerspectiveTransform", [("scale=%.3f" % (scale,), iaa.PerspectiveTransform(scale=scale)) for scale in [0.025, 0.05, 0.075, 0.10, 0.125]]),
        (0, "PiecewiseAffine", [("scale=%.3f" % (scale,), iaa.PiecewiseAffine(scale=scale)) for scale in [0.015, 0.03, 0.045, 0.06, 0.075]]),
        (0, "Affine: Scale", [("%.1fx" % (scale,), iaa.Affine(scale=scale)) for scale in [0.1, 0.5, 1.0, 1.5, 1.9]]),
        (0, "Affine: Translate", [("x=%d y=%d" % (x, y), iaa.Affine(translate_px={"x": x, "y": y})) for x, y in [(-32, -16), (-16, -32), (-16, -8), (16, 8), (16, 32)]]),
        (0, "Affine: Rotate", [("%d deg" % (rotate,), iaa.Affine(rotate=rotate)) for rotate in [-90, -45, 0, 45, 90]]),
        (0, "Affine: Shear", [("%d deg" % (shear,), iaa.Affine(shear=shear)) for shear in [-45, -25, 0, 25, 45]]),
        (0, "Affine: Modes", [(mode, iaa.Affine(translate_px=-32, mode=mode)) for mode in ["constant", "edge", "symmetric", "reflect", "wrap"]]),
        (0, "Affine: cval", [("%d" % (int(cval*255),), iaa.Affine(translate_px=-32, cval=int(cval*255), mode="constant")) for cval in [0.0, 0.25, 0.5, 0.75, 1.0]]),
        (
            2, "Affine: all", [
                (
                    "",
                    iaa.Affine(
                        scale={"x": (0.5, 1.5), "y": (0.5, 1.5)},
                        translate_px={"x": (-32, 32), "y": (-32, 32)},
                        rotate=(-45, 45),
                        shear=(-32, 32),
                        mode=ia.ALL,
                        cval=(0.0, 1.0)
                    )
                )
                for _ in sm.xrange(5)
            ]
        ),
        (1, "ElasticTransformation\n(sigma=0.2)", [("alpha=%.1f" % (alpha,), iaa.ElasticTransformation(alpha=alpha, sigma=0.2)) for alpha in [0.1, 0.5, 1.0, 3.0, 9.0]]),
        (0, "Alpha\nwith EdgeDetect(1.0)", [("factor=%.1f" % (factor,), iaa.Alpha(factor=factor, first=iaa.EdgeDetect(1.0))) for factor in [0.0, 0.25, 0.5, 0.75, 1.0]]),
        (4, "Alpha\nwith EdgeDetect(1.0)\n(per channel)", [("factor=(%.2f, %.2f)" % (factor[0], factor[1]), iaa.Alpha(factor=factor, first=iaa.EdgeDetect(1.0), per_channel=0.5)) for factor in [(0.0, 0.2), (0.15, 0.35), (0.4, 0.6), (0.65, 0.85), (0.8, 1.0)]]),
        (15, "SimplexNoiseAlpha\nwith EdgeDetect(1.0)", [("", iaa.SimplexNoiseAlpha(first=iaa.EdgeDetect(1.0))) for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]]),
        (9, "FrequencyNoiseAlpha\nwith EdgeDetect(1.0)", [("exponent=%.1f" % (exponent,), iaa.FrequencyNoiseAlpha(exponent=exponent, first=iaa.EdgeDetect(1.0), size_px_max=16, upscale_method="linear", sigmoid=False)) for exponent in [-4, -2, 0, 2, 4]])
    ]

    print("[draw_per_augmenter_images] Augmenting...")
    rows = []
    for (row_seed, row_name, augmenters) in rows_augmenters:
        ia.seed(row_seed)
        #for img_title, augmenter in augmenters:
        #    #aug.reseed(1000)
        #    pass

        row_images = []
        row_keypoints = []
        row_titles = []
        for img_title, augmenter in augmenters:
            aug_det = augmenter.to_deterministic()
            row_images.append(aug_det.augment_image(image))
            row_keypoints.append(aug_det.augment_keypoints(keypoints)[0])
            row_titles.append(img_title)
        rows.append((row_name, row_images, row_keypoints, row_titles))

    # matplotlib drawin routine
    """
    print("[draw_per_augmenter_images] Plotting...")
    width = 8
    height = int(1.5 * len(rows_augmenters))
    fig = plt.figure(figsize=(width, height))
    grid_rows = len(rows)
    grid_cols = 1 + 5
    gs = gridspec.GridSpec(grid_rows, grid_cols, width_ratios=[2, 1, 1, 1, 1, 1])
    axes = []
    for i in sm.xrange(grid_rows):
        axes.append([plt.subplot(gs[i, col_idx]) for col_idx in sm.xrange(grid_cols)])
    fig.tight_layout()
    #fig.subplots_adjust(bottom=0.2 / grid_rows, hspace=0.22)
    #fig.subplots_adjust(wspace=0.005, hspace=0.425, bottom=0.02)
    fig.subplots_adjust(wspace=0.005, hspace=0.005, bottom=0.02)

    for row_idx, (row_name, row_images, row_keypoints, row_titles) in enumerate(rows):
        axes_row = axes[row_idx]

        for col_idx in sm.xrange(grid_cols):
            ax = axes_row[col_idx]

            ax.cla()
            ax.axis("off")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            if col_idx == 0:
                ax.text(0, 0.5, row_name, color="black")
            else:
                cell_image = row_images[col_idx-1]
                cell_keypoints = row_keypoints[col_idx-1]
                cell_image_kp = cell_keypoints.draw_on_image(cell_image, size=5)
                ax.imshow(cell_image_kp)
                x = 0
                y = 145
                #ax.text(x, y, row_titles[col_idx-1], color="black", backgroundcolor="white", fontsize=6)
                ax.text(x, y, row_titles[col_idx-1], color="black", fontsize=7)


    fig.savefig("examples.jpg", bbox_inches="tight")
    #plt.show()
    """

    # simpler and faster drawing routine
    """
    output_image = ExamplesImage(128, 128, 128+64, 32)
    for (row_name, row_images, row_keypoints, row_titles) in rows:
        row_images_kps = []
        for image, keypoints in zip(row_images, row_keypoints):
            row_images_kps.append(keypoints.draw_on_image(image, size=5))
        output_image.add_row(row_name, row_images_kps, row_titles)
    misc.imsave("examples.jpg", output_image.draw())
    """

    # routine to draw many single files
    seen = defaultdict(lambda: 0)
    markups = []
    for (row_name, row_images, row_keypoints, row_titles) in rows:
        output_image = ExamplesImage(128, 128, 128+64, 32)
        row_images_kps = []
        for image, keypoints in zip(row_images, row_keypoints):
            row_images_kps.append(keypoints.draw_on_image(image, size=5))
        output_image.add_row(row_name, row_images_kps, row_titles)
        if "\n" in row_name:
            row_name_clean = row_name[0:row_name.find("\n")+1]
        else:
            row_name_clean = row_name
        row_name_clean = re.sub(r"[^a-z0-9]+", "_", row_name_clean.lower())
        row_name_clean = row_name_clean.strip("_")
        if seen[row_name_clean] > 0:
            row_name_clean = "%s_%d" % (row_name_clean, seen[row_name_clean] + 1)
        fp = os.path.join(IMAGES_DIR, "examples_%s.jpg" % (row_name_clean,))
        #misc.imsave(fp, output_image.draw())
        save(fp, output_image.draw())
        seen[row_name_clean] += 1

        markup_descr = row_name.replace('"', '') \
                               .replace("\n", " ") \
                               .replace("(", "") \
                               .replace(")", "")
        markup = '![%s](%s?raw=true "%s")' % (markup_descr, fp, markup_descr)
        markups.append(markup)

    for markup in markups:
        print(markup)

class ExamplesImage(object):
    def __init__(self, image_height, image_width, title_cell_width, subtitle_height):
        self.rows = []
        self.image_height = image_height
        self.image_width = image_width
        self.title_cell_width = title_cell_width
        self.cell_height = image_height + subtitle_height
        self.cell_width = image_width

    def add_row(self, title, images, subtitles):
        assert len(images) == len(subtitles)
        images_rs = []
        for image in images:
            images_rs.append(ia.imresize_single_image(image, (self.image_height, self.image_width)))
        self.rows.append((title, images_rs, subtitles))

    def draw(self):
        rows_drawn = [self.draw_row(title, images, subtitles) for title, images, subtitles in self.rows]
        grid = np.vstack(rows_drawn)
        return grid

    def draw_row(self, title, images, subtitles):
        title_cell = np.zeros((self.cell_height, self.title_cell_width, 3), dtype=np.uint8) + 255
        title_cell = ia.draw_text(title_cell, x=2, y=12, text=title, color=[0, 0, 0], size=16)

        image_cells = []
        for image, subtitle in zip(images, subtitles):
            image_cell = np.zeros((self.cell_height, self.cell_width, 3), dtype=np.uint8) + 255
            image_cell[0:image.shape[0], 0:image.shape[1], :] = image
            image_cell = ia.draw_text(image_cell, x=2, y=image.shape[0]+2, text=subtitle, color=[0, 0, 0], size=11)
            image_cells.append(image_cell)

        row = np.hstack([title_cell] + image_cells)
        return row

#
# TODO this part is largely copied from generate_documentation_images, make DRY
#

def compress_to_jpg(image, quality=75):
    quality = quality if quality is not None else 75
    im = PIL.Image.fromarray(image)
    out = BytesIO()
    im.save(out, format="JPEG", quality=quality)
    jpg_string = out.getvalue()
    out.close()
    return jpg_string

def decompress_jpg(image_compressed):
    img_compressed_buffer = BytesIO()
    img_compressed_buffer.write(image_compressed)
    img = ndimage.imread(img_compressed_buffer, mode="RGB")
    img_compressed_buffer.close()
    return img

def arrdiff(arr1, arr2):
    nb_cells = np.prod(arr2.shape)
    d_avg = np.sum(np.power(np.abs(arr1.astype(np.float64) - arr2.astype(np.float64)), 2)) / nb_cells
    return d_avg

def save(fp, image, quality=75):
    image_jpg = compress_to_jpg(image, quality=quality)
    image_jpg_decompressed = decompress_jpg(image_jpg)

    # If the image file already exists and is (practically) identical,
    # then don't save it again to avoid polluting the repository with tons
    # of image updates.
    # Not that we have to compare here the results AFTER jpg compression
    # and then decompression. Otherwise we compare two images of which
    # image (1) has never been compressed while image (2) was compressed and
    # then decompressed.
    if os.path.isfile(fp):
        image_saved = ndimage.imread(fp, mode="RGB")
        #print("arrdiff", arrdiff(image_jpg_decompressed, image_saved))
        same_shape = (image_jpg_decompressed.shape == image_saved.shape)
        d_avg = arrdiff(image_jpg_decompressed, image_saved) if same_shape else -1
        if same_shape and d_avg <= 1.0:
            print("[INFO] Did not save image '%s', because the already saved image is basically identical (d_avg=%.8f)" % (fp, d_avg,))
            return
        else:
            print("[INFO] Saving image '%s'..." % (fp,))

    with open(fp, "w") as f:
        f.write(image_jpg)

if __name__ == "__main__":
    main()
