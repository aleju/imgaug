from __future__ import print_function, division
import imgaug as ia
import augmenters as iaa
import parameters as iap
#from skimage import
import numpy as np
from scipy import ndimage, misc
from skimage import data
import matplotlib.pyplot as plt
from matplotlib import gridspec
import six
import six.moves as sm

def main():
    draw_single_sequential_images()
    draw_per_augmenter_images()

def draw_single_sequential_images():
    image = misc.imresize(ndimage.imread("quokka.jpg")[0:643, 0:643], (128, 128))

    st = lambda aug: iaa.Sometimes(0.5, aug)

    seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            st(iaa.Crop(percent=(0, 0.1))),
            st(iaa.GaussianBlur((0, 3.0))),
            st(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)),
            st(iaa.Dropout((0.0, 0.1), per_channel=0.5)),
            st(iaa.Add((-10, 10), per_channel=0.5)),
            st(iaa.Multiply((0.5, 1.5), per_channel=0.5)),
            st(iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)),
            st(iaa.Grayscale(alpha=(0.0, 1.0), name="Grayscale")),
            st(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_px={"x": (-16, 16), "y": (-16, 16)},
                rotate=(-45, 45),
                shear=(-16, 16),
                order=[0, 1],
                cval=(0, 1.0),
                mode=ia.ALL
            )),
            st(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25))
        ],
        random_order=True
    )

    grid = seq.draw_grid(image, cols=8, rows=8)
    misc.imsave("examples_grid.jpg", grid)

def draw_per_augmenter_images():
    print("[draw_per_augmenter_images] Loading image...")
    image = misc.imresize(ndimage.imread("quokka.jpg")[0:643, 0:643], (128, 128))
    #image = misc.imresize(data.chelsea()[0:300, 50:350, :], (128, 128))
    #image = misc.imresize(data.astronaut(), (128, 128))

    #keypoints = [ia.Keypoint(x=43, y=43), ia.Keypoint(x=78, y=40), ia.Keypoint(x=64, y=73)] # left eye, right eye, mouth
    keypoints = [ia.Keypoint(x=34, y=15), ia.Keypoint(x=85, y=13), ia.Keypoint(x=63, y=73)] # left ear, right ear, mouth
    keypoints = [ia.KeypointsOnImage(keypoints, shape=image.shape)]

    print("[draw_per_augmenter_images] Initializing...")
    rows_augmenters = [
        ("Noop", [("", iaa.Noop()) for _ in sm.xrange(5)]),
        #("Crop", [iaa.Crop(px=vals) for vals in [(2, 4), (4, 8), (6, 16), (8, 32), (10, 64)]]),
        ("Crop\n(top, right,\nbottom, left)", [(str(vals), iaa.Crop(px=vals)) for vals in [(2, 0, 0, 0), (0, 8, 8, 0), (4, 0, 16, 4), (8, 0, 0, 32), (32, 64, 0, 0)]]),
        ("Fliplr", [(str(p), iaa.Fliplr(p)) for p in [0, 0, 1, 1, 1]]),
        ("Flipud", [(str(p), iaa.Flipud(p)) for p in [0, 0, 1, 1, 1]]),
        ("Add", [("value=%d" % (val,), iaa.Add(val)) for val in [-45, -25, 0, 25, 45]]),
        ("Add\n(per channel)", [("value=(%d, %d)" % (vals[0], vals[1],), iaa.Add(vals, per_channel=True)) for vals in [(-55, -35), (-35, -15), (-10, 10), (15, 35), (35, 55)]]),
        ("Multiply", [("value=%.2f" % (val,), iaa.Multiply(val)) for val in [0.25, 0.5, 1.0, 1.25, 1.5]]),
        ("Multiply\n(per channel)", [("value=(%.2f, %.2f)" % (vals[0], vals[1],), iaa.Multiply(vals, per_channel=True)) for vals in [(0.15, 0.35), (0.4, 0.6), (0.9, 1.1), (1.15, 1.35), (1.4, 1.6)]]),
        ("GaussianBlur", [("sigma=%.2f" % (sigma,), iaa.GaussianBlur(sigma=sigma)) for sigma in [0.25, 0.50, 1.0, 2.0, 4.0]]),
        ("AdditiveGaussianNoise", [("scale=%.2f" % (scale,), iaa.AdditiveGaussianNoise(scale=scale * 255)) for scale in [0.025, 0.05, 0.1, 0.2, 0.3]]),
        ("AdditiveGaussianNoise\n(per channel)", [("scale=%.2f" % (scale,), iaa.AdditiveGaussianNoise(scale=scale * 255, per_channel=True)) for scale in [0.025, 0.05, 0.1, 0.2, 0.3]]),
        ("Dropout", [("p=%.2f" % (p,), iaa.Dropout(p=p)) for p in [0.025, 0.05, 0.1, 0.2, 0.4]]),
        ("Dropout\n(per channel)", [("p=%.2f" % (p,), iaa.Dropout(p=p, per_channel=True)) for p in [0.025, 0.05, 0.1, 0.2, 0.4]]),
        ("ContrastNormalization", [("alpha=%.1f" % (alpha,), iaa.ContrastNormalization(alpha=alpha)) for alpha in [0.5, 0.75, 1.0, 1.25, 1.50]]),
        ("ContrastNormalization\n(per channel)", [("alpha=(%.2f, %.2f)" % (alphas[0], alphas[1],), iaa.ContrastNormalization(alpha=alphas, per_channel=True)) for alphas in [(0.4, 0.6), (0.65, 0.85), (0.9, 1.1), (1.15, 1.35), (1.4, 1.6)]]),
        ("Grayscale", [("alpha=%.1f" % (alpha,), iaa.Grayscale(alpha=alpha)) for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]]),
        ("Affine: Scale", [("%.1fx" % (scale,), iaa.Affine(scale=scale)) for scale in [0.1, 0.5, 1.0, 1.5, 1.9]]),
        ("Affine: Translate", [("x=%d y=%d" % (x, y), iaa.Affine(translate_px={"x": x, "y": y})) for x, y in [(-32, -16), (-16, -32), (-16, -8), (16, 8), (16, 32)]]),
        ("Affine: Rotate", [("%d deg" % (rotate,), iaa.Affine(rotate=rotate)) for rotate in [-90, -45, 0, 45, 90]]),
        ("Affine: Shear", [("%d deg" % (shear,), iaa.Affine(shear=shear)) for shear in [-45, -25, 0, 25, 45]]),
        ("Affine: Modes", [(mode, iaa.Affine(translate_px=-32, mode=mode)) for mode in ["constant", "edge", "symmetric", "reflect", "wrap"]]),
        ("Affine: cval", [("%.2f" % (cval,), iaa.Affine(translate_px=-32, cval=cval, mode="constant")) for cval in [0.0, 0.25, 0.5, 0.75, 1.0]]),
        (
            "Affine: all", [
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
        ("ElasticTransformation\n(sigma=0.2)", [("alpha=%.1f" % (alpha,), iaa.ElasticTransformation(alpha=alpha, sigma=0.2)) for alpha in [0.1, 0.5, 1.0, 3.0, 9.0]])
    ]

    print("[draw_per_augmenter_images] Augmenting...")
    rows = []
    for (row_name, augmenters) in rows_augmenters:
        row_images = []
        row_keypoints = []
        row_titles = []
        for img_title, augmenter in augmenters:
            aug_det = augmenter.to_deterministic()
            row_images.append(aug_det.augment_image(image))
            row_keypoints.append(aug_det.augment_keypoints(keypoints)[0])
            row_titles.append(img_title)
        rows.append((row_name, row_images, row_keypoints, row_titles))

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

if __name__ == "__main__":
    main()
