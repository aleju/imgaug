from __future__ import print_function, division
import pyimgaug as ia
import augmenters2 as iaa
import parameters as iap
#from skimage import
import numpy as np
from scipy import ndimage, misc
from skimage import data
import matplotlib.pyplot as plt
from matplotlib import gridspec

def main():
    draw_single_sequential_images()
    draw_per_augmenter_images()

def draw_single_sequential_images():
    image = misc.imresize(ndimage.imread("quokka.jpg")[0:643, 0:643], (128, 128))

    def st(aug):
        return iaa.Sometimes(0.5, aug)

    seq = iaa.Sequential([
            st(iaa.Noop()),
            st(iaa.Crop(percent=(0, 0.1))),
            st(iaa.Fliplr(0.5)),
            st(iaa.Flipud(0.5)),
            st(iaa.GaussianBlur((0, 3.0))),
            st(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.2))),
            st(iaa.Dropout((0.0, 0.1))),
            st(iaa.Multiply((0.5, 1.5))),
            st(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_px={"x": (-16, 16), "y": (-16, 16)},
                rotate=(-45, 45),
                shear=(-16, 16),
                order=ia.ALL,
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
    images = [image]

    print("[draw_per_augmenter_images] Initializing...")
    rows_augmenters = [
        ("Noop", [("", iaa.Noop()) for _ in range(5)]),
        #("Crop", [iaa.Crop(px=vals) for vals in [(2, 4), (4, 8), (6, 16), (8, 32), (10, 64)]]),
        ("Crop", [(str(vals), iaa.Crop(px=vals)) for vals in [(2, 0, 0, 0), (0, 8, 8, 0), (4, 0, 16, 4), (8, 0, 0, 32), (32, 64, 0, 0)]]),
        ("Fliplr", [(str(p), iaa.Fliplr(p)) for p in [0, 0, 1, 1, 1]]),
        ("Flipud", [(str(p), iaa.Flipud(p)) for p in [0, 0, 1, 1, 1]]),
        ("GaussianBlur", [("sigma=%.2f" % (sigma,), iaa.GaussianBlur(sigma=sigma)) for sigma in [0.25, 0.50, 1.0, 2.0, 4.0]]),
        ("AdditiveGaussianNoise", [("scale=%.2f" % (scale,), iaa.AdditiveGaussianNoise(scale=scale)) for scale in [0.025, 0.05, 0.1, 0.2, 0.3]]),
        ("AdditiveGaussianNoise\n(per channel)", [("scale=%.2f" % (scale,), iaa.AdditiveGaussianNoise(scale=scale, per_channel=True)) for scale in [0.025, 0.05, 0.1, 0.2, 0.3]]),
        ("Dropout", [("p=%.2f" % (p,), iaa.Dropout(p=p)) for p in [0.025, 0.05, 0.1, 0.2, 0.4]]),
        ("Dropout\n(per channel)", [("p=%.2f" % (p,), iaa.Dropout(p=p, per_channel=True)) for p in [0.025, 0.05, 0.1, 0.2, 0.4]]),
        ("Multiply", [("mul=%.1f" % (mul,), iaa.Multiply(mul=mul)) for mul in [0.3, 0.6, 1.0, 1.4, 1.7]]),
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
                for _ in range(5)
            ]
        ),
        ("ElasticTransformation\n(sigma=0.2)", [("a=%.1f" % (alpha,), iaa.ElasticTransformation(alpha=alpha, sigma=0.2)) for alpha in [0.1, 0.5, 1.0, 3.0, 9.0]])
    ]

    print("[draw_per_augmenter_images] Augmenting...")
    rows = []
    for (row_name, augmenters) in rows_augmenters:
        row_images = []
        row_titles = []
        for img_title, augmenter in augmenters:
            row_images.append(augmenter.augment_images(images)[0])
            row_titles.append(img_title)
        rows.append((row_name, row_images, row_titles))

    print("[draw_per_augmenter_images] Plotting...")
    width = 8
    height = 24
    fig = plt.figure(figsize=(width, height))
    grid_rows = len(rows)
    grid_cols = 1 + 5
    gs = gridspec.GridSpec(grid_rows, grid_cols, width_ratios=[2, 1, 1, 1, 1, 1])
    axes = []
    for i in range(grid_rows):
        axes.append([plt.subplot(gs[i, col_idx]) for col_idx in range(grid_cols)])
    fig.tight_layout()
    #fig.subplots_adjust(bottom=0.2 / grid_rows, hspace=0.22)
    #fig.subplots_adjust(wspace=0.005, hspace=0.425, bottom=0.02)
    fig.subplots_adjust(wspace=0.005, hspace=0.005, bottom=0.02)

    for row_idx, (row_name, row_images, row_titles) in enumerate(rows):
        axes_row = axes[row_idx]

        for col_idx in range(grid_cols):
            ax = axes_row[col_idx]

            ax.cla()
            ax.axis("off")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            if col_idx == 0:
                ax.text(0, 0.5, row_name, color="black")
            else:
                ax.imshow(row_images[col_idx-1])
                x = 8
                y = 150
                #ax.text(x, y, row_titles[col_idx-1], color="black", backgroundcolor="white", fontsize=6)
                ax.text(x, y, row_titles[col_idx-1], color="black", fontsize=8)


    fig.savefig("examples.jpg", bbox_inches="tight")
    #plt.show()

if __name__ == "__main__":
    main()
