from __future__ import print_function, division, absolute_import
import imgaug as ia
import imgaug.augmenters as iaa
import timeit


def main():
    for size in [64, 128, 256, 512, 1024]:
        for threshold in [64, 128, 192]:
            time_iaa = timeit.timeit(
                "iaa.solarize(image, %d)" % (threshold,),
                number=1000,
                setup=(
                    "import imgaug as ia; "
                    "import imgaug.augmenters as iaa; "
                    "image = ia.quokka_square((%d, %d))" % (size, size))
            )
            time_pil = timeit.timeit(
                "np.asarray("
                "PIL.ImageOps.solarize(PIL.Image.fromarray(image), %d)"
                ")" % (threshold,),
                number=1000,
                setup=(
                    "import numpy as np; "
                    "import PIL.Image; "
                    "import PIL.ImageOps; "
                    "import imgaug as ia; "
                    "image = ia.quokka_square((%d, %d))" % (size, size))
            )
            print("[size=%04d, thresh=%03d] iaa=%.4f pil=%.4f" % (
                size, threshold, time_iaa, time_pil))

    image = ia.quokka_square((128, 128))
    images_aug = iaa.Solarize(1.0)(images=[image] * (5*5))
    ia.imshow(ia.draw_grid(images_aug))


if __name__ == "__main__":
    main()
