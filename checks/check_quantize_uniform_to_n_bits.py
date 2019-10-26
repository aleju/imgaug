from __future__ import print_function, division, absolute_import
import imgaug.augmenters as iaa
import imgaug as ia
import timeit


def main():
    print("--------")
    for size in [64, 128, 256, 512, 1024]:
        for nb_bits in [1, 2, 3, 4, 5, 6, 7, 8]:
            time_iaa = timeit.timeit(
                "iaa.quantize_uniform_to_n_bits(image, %d)" % (nb_bits,),
                number=1000,
                setup=(
                    "import imgaug as ia; "
                    "import imgaug.augmenters as iaa; "
                    "image = ia.quokka_square((%d, %d))" % (size, size))
            )
            time_pil = timeit.timeit(
                "np.asarray("
                "PIL.ImageOps.posterize(PIL.Image.fromarray(image), %d)"
                ")" % (nb_bits,),
                number=1000,
                setup=(
                    "import numpy as np; "
                    "import PIL.Image; "
                    "import PIL.ImageOps; "
                    "import imgaug as ia; "
                    "image = ia.quokka_square((%d, %d))" % (size, size))
            )
            print("[size=%04d, bits=%d] iaa=%.4f pil=%.4f" % (
                size, nb_bits, time_iaa, time_pil))

    image = ia.quokka_square((128, 128))
    images_q = [iaa.quantize_uniform_to_n_bits(image, nb_bits)
                for nb_bits
                in [1, 2, 3, 4, 5, 6, 7, 8]]

    ia.imshow(ia.draw_grid(images_q, cols=8, rows=1))


def posterize(arr, n_bits):
    import numpy as np
    import PIL.Image
    import PIL.ImageOps
    img = PIL.Image.fromarray(arr)
    img_q = PIL.ImageOps.posterize(img, n_bits)
    return np.asarray(img_q)


if __name__ == "__main__":
    main()
