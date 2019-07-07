from __future__ import print_function, division
import imgaug as ia
import imgaug.augmenters as iaa


def main():
    image = ia.quokka_square((256, 256))

    reggrid_sampler = iaa.DropoutPointsSampler(
        iaa.RegularGridPointsSampler(n_rows=50, n_cols=50),
        0.5
    )

    aug = iaa.Voronoi(point_sampler=reggrid_sampler, p_replace=1.0,
                      max_size=128)

    ia.imshow(aug(image=image))


if __name__ == "__main__":
    main()
