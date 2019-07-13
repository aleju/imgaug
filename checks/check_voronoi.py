from __future__ import print_function, division

import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa


def main():
    image = ia.quokka_square((256, 256))

    reggrid_sampler = iaa.DropoutPointsSampler(
        iaa.RegularGridPointsSampler(n_rows=50, n_cols=50),
        0.5
    )
    uniform_sampler = iaa.UniformPointsSampler(50*50)

    augs = [
        iaa.Voronoi(points_sampler=reggrid_sampler, p_replace=1.0,
                    max_size=128),
        iaa.Voronoi(points_sampler=uniform_sampler, p_replace=1.0,
                    max_size=128),
        iaa.UniformVoronoi(50*50, p_replace=1.0, max_size=128),
        iaa.RegularGridVoronoi(50, 50, p_drop_points=0.4, p_replace=1.0,
                               max_size=128),
    ]

    images = [aug(image=image) for aug in augs]

    ia.imshow(np.hstack(images))


if __name__ == "__main__":
    main()
