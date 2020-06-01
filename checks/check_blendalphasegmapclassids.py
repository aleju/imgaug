from __future__ import print_function, division, absolute_import
import imageio
import imgaug as ia
import imgaug.augmenters as iaa


def main():
    aug = iaa.BlendAlphaMask(
        iaa.SegMapClassIdsMaskGen(1),
        iaa.OneOf([
            iaa.TotalDropout(1.0),
            iaa.AveragePooling(8)
        ])
    )

    aug2 = iaa.BlendAlphaSegMapClassIds(
        1, iaa.OneOf([
            iaa.TotalDropout(1.0),
            iaa.AveragePooling(8)
        ])
    )

    image = ia.data.quokka(0.25)
    segmap = ia.data.quokka_segmentation_map(0.25)

    images_aug, segmaps_aug = aug(images=[image]*25,
                                  segmentation_maps=[segmap]*25)
    ia.imshow(ia.draw_grid(images_aug, cols=5, rows=5))

    images_aug, segmaps_aug = aug2(images=[image]*25,
                                  segmentation_maps=[segmap]*25)
    ia.imshow(ia.draw_grid(images_aug, cols=5, rows=5))

if __name__ == "__main__":
    main()
