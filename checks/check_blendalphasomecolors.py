from __future__ import print_function, division, absolute_import
import imageio
import imgaug as ia
import imgaug.augmenters as iaa


def main():
    aug = iaa.BlendAlphaMask(
        iaa.SomeColorsMaskGen(),
        iaa.OneOf([
            iaa.TotalDropout(1.0),
            iaa.AveragePooling(8)
        ])
    )

    aug2 = iaa.BlendAlphaSomeColors(iaa.OneOf([
            iaa.TotalDropout(1.0),
            iaa.AveragePooling(8)
    ]))

    urls = [
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/"
         "Sarcophilus_harrisii_taranna.jpg/"
         "320px-Sarcophilus_harrisii_taranna.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/b/ba/"
         "Vincent_van_Gogh_-_Wheatfield_with_crows_-_Google_Art_Project.jpg/"
         "320px-Vincent_van_Gogh_-_Wheatfield_with_crows_-_Google_Art_Project"
         ".jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/"
         "Galerella_sanguinea_Zoo_Praha_2011-2.jpg/207px-Galerella_sanguinea_"
         "Zoo_Praha_2011-2.jpg"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/"
         "Ambrosius_Bosschaert_the_Elder_%28Dutch_-_Flower_Still_Life_-_"
         "Google_Art_Project.jpg/307px-Ambrosius_Bosschaert_the_Elder_%28"
         "Dutch_-_Flower_Still_Life_-_Google_Art_Project.jpg")
    ]

    for url in urls:
        img = imageio.imread(url)
        ia.imshow(ia.draw_grid(aug(images=[img]*25), cols=5, rows=5))
        ia.imshow(ia.draw_grid(aug2(images=[img]*25), cols=5, rows=5))


if __name__ == "__main__":
    main()
