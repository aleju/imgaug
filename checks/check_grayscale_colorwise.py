import imgaug as ia
import imgaug.augmenters as iaa
import imageio


def main():
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

    image = imageio.imread(urls[0])

    aug = iaa.GrayscaleColorwise(255, 0.2, alpha=[0.0, 1.0])
    images_aug = aug(images=[image] * (5*5))

    ia.imshow(ia.draw_grid(images_aug))


if __name__ == "__main__":
    main()
