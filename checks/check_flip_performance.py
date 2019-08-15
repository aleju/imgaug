from __future__ import print_function, division
import timeit
import numpy as np


def main():
    commands = [
        ("slice",
         "arr2 = arr[:, ::-1, :]; "),
        ("slice, contig",
         "arr2 = arr[:, ::-1, :]; "
         "arr2 = np.ascontiguousarray(arr2);"),
        ("fliplr",
         "arr2 = np.fliplr(arr); "),
        ("fliplr contig",
         "arr2 = np.fliplr(arr); "
         "arr2 = np.ascontiguousarray(arr2);"),
        ("cv2",
         "arr2 = cv2.flip(arr, 1); "
         "arr2 = arr2 if arr2.ndim == 3 else arr2[..., np.newaxis]; "),
        ("cv2 contig",
         "arr2 = cv2.flip(arr, 1); "
         "arr2 = arr2 if arr2.ndim == 3 else arr2[..., np.newaxis]; "
         "arr2 = np.ascontiguousarray(arr2); "),
        ("fort cv2",
         "arr = np.asfortranarray(arr); "
         "arr2 = cv2.flip(arr, 1); "
         "arr2 = arr2 if arr2.ndim == 3 else arr2[..., np.newaxis]; "),
        ("fort cv2 contig",
         "arr = np.asfortranarray(arr); "
         "arr2 = cv2.flip(arr, 1); "
         "arr2 = arr2 if arr2.ndim == 3 else arr2[..., np.newaxis]; "
         "arr2 = np.ascontiguousarray(arr2); "),
        ("cv2_",
         "arr = cv2.flip(arr, 1, dst=arr); "
         "arr = arr if arr.ndim == 3 else arr[..., np.newaxis]; "),
        ("cv2_ contig",
         "arr = cv2.flip(arr, 1, dst=arr); "
         "arr = np.ascontiguousarray(arr); "
         "arr = arr if arr.ndim == 3 else arr[..., np.newaxis]; "),
        ("fort cv2_",
         "arr = np.asfortranarray(arr); "
         "arr = cv2.flip(arr, 1, dst=arr); "
         "arr = arr if arr.ndim == 3 else arr[..., np.newaxis]; "),
        ("fort cv2_ contig",
         "arr = np.asfortranarray(arr); "
         "arr = cv2.flip(arr, 1, dst=arr); "
         "arr = np.ascontiguousarray(arr); "
         "arr = arr if arr.ndim == 3 else arr[..., np.newaxis]; "),
        ("cv2_ get",
         "arr = cv2.flip(arr, 1, dst=arr); "
         "arr = arr.get(); "
         "arr = arr if arr.ndim == 3 else arr[..., np.newaxis]; "),
        ("cv2_ get contig",
         "arr = cv2.flip(arr, 1, dst=arr); "
         "arr = arr.get(); "
         "arr = arr if arr.ndim == 3 else arr[..., np.newaxis]; "
         "arr = np.ascontiguousarray(arr); "),
        ("fort cv2_ get",
         "arr = np.asfortranarray(arr); "
         "arr = cv2.flip(arr, 1, dst=arr); "
         "arr = arr.get(); "
         "arr = arr if arr.ndim == 3 else arr[..., np.newaxis]; "),
        ("fort cv2_ get contig",
         "arr = np.asfortranarray(arr); "
         "arr = cv2.flip(arr, 1, dst=arr); "
         "arr = arr.get(); "
         "arr = arr if arr.ndim == 3 else arr[..., np.newaxis]; "
         "arr = np.ascontiguousarray(arr); ")
    ]

    number = 10000
    for dt in ["bool",
               "uint8", "uint16", "uint32", "uint64",
               "int8", "int16", "int32", "int64",
               "float16", "float32", "float64", "float128"]:
        print("")
        print("----------")
        print(dt)
        print("----------")

        last_fliplr_time = None
        for command_i_title, commands_i in commands:
            try:
                # verify that dtype does not change
                arr_name = "arr2"
                if "arr2" not in commands_i:
                    arr_name = "arr"
                _ = timeit.repeat(
                    "%s assert %s.dtype.name == '%s', ('Got dtype ' + %s.dtype.name)" % (
                        commands_i, arr_name, dt, arr_name),
                    setup="import cv2; "
                          "import numpy as np; "
                          "arr = np.ones((224, 224, 3), dtype=np.%s)" % (dt,),
                    repeat=1, number=1)

                times = timeit.repeat(
                    commands_i,
                    setup="import cv2; "
                          "import numpy as np; "
                          "arr = np.ones((224, 224, 3), dtype=np.%s)" % (dt,),
                    repeat=number, number=1)
                time = np.average(times) * 1000
                if command_i_title == "slice, contig":
                    last_fliplr_time = time
                if "cv2" not in command_i_title:
                    print("{:>20s} {:.5f}ms".format(command_i_title, time))
                else:
                    rel_time = last_fliplr_time / time
                    print("{:>20s} {:.5f}ms ({:.2f}x)".format(
                        command_i_title, time, rel_time))
            except (AssertionError, AttributeError, TypeError) as exc:
                print("{:>20s} Error: {}".format(command_i_title, str(exc)))
                # import traceback
                # traceback.print_exc()

    augs = [
        ("Add", "iaa.Add(10)"),
        ("Affine", "iaa.Affine(translate_px={'x': 10})"),
        ("AverageBlur", "iaa.AverageBlur(3)")
    ]
    for aug_name, aug_command in augs:
        print("")
        print("==============================")
        print("flip method followed by %s" % (aug_name,))
        print("==============================")

        number = 5000
        for command_i_title, commands_i in commands:
            dt = "uint8"

            try:
                arr_name = "arr"
                if "arr2" in commands_i:
                    arr_name = "arr2"

                _ = timeit.repeat(
                        "%s assert %s.dtype.name == '%s', ('Got dtype ' + %s.dtype.name)" % (
                            commands_i, arr_name, dt, arr_name),
                        setup="import cv2; "
                              "import numpy as np; "
                              "arr = np.ones((224, 224, 3), dtype=np.%s)" % (dt,),
                        repeat=1, number=1)

                times = timeit.repeat(
                    "%s _ = aug(image=%s);" % (commands_i, arr_name),
                    setup="import cv2; "
                          "import numpy as np; "
                          "import imgaug.augmenters as iaa; "
                          "arr = np.ones((224, 224, 3), dtype=np.%s); "
                          "aug = %s" % (dt, aug_command),
                    repeat=number, number=1)
                time = np.average(times) * 1000
                print("{:>20s} {:.5f}ms".format(command_i_title, time))
            except (AssertionError, AttributeError, TypeError) as exc:
                print("{:>20s} Error: {}".format(command_i_title, str(exc)))


if __name__ == "__main__":
    main()
