import numpy as np
from PIL import Image
import sys
import getopt


def gamma_transform(array, gamma=1):
    array = 255 * ((array/255)**(1/gamma))
    return array


if __name__ == '__main__':
    opts, args = getopt.gnu_getopt(sys.argv[1:], "sf:g:", ["save", "filename=", "gamma="])
    do_save = False
    targetfile = './HLSColorSpace.png'
    gamma = 4

    for o, a in opts:
        if o in ("-s", "--save"):
            do_save = True
            if a == '':
                result_destination = './ex1_result.png'
            else:
                result_destination = a
        elif o in ("-f", "--filename"):
            targetfile = a
        elif o in ("-g", "--gamma"):
            gamma = a

    img_array = np.array(Image.open(targetfile))
    img_result = gamma_transform(img_array, gamma)
    pil_img = Image.fromarray(img_result.astype(np.uint8))
    if do_save:
        pil_img.save(result_destination)
    else:
        pil_img.show()
