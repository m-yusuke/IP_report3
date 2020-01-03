import numpy as np
from PIL import Image, ImageFilter
import argparse
import sys
from ex5 import conv2d


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', '-t', default='./ngym.jpeg')
    parser.add_argument('--save', '-s', action='store_true', default=False)
    parser.add_argument('--outname', '-o', default='./ex7_result.png')
    parser.add_argument('--radius', default=1, type=int)
    parser.add_argument('--debug', default=False, action='store_true')

    args = parser.parse_args()
    return args

def gray_scale(img_array):
    result = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]

    return result

if __name__ == '__main__':
    args = get_args()

    img = Image.open(args.target)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img_array = np.array(img)

    gray = gray_scale(img_array)

    kernel_x = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

    kernel_y = np.array([[-1, -2, -1],
                     [0, 0, 0],
                     [1, 2, 1]])

    gray_x = conv2d(gray, kernel_x)
    gray_y = conv2d(gray, kernel_y)
    img_result = np.sqrt(gray_x ** 2 + gray_y ** 2)

    pil_img = Image.fromarray(img_result.astype(np.uint8))

    if args.save:
        pil_img.save(args.outname)
    elif args.debug:
        pass
    else:
        pil_img.show()
