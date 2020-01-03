import numpy as np
from PIL import Image, ImageFilter
import argparse
import ex5


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', '-t', default='./ngym.jpeg')
    parser.add_argument('--save', '-s', action='store_true', default=False)
    parser.add_argument('--outname', '-o', default='./ex6_result.png')
    parser.add_argument('--radius', default=1, type=int)
    parser.add_argument('-k', default=1, type=int)
    parser.add_argument('--debug', default=False, action='store_true')

    args = parser.parse_args()
    return args

def sharpning_filter(array, k=1, radius=1):
    smoothing_array = ex5.gaussian_filter(array, radius=radius)
    result = float(k + 1) * array - float(k) * smoothing_array
    result = np.maximum(result, np.zeros(result.shape))
    result = np.minimum(result, 255 * np.ones(result.shape))
    return result

if __name__ == '__main__':
    args = get_args()

    img = Image.open(args.target)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img_array = np.array(img)

    img_result = sharpning_filter(img_array, args.k, args.radius)

    pil_img = Image.fromarray(img_result.astype(np.uint8))

    if args.save:
        pil_img.save(args.outname)
    elif args.debug:
        pass
    else:
        pil_img.show()