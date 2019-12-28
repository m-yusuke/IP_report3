import numpy as np
from PIL import Image, ImageFilter
import argparse
import ex5
import cv2


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
    print(array.shape)
    #Image.fromarray(array.astype(np.uint8)).show()
    smoothing_array = ex5.gaussian_filter(array, radius=radius)
    #unsharp_image = cv2.addWeighted(array, 1.5, smoothing_array, -0.5, 0, array)
    #Image.fromarray(smoothing_array.astype(np.uint8)).show()
    print(smoothing_array.shape)
    diff_array = array - smoothing_array
    #Image.fromarray(diff_array.astype(np.uint8)).show()
    print(diff_array.shape)
    result = array + diff_array * k
    #Image.fromarray(result.astype(np.uint8)).show()
    print(result.shape)
    return result
    #return unsharp_image
    #return hoge


if __name__ == '__main__':
    args = get_args()

    img = Image.open(args.target)
    #img.filter(ImageFilter.UnsharpMask(radius=1)).show()
    if img.mode != "RGB":
        print("hoge")
        img = img.convert("RGB")
    img_array = np.array(img)

    img_result = sharpning_filter(img_array, args.k, args.radius)

    #diff_array = img_array - img.filter(ImageFilter.BoxBlur(radius=1))
    #img_result = img_array + diff_array

    pil_img = Image.fromarray(img_result.astype(np.uint8))

    if args.save:
        pil_img.save(args.outname)
    elif args.debug:
        pass
    else:
        pil_img.show()
