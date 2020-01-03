import numpy as np
from PIL import Image
import argparse
import sys


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', '-s', action='store_true', default=False)
    parser.add_argument('--outname', '-o', default='./ex2_result.png')
    parser.add_argument('--target', '-t', nargs=2, default=['./nkmr.png', './ngym.jpeg'])
    parser.add_argument('--alpha', '-a', type=float)

    args = parser.parse_args()
    return args

def alpha_blending(array1, array2, alpha=0.5):
    result = alpha * array1 + (1 - alpha) * array2
    return result

def processing_alpha_blending(array1, array2):
    result = array1.copy()
    itr = np.linspace(0, 1, num=array1.shape[0])
    for index in range(len(array1)):
        tmp = alpha_blending(array1[index], array2[index], itr[index])
        result[index] = tmp

    return result

def align_size(img1, img2):
    if img1.size > img2.size:
        img1 = img1.resize(img2.size)
    elif img2.size > img1.size:
        img2 = img2.resize(img1.size)

    if img1.mode != img2.mode:
        img2 = img2.convert(img1.mode)
    
    return img1.rotate(90, expand=True), img2.rotate(90, expand=True)

if __name__ == '__main__':
    args = get_args()

    img1, img2 = align_size(Image.open(args.target[0]), Image.open(args.target[1]))
    
    img_array1 = np.array(img1)
    img_array2 = np.array(img2)

    if args.alpha is None:
        img_result = processing_alpha_blending(img_array1, img_array2)
    else:
        if 0 <= args.alpha <= 1:
            img_result = alpha_blending(img_array1, img_array2, args.alpha[0])
        else:
            print("invalid value")
            sys.exit(2)

    pil_img = Image.fromarray(img_result.astype(np.uint8)).rotate(-90, expand=True)

    if args.save:
        pil_img.save(args.outname)
    else:
        pil_img.show()
