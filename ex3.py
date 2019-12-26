import numpy as np
from PIL import Image
import argparse

def turn_image(img_array):
    result = img_array.copy()
    for i, line in enumerate(img_array):
        index = 1
        for row in line:
            result[i][-1 * index] = row
            index += 1
    return result

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', '-s', action='store_true', default=False)
    parser.add_argument('--outname', '-o', default='./ex3_result.png')
    parser.add_argument('--target', '-t', default='./ngym.jpeg')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    img_array = np.array(Image.open(args.target))

    img_result = turn_image(img_array)
    pil_img = Image.fromarray(img_result.astype(np.uint8))

    if args.save:
        pil_img.save(args.outname)
    else:
        pil_img.show()
