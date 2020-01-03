import numpy as np
from PIL import Image
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', '-t', default='./HLSColorSpace.png')
    parser.add_argument('--save', '-s', action='store_true', default=False)
    parser.add_argument('--outname', '-o', default='./ex1_result.png')
    parser.add_argument('--gamma', default=4, type=int)
    parser.add_argument('--debug', default=False, action='store_true')

    args = parser.parse_args()
    return args

def gamma_transform(array, gamma=1):
    array = 255 * ((array/255)**(1/gamma))
    return array

if __name__ == '__main__':
    args = get_args()

    img_array = np.array(Image.open(args.target))

    img_result = gamma_transform(img_array, args.gamma)
    
    pil_img = Image.fromarray(img_result.astype(np.uint8))
    
    if args.save:
        pil_img.save(result_destination)
    elif args.debug:
        pass
    else:
        pil_img.show()
