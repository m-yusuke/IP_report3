import numpy as np
from PIL import Image, ImageFilter
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', '-t', default='./ngym.jpeg')
    parser.add_argument('--save', '-s', action='store_true', default=False)
    parser.add_argument('--outname', '-o', default='./ex5_result.png')
    parser.add_argument('--radius', default=1, type=int)
    parser.add_argument('--gaus', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')

    args = parser.parse_args()
    return args

def averaging_filter(array, radius=1):
    result = array.copy()
    for num_line, line in enumerate(array):
        for num_row, pixel in enumerate(line):
            l_start = num_line - radius if num_line - radius >= 0 else 0
            l_end = num_line + radius if num_line + radius < array.shape[0] else array.shape[0] - 1
            r_start = num_row - radius if num_row - radius >= 0 else 0
            r_end = num_row + radius if num_row + radius < array.shape[1] else array.shape[1] - 1
            result[num_line, num_row, 0] = np.mean(array[l_start:l_end+1, r_start:r_end+1, 0])
            result[num_line, num_row, 1] = np.mean(array[l_start:l_end+1, r_start:r_end+1, 1])
            result[num_line, num_row, 2] = np.mean(array[l_start:l_end+1, r_start:r_end+1, 2])

    return result

def gaus2d(x, y, sigma):
    h = np.exp(-(x**2 + y**2)/(2 * sigma**2))/(2 * np.pi * sigma**2)
    return h

def gaussian_kernel(radius):
    sigma = radius/2
    size = radius * 2 + 1
    
    x = y = np.arange(0,size) - radius
    X,Y = np.meshgrid(x,y) 
    
    mat = gaus2d(X,Y,sigma)
    
    kernel = mat / np.sum(mat)
    return kernel

def gaussian_filter(array, radius=1):
    result = array.copy()
    kernel = gaussian_kernel(radius)
    print(kernel)
    for num_line, line in enumerate(array):
        for num_row, pixel in enumerate(line):
            l_start = num_line - radius if num_line - radius >= 0 else 0
            l_end = num_line + radius if num_line + radius < array.shape[0] else array.shape[0] - 1
            r_start = num_row - radius if num_row - radius >= 0 else 0
            r_end = num_row + radius if num_row + radius < array.shape[1] else array.shape[1] - 1
            filted_area_R = array[l_start:l_end+1, r_start:r_end+1, 0]
            filted_area_G = array[l_start:l_end+1, r_start:r_end+1, 1]
            filted_area_B = array[l_start:l_end+1, r_start:r_end+1, 2]
            result[num_line, num_row, 0] = np.mean(array[l_start:l_end+1, r_start:r_end+1, 0])
            result[num_line, num_row, 1] = np.mean(array[l_start:l_end+1, r_start:r_end+1, 1])
            result[num_line, num_row, 2] = np.mean(array[l_start:l_end+1, r_start:r_end+1, 2])

    return result

if __name__ == '__main__':
    args = get_args()

    img = Image.open(args.target)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img_array = np.array(img)

    if args.gaus:
        img_result = gaussian_filter(img_array, args.radius)
    else:
        img_result = averaging_filter(img_array, args.radius)

    pil_img = Image.fromarray(img_result.astype(np.uint8))

    if args.save:
        pil_img.save(args.outname)
    elif args.debug:
        pass
    else:
        pil_img.show()
