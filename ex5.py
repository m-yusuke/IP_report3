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

def conv2d(array, kernel):
    radius = int(kernel.shape[1]/2)
    result = np.zeros(array.shape)
    for num_line, line in enumerate(array):
        for num_row, pixel in enumerate(line):
            l_start = num_line - radius if num_line - radius >= 0 else 0
            l_end = num_line + radius if num_line + radius < array.shape[0] else array.shape[0] - 1
            r_start = num_row - radius if num_row - radius >= 0 else 0
            r_end = num_row + radius if num_row + radius < array.shape[1] else array.shape[1] - 1

            padding_size = ((l_start - num_line + radius, num_line - l_end + radius),(r_start - num_row + radius, num_row - r_end + radius))

            filted_area = array[l_start:l_end+1, r_start:r_end+1]

            padded = np.pad(filted_area, padding_size, mode='edge')

            convolved = padded * kernel

            result[num_line, num_row] = np.sum(convolved)

    return result

def gaussian_filter(array, radius=1):
    result = np.zeros(array.shape)
    kernel = gaussian_kernel(radius)
    result[:, :, 0] = conv2d(array[:, :, 0], kernel)
    result[:, :, 1] = conv2d(array[:, :, 1], kernel)
    result[:, :, 2] = conv2d(array[:, :, 2], kernel)

    return result

def averaging_filter(array, radius=1):
    result = np.zeros(array.shape)
    kernel = np.zeros((radius*2+1, radius*2+1))
    kernel[:,:] = 1/(kernel.shape[0]**2)
    result[:, :, 0] = conv2d(array[:, :, 0], kernel)
    result[:, :, 1] = conv2d(array[:, :, 1], kernel)
    result[:, :, 2] = conv2d(array[:, :, 2], kernel)

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
