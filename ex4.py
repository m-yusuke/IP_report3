import numpy as np
from PIL import Image
import argparse
import ex3
import random
import matplotlib.pyplot as plt


def show_img(img, sec):
    # 画像表示
    plt.imshow(img)
    # sec秒だけ固定
    if sec <= 0:
        sec = 0.01
    plt.pause(sec)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', '-t', default='./ngym.jpeg')
    parser.add_argument('--seconds', default=5, type=float)
    parser.add_argument('--line', default=3, type=int)
    parser.add_argument('--row', default=3, type=int)
    parser.add_argument('--random', default=False, action='store_true')

    args = parser.parse_args()
    return args

def array_divide(array, num_line=3, num_row=3):
    divided_array = []
    for line in np.array_split(array, num_line, 0):
        for row in np.array_split(line, num_row, 1):
            divided_array.append(row)

    return divided_array

def array_combine(divided_array, num_line, num_row):
    index = 0
    for i in range(num_line):
        for j in range(num_row - 1):
            if j == 0:
                line_array = divided_array[index]
            line_array = np.hstack([line_array, divided_array[index+1]])
            index += 1
        index += 1
        if i == 0:
            combined_array = line_array
            continue
        combined_array = np.vstack([combined_array, line_array])

    return combined_array

if __name__ == '__main__':
    args = get_args()

    img_array = np.array(Image.open(args.target))

    num_line = args.line
    num_row = args.row
    num_elem = num_line * num_row
    divided_img_array = array_divide(img_array, num_line, num_row)

    targetindex = 0
    while True:
        if not args.random:
            if targetindex >= num_elem:
                targetindex = 0
        else:
            targetindex = random.randrange(num_elem)
        divided_img_array[targetindex] = ex3.turn_image(divided_img_array[targetindex])

        img_result = array_combine(divided_img_array, num_line, num_row)
        pil_img = Image.fromarray(img_result.astype(np.uint8))
        show_img(pil_img, args.seconds)
        divided_img_array = array_divide(img_array, num_line, num_row)
        if not args.random:
            targetindex += 1
