import numpy as np
from PIL import Image
import argparse

def turn_image(img_array):
    result = np.zeros(img_array.shape)
    for i, line in enumerate(img_array):
        index = 1
        for row in line:
            # n番目の画素値を-(n+1)に格納することで画像を左右反転
            result[i][-1 * index] = row
            index += 1
    return result

def get_args():
    parser = argparse.ArgumentParser(description='画像左右反転のスクリプト')
    parser.add_argument('--save', '-s', action='store_true', default=False, help='出力結果を保存するするかどうか')
    parser.add_argument('--outname', '-o', default='./ex3_result.png', help='出力結果を保存する際のファイル名の指定')
    parser.add_argument('--target', '-t', default='./sample1.jpeg', help='使用する画像ファイルの指定')
    parser.add_argument('--debug', default=False, action='store_true', help='結果を出力するかどうか')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    img_array = np.array(Image.open(args.target))

    img_result = turn_image(img_array)
    pil_img = Image.fromarray(img_result.astype(np.uint8))

    if args.debug:
        pass
    elif args.save:
        pil_img.save(args.outname)
    else:
        pil_img.show()
