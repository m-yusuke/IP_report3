import numpy as np
from PIL import Image
import argparse
import sys
from ex5 import conv2d


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', '-t', default='./ngym.jpeg', help='使用する画像ファイルの指定')
    parser.add_argument('--save', '-s', action='store_true', default=False, help='出力結果を保存するするかどうか')
    parser.add_argument('--outname', '-o', default='./ex7_result.png', help='出力結果を保存する際のファイル名の指定')
    parser.add_argument('--radius', default=1, type=int, help='カーネルの大きさ')
    parser.add_argument('--debug', default=False, action='store_true', help='結果を出力するかどうか')

    args = parser.parse_args()
    return args

def gray_scale(img_array):
    # グレースケール化する
    # 係数については以下のURL参照
    # https://en.wikipedia.org/wiki/Luma_(video)#Rec._601_luma_versus_Rec._709_luma_coefficients
    result = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]

    return result

def sobel_filter(array):
    # グレースケール化する
    gray = gray_scale(array)

    # 横方向のソーベルフィルタ
    kernel_x = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    # 縦方向のソーベルフィルタ
    kernel_y = np.array([[-1, -2, -1],
                     [0, 0, 0],
                     [1, 2, 1]])

    # 畳み込み(ex5.py参照)
    conved_x = conv2d(gray, kernel_x)
    conved_y = conv2d(gray, kernel_y)

    # 画素値の勾配を計算
    result = np.sqrt(conved_x ** 2 + conved_y ** 2)

    return result
    

if __name__ == '__main__':
    args = get_args()

    img = Image.open(args.target)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img_array = np.array(img)

    # ソーベルフィルタを適応
    img_result = sobel_filter(img_array)

    pil_img = Image.fromarray(img_result.astype(np.uint8))

    if args.debug:
        pass
    elif args.save:
        pil_img.save(args.outname)
    else:
        pil_img.show()
