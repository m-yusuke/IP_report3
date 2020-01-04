import numpy as np
from PIL import Image
import argparse
import ex3
import random
import matplotlib.pyplot as plt


def show_img(img, sec):
    # 画像表示
    plt.imshow(img)
    # secが0以下の場合は0.01に設定
    if sec <= 0:
        sec = 0.01
    # sec秒だけ固定
    plt.pause(sec)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', '-t', default='./sample1.jpeg', help='使用する画像ファイルの指定')
    parser.add_argument('--seconds', default=5, type=float, help='タイルを入れ替える秒数を指定')
    parser.add_argument('--line', default=3, type=int, help='区切られる行数を指定')
    parser.add_argument('--row', default=3, type=int, help='区切られる列数を指定')
    parser.add_argument('--random', default=False, action='store_true', help='反転するタイルの順番をランダムにする')
    parser.add_argument('--debug', default=False, action='store_true', help='結果を出力するかどうか')

    args = parser.parse_args()
    return args

def array_divide(array, num_line=3, num_row=3):
    # 配列を指定された行, 列数に分割
    divided_array = []
    for line in np.array_split(array, num_line, 0):
        for row in np.array_split(line, num_row, 1):
            divided_array.append(row)

    return divided_array

def array_combine(divided_array, num_line, num_row):
    # 分割された配列を結合し直す
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

    num_line = args.line # 行数
    num_row = args.row # 列数
    num_elem = num_line * num_row # タイルの枚数
    divided_img_array = array_divide(img_array, num_line, num_row)

    targetindex = 0
    while True:
        if args.debug:
            break
        if not args.random:
            if targetindex >= num_elem:
                targetindex = 0
        else:
            # ランダムオプションが有効の場合はtargetindexをタイルの枚数のうちからランダムに指定
            targetindex = random.randrange(num_elem)
        # targetindexで指定した箇所のタイルを左右反転(ex3.py参照)
        divided_img_array[targetindex] = ex3.turn_image(divided_img_array[targetindex])

        img_result = array_combine(divided_img_array, num_line, num_row)
        pil_img = Image.fromarray(img_result.astype(np.uint8))
        show_img(pil_img, args.seconds)
        # タイルを反転する前の配列を上書きしもとに戻す
        divided_img_array = array_divide(img_array, num_line, num_row)
        if not args.random:
            # ランダムオプションが有効でない場合はtargetindexをインクリメント
            targetindex += 1
