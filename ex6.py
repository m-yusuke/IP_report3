import numpy as np
from PIL import Image
import argparse
import ex5


def get_args():
    parser = argparse.ArgumentParser(description='鮮鋭化のスクリプト')
    parser.add_argument('--target', '-t', default='./sample1.jpeg', help='使用する画像ファイルの指定')
    parser.add_argument('--save', '-s', action='store_true', default=False, help='出力結果を保存するするかどうか')
    parser.add_argument('--outname', '-o', default='./ex6_result.png', help='出力結果を保存する際のファイル名の指定')
    parser.add_argument('--radius', default=1, type=int, help='カーネルの大きさ')
    parser.add_argument('-k', default=1, type=int, help='鮮鋭化の度合い')
    parser.add_argument('--debug', default=False, action='store_true', help='結果を出力するかどうか')

    args = parser.parse_args()
    return args

def sharpning_filter(array, k=1, radius=1):
    # ガウシアンフィルタにより平滑化
    smoothing_array = ex5.gaussian_filter(array, radius=radius)
    # arrayにarrayと平滑化したarrayの差をk倍したものを足す
    result = float(k + 1) * array - float(k) * smoothing_array
    # 0未満の値を取り除く
    result = np.maximum(result, np.zeros(result.shape))
    # 255より大きい値を取り除く
    result = np.minimum(result, 255 * np.ones(result.shape))
    return result

if __name__ == '__main__':
    args = get_args()

    img = Image.open(args.target)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img_array = np.array(img)

    # 鮮鋭化フィルタを適応
    img_result = sharpning_filter(img_array, args.k, args.radius)

    pil_img = Image.fromarray(img_result.astype(np.uint8))

    if args.debug:
        pass
    elif args.save:
        pil_img.save(args.outname)
    else:
        pil_img.show()
