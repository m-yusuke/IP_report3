import numpy as np
from PIL import Image
import argparse
import sys


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', '-t', nargs=2, default=['./sample2.png', './sample1.jpeg'], help='使用する画像ファイルの指定')
    parser.add_argument('--save', '-s', action='store_true', default=False, help='出力結果を保存するするかどうか')
    parser.add_argument('--outname', '-o', default='./ex1_result.png', help='出力結果を保存する際のファイル名の指定')
    parser.add_argument('--alpha', type=float, help='アルファ値')
    parser.add_argument('--debug', default=False, action='store_true', help='結果を出力するかどうか')

    args = parser.parse_args()
    return args

def alpha_blending(array1, array2, alpha=0.5):
    result = alpha * array1 + (1 - alpha) * array2 # 引数で指定したアルファ値でアルファブレンディングを実行
    return result

def processing_alpha_blending(array1, array2):
    result = np.zeros(array1.shape)
    itr = np.linspace(0, 1, num=array1.shape[0]) # 画像の位置によってアルファ値を0から1まで変化
    for index in range(array1.shape[0]):
        # アルファブレンディングの実行
        tmp = alpha_blending(array1[index], array2[index], itr[index])
        result[index] = tmp

    return result

def align_size(img1, img2):
    # 画像サイズが異なる場合は揃える
    if img1.size > img2.size:
        img1 = img1.resize(img2.size)
    elif img2.size > img1.size:
        img2 = img2.resize(img1.size)

    # 画像のカラーモードが異なる場合は揃える
    if img1.mode != img2.mode:
        img2 = img2.convert(img1.mode)
    
    # for文で処理しやすくするために回転
    return img1.rotate(90, expand=True), img2.rotate(90, expand=True)

if __name__ == '__main__':
    args = get_args()

    img1, img2 = align_size(Image.open(args.target[0]), Image.open(args.target[1]))
    
    img_array1 = np.array(img1)
    img_array2 = np.array(img2)

    if args.alpha is None:
        # オプションでアルファ値が設定されていない場合は画像位置に寄ってアルファ値を変化させる処理を実行する
        img_result = processing_alpha_blending(img_array1, img_array2)
    else:
        # アルファ値が設定されていればその値でアルファブレンディングを実行
        if 0 <= args.alpha <= 1:
            img_result = alpha_blending(img_array1, img_array2, args.alpha[0])
        else:
            # アルファ値が0から1の範囲に指定されていなければ処理を終了
            print("invalid value")
            sys.exit(2)

    pil_img = Image.fromarray(img_result.astype(np.uint8)).rotate(-90, expand=True)

    if args.debug:
        pass
    elif args.save:
        pil_img.save(args.outname)
    else:
        pil_img.show()
