import numpy as np
from PIL import Image
import argparse


def get_args():
    # 使用可能なオプションの指定
    parser = argparse.ArgumentParser(description='トーンカーブ変換のスクリプト')
    parser.add_argument('--target', '-t', default='./HLSColorSpace.png', help='使用する画像ファイルの指定')
    parser.add_argument('--save', '-s', action='store_true', default=False, help='出力結果を保存するするかどうか')
    parser.add_argument('--outname', '-o', default='./ex1_result.png', help='出力結果を保存する際のファイル名の指定')
    parser.add_argument('--gamma', default=4, type=int, help='ガンマ値')
    parser.add_argument('--debug', default=False, action='store_true', help='結果を出力するかどうか')

    args = parser.parse_args()
    return args

def gamma_transform(array, gamma=1):
    array = 255 * ((array/255)**(1/gamma)) # ガンマ変換の式
    return array

if __name__ == '__main__':
    args = get_args()

    img_array = np.array(Image.open(args.target)) # 画像をnumpy形式の配列に変換

    img_result = gamma_transform(img_array, args.gamma) # ガンマ変換を実行
    
    pil_img = Image.fromarray(img_result.astype(np.uint8)) # numpy形式の配列から画像へ変換
    
    if args.debug:
        pass # debugオプションが有効の場合は出力せず終了
    elif args.save:
        pil_img.save(args.outname) # saveオプションが有効の場合はファイルに保存
    else:
        pil_img.show() # Finderに出力
