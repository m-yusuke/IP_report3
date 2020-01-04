import numpy as np
from PIL import Image
import argparse
import sys
import ex5


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', '-t', default='./ngym.jpeg', help='使用する画像ファイルの指定')
    parser.add_argument('--save', '-s', action='store_true', default=False, help='出力結果を保存するするかどうか')
    parser.add_argument('--outname', '-o', default='./ex7_result.png', help='出力結果を保存する際のファイル名の指定')
    parser.add_argument('--radius', default=1, type=int, help='カーネルの大きさ')
    parser.add_argument('--debug', default=False, action='store_true', help='結果を出力するかどうか')
    parser.add_argument('--low', default=100, type=int, help='下側しきい値')
    parser.add_argument('--high', default=200, type=int, help='上側しきい値')
    parser.add_argument('--sigma', default=None, type=float, help='ガウシアンフィルタの標準偏差')
    parser.add_argument('--canny', default=False, action='store_true', help='canny法でエッジ抽出を行う')

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
    conved_x = ex5.conv2d(gray, kernel_x)
    conved_y = ex5.conv2d(gray, kernel_y)

    return conved_x, conved_y

def non_max_sup(grad, grad_size):
    h, w = grad.shape
    result = grad.copy()

    # 勾配方向を4方向(垂直・水平・斜め右上・斜め左上)に近似
    grad_size[np.where((grad_size >= -22.5) & (grad_size < 22.5))] = 0
    grad_size[np.where((grad_size >= 157.5) & (grad_size < 180))] = 0
    grad_size[np.where((grad_size >= -180) & (grad_size < -157.5))] = 0
    grad_size[np.where((grad_size >= 22.5) & (grad_size < 67.5))] = 45
    grad_size[np.where((grad_size >= -157.5) & (grad_size < -112.5))] = 45
    grad_size[np.where((grad_size >= 67.5) & (grad_size < 112.5))] = 90
    grad_size[np.where((grad_size >= -112.5) & (grad_size < -67.5))] = 90
    grad_size[np.where((grad_size >= 112.5) & (grad_size < 157.5))] = 135
    grad_size[np.where((grad_size >= -67.5) & (grad_size < -22.5))] = 135
    # 注目画素と勾配方向に隣接する2つの画素値を比較し、注目画素値が最大でなければ0に
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if grad_size[y][x] == 0:
                if (grad[y][x] < grad[y][x+1]) or (grad[y][x] < grad[y][x-1]):
                    result[y][x] = 0
            elif grad_size[y][x] == 45:
                if (grad[y][x] < grad[y-1][x+1]) or (grad[y][x] < grad[y+1][x-1]):
                    result[y][x] = 0
            elif grad_size[y][x] == 90:
                if (grad[y][x] < grad[y+1][x]) or (grad[y][x] < grad[y-1][x]):
                    result[y][x] = 0
            else:
                if (grad[y][x] < grad[y+1][x+1]) or (grad[y][x] < grad[y-1][x-1]):
                    result[y][x] = 0

    return result

# Hysteresis Threshold処理
def hysteresis_threshold(array, th_low=75, th_high=150, d=1):

    h, w = array.shape
    result = array.copy()

    for y in range(0, h):
        for x in range(0, w):
            # 最大閾値より大きければ信頼性の高い輪郭
            if array[y][x] >= th_high:
                result[y][x] = 0
            # 最小閾値より小さければ信頼性の低い輪郭(除去)
            elif array[y][x] < th_low:
                result[y][x] = 255
            # 最小閾値～最大閾値の間なら、近傍に信頼性の高い輪郭が1つでもあれば輪郭と判定、無ければ除去
            else:
                try:
                    if np.max(array[y-d:y+d+1, x-d:x+d+1]) >= th_high:
                        result[y][x] = 0
                    else:
                        result[y][x] = 255
                except ValueError:
                    result[y][x] = 255

    return result

def canny_edge_detecter(array, th_low, th_high, sigma=None, radius=1):
    # ガウシアンフィルタを適応
    smoothed = ex5.gaussian_filter(array, radius, sigma)

    # x, yそれぞれの微分画像
    diff_x, diff_y = sobel_filter(smoothed)

    # 勾配
    grad = np.sqrt(diff_x ** 2 + diff_y ** 2)
    # 勾配の大きさ
    grad_size = np.arctan2(diff_y, diff_x) * 180/np.pi

    grad = non_max_sup(grad, grad_size)

    result = hysteresis_threshold(grad, th_low, th_high)

    return result
    

if __name__ == '__main__':
    args = get_args()

    img = Image.open(args.target)
    if img.size[1] > 164:
        img = img.resize((int(img.size[0] / (img.size[1]/164)), 164))
    if img.mode != "RGB":
        img = img.convert("RGB")

    print(img.size)
    img_array = np.array(img)

    # ソーベルフィルタを適応
    img_result = sobel_filter(img_array)

    if args.canny:
        # canny法でエッジ抽出を行う
        img_result = canny_edge_detecter(img_array, args.low, args.high, sigma=args.sigma)
    else:
        # ソーベルフィルタを適応
        conved_x, conved_y = sobel_filter(img_array)
        img_result = np.sqrt(conved_x ** 2 + conved_y ** 2)

    pil_img = Image.fromarray(img_result.astype(np.uint8))

    if args.debug:
        pass
    elif args.save:
        pil_img.save(args.outname)
    else:
        pil_img.show()
