import numpy as np
from PIL import Image
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='平滑化のスクリプト')
    parser.add_argument('--target', '-t', default='./sample1.jpeg', help='使用する画像ファイルの指定')
    parser.add_argument('--save', '-s', action='store_true', default=False, help='出力結果を保存するするかどうか')
    parser.add_argument('--outname', '-o', default='./ex5_result.png', help='出力結果を保存する際のファイル名の指定')
    parser.add_argument('--radius', default=1, type=int, help='カーネルの大きさ')
    parser.add_argument('--gaus', default=False, action='store_true', help='フィルタをガウシアンフィルタに変更')
    parser.add_argument('--debug', default=False, action='store_true', help='結果を出力するかどうか')
    parser.add_argument('--sigma', default=None, type=float, help='ガウシアンフィルタの標準偏差')

    args = parser.parse_args()
    return args

def gaus2d(x, y, sigma):
    # ガウス分布の式
    h = np.exp(-(x**2 + y**2)/(2 * sigma**2))/(2 * np.pi * sigma**2)
    return h

def gaussian_kernel(radius, sigma=None):
    if sigma is None:
        sigma = radius/2 # ガウス分布の計算に用いるシグマ値を設定
    size = radius * 2 + 1 # カーネルサイズ
    
    x = y = np.arange(0,size) - radius #x, yの値をx=y=0を中心に移動 
    X,Y = np.meshgrid(x,y) 
    
    # 値の分布をガウス分布へ
    mat = gaus2d(X,Y,sigma)
    
    # 加重平均化カーネルの作成
    kernel = mat / np.sum(mat)
    return kernel

def conv2d(array, kernel):
    # arrayをkernelで畳み込み
    radius = int(kernel.shape[1]/2)
    result = np.zeros(array.shape)
    for num_line, line in enumerate(array):
        for num_row, pixel in enumerate(line):
            # カーネルで覆う範囲の指定(範囲がarrayから見切れる場合はarrayの端を指定)
            l_start = num_line - radius if num_line - radius >= 0 else 0
            l_end = num_line + radius if num_line + radius < array.shape[0] else array.shape[0] - 1
            r_start = num_row - radius if num_row - radius >= 0 else 0
            r_end = num_row + radius if num_row + radius < array.shape[1] else array.shape[1] - 1

            # 範囲が見切れた場合にpaddingするサイズを指定
            padding_size = ((l_start - num_line + radius, num_line - l_end + radius),(r_start - num_row + radius, num_row - r_end + radius))

            filted_area = array[l_start:l_end+1, r_start:r_end+1]

            # 見切れる場合はarrayの端の値でpadding
            padded = np.pad(filted_area, padding_size, mode='edge')

            convolved = padded * kernel

            # 畳み込み
            result[num_line, num_row] = np.sum(convolved)

    return result

def gaussian_filter(array, radius=1, sigma=None):
    result = np.zeros(array.shape)
    kernel = gaussian_kernel(radius, sigma)
    # R,G,Bそれぞれに対して畳み込みを行う
    result[:, :, 0] = conv2d(array[:, :, 0], kernel)
    result[:, :, 1] = conv2d(array[:, :, 1], kernel)
    result[:, :, 2] = conv2d(array[:, :, 2], kernel)

    return result

def averaging_filter(array, radius=1):
    result = np.zeros(array.shape)
    kernel = np.zeros((radius*2+1, radius*2+1))
    # 平均化カーネル
    kernel[:,:] = 1/(kernel.shape[0]**2)
    # R,G,Bそれぞれに対して畳み込みを行う
    result[:, :, 0] = conv2d(array[:, :, 0], kernel)
    result[:, :, 1] = conv2d(array[:, :, 1], kernel)
    result[:, :, 2] = conv2d(array[:, :, 2], kernel)

    return result

if __name__ == '__main__':
    args = get_args()

    img = Image.open(args.target)
    # カラーモードをRGBへ
    if img.mode != "RGB":
        img = img.convert("RGB")
    img_array = np.array(img)

    if args.gaus:
        # ガウシアンフィルタにより平滑化
        img_result = gaussian_filter(img_array, args.radius, args.sigma)
    else:
        # 平均化フィルタにより平滑化
        img_result = averaging_filter(img_array, args.radius)

    pil_img = Image.fromarray(img_result.astype(np.uint8))

    if args.debug:
        pass
    elif args.save:
        pil_img.save(args.outname)
    else:
        pil_img.show()
