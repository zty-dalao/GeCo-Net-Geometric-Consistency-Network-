import pywt
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def show_wavelet_subbands(image_path, wavelet='db4', save_path=None):
    """
    对图像做单层二维离散小波变换，并可视化 LL / LH / HL / HH 四个子带。

    参数:
        image_path : str, 图片路径
        wavelet    : str, 小波名称，如 'db4', 'sym5', 'db8', 'haar'
        save_path  : str 或 None, 若给出则把子带拼图保存到该路径
    """
    # 1. 读图并转灰度（单通道）
    img = Image.open(image_path).convert('L')
    img = np.array(img, dtype=np.float32)

    # 2. 单层二维小波分解
    #    coeffs = (LL, (LH, HL, HH))
    coeffs = pywt.dwt2(img, wavelet=wavelet, mode='symmetric')
    LL, (LH, HL, HH) = coeffs

    print(f"原图尺寸: {img.shape}")
    print(f"LL 子带尺寸: {LL.shape}")
    print(f"各子带取值范围: LL [{LL.min():.2f}, {LL.max():.2f}], "
          f"LH [{LH.min():.2f}, {LH.max():.2f}], "
          f"HL [{HL.min():.2f}, {HL.max():.2f}], "
          f"HH [{HH.min():.2f}, {HH.max():.2f}]")

    # 3. 可视化：LL 用普通灰度显示；高频子带取绝对值增强对比度
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(LL, cmap='gray')
    axes[1].set_title(f'LL (approx)\n{LL.shape}')
    axes[1].axis('off')

    # 高频子带用绝对值，方便观察边缘/细节
    axes[2].imshow(np.abs(LH), cmap='gray')
    axes[2].set_title(f'LH (horizontal detail)\n{LH.shape}')
    axes[2].axis('off')

    axes[3].imshow(np.abs(HL), cmap='gray')
    axes[3].set_title(f'HL (vertical detail)\n{HL.shape}')
    axes[3].axis('off')

    axes[4].imshow(np.abs(HH), cmap='gray')
    axes[4].set_title(f'HH (diagonal detail)\n{HH.shape}')
    axes[4].axis('off')

    plt.suptitle(f'2D DWT with wavelet = "{wavelet}"', fontsize=14)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"拼图已保存到: {save_path}")

    plt.show()

    return coeffs


if __name__ == '__main__':
    # 修改成你自己的图片路径
    image_path = './image/results_fig.png'

    # 可选：比较不同小波的效果
    # show_wavelet_subbands(image_path, wavelet='db4')
    show_wavelet_subbands(image_path, wavelet='sym5')
    # show_wavelet_subbands(image_path, wavelet='haar')