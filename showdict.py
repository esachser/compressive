import numpy as np
import glob
import spams
import time
from skimage import io, color
import matplotlib.pyplot as plt

import dictlearn as dl

sizeX = 4
sizeY = 4

if __name__ == "__main__":
    from skimage.measure import compare_psnr

    D = np.loadtxt('dltrainfiles/dl4_rgb_ds16_Bunny.txt')
    # D = np.loadtxt('ksvdtrain/ksvd8_rgb_ds128.txt')
    print(D.shape)
    
    D = D.reshape(-1, 4, 4, 3)
    # D = (D - D.min()) / 2
    # D = (D - D.min()) / np.ptp(D)
    print(D.max(), D.min())
    D = (D+1) / 2
    # D = color.ycbcr2rgb(D)

    # l, c = (0,0)
    for i in range(D.shape[0]):
        plt.subplot(sizeX, sizeY, i+1)
        plt.axis('off')
        plt.imshow(D[i])
    # io.imshow(color.yuv2rgb(D).clip(0,1))
    io.show()