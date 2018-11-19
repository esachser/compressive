import numpy as np
import glob
import spams
import time
from skimage import io, color
import matplotlib.pyplot as plt
import sys

import dictlearn as dl

sizeL = 4
sizeC = 4

pl = 4
pc = 4

if __name__ == "__main__":
    from skimage.measure import compare_psnr

    D = np.loadtxt(sys.argv[1])
    sizeL = int(sys.argv[4])
    sizeC = int(sys.argv[5])
    pl = int(sys.argv[2])
    pc = int(sys.argv[3])
    # D = np.loadtxt('ksvdtrain/ksvd8_rgb_ds128.txt')
    print(D.shape)
    
    D = D.reshape(sizeL*sizeC, pl, pc, 3)
    # D = (D - D.min()) / 2
    # D = (D - D.min()) / np.ptp(D)
    # print(D.max(), D.min())
    D = (D+1) / 2
    # D = color.ycbcr2rgb(D)

    # plt.subplots_adjust(wspace=0, hspace=0)
    # l, c = (0,0)
    # for i in range(D.shape[0]):
    #     plt.subplot(sizeL, sizeC, i+1)
    #     plt.axis('off')
    #     plt.imshow(D[i])

    D = D.reshape(sizeL, sizeC, pl, pc, 3)
    DD = np.ones((sizeL*pl+sizeL-1,sizeC*pc+sizeC-1,3))
    for i in range(sizeL):
        for j in range(sizeC):
            inil = i*pl + i
            inic = j*pc + j
            DD[inil:inil+pl,inic:inic+pc] = D[i,j]
    plt.axis('off')
    plt.imshow(DD)

    # io.imshow(color.yuv2rgb(D).clip(0,1))
    io.show()