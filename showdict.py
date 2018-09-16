import numpy as np
import glob
import spams
import time
from skimage import io, color
from sklearn.feature_extraction import image
from sklearn.linear_model import orthogonal_mp_gram
import matplotlib.pyplot as plt

if __name__ == "__main__":
    from skimage.measure import compare_psnr

    D = np.loadtxt('dltrain/dl8_yuv_ds96_2.txt')
    
    D = D.reshape(-1, 8, 8, 3)
    D = (D - D.min()) / np.ptp(D)
    print(D.shape)
    ps = []
    for i in range(D.shape[0]):
        ps.append(D[i])
    print(D.max(), D.min())
    
    io.imshow(np.hstack(ps))
    # io.imshow(color.yuv2rgb(D).clip(0,1))
    io.show()