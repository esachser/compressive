import numpy as np
import glob
import spams
import time
from skimage import io, color
from sklearn.feature_extraction import image
import matplotlib.pyplot as plt
import dictlearn as dl
from pathlib import Path
import time

fname = str(Path.home()) + '/Imagens/720pMoria.png'

m11 = 8
m22 = m11

if __name__ == "__main__":
    from skimage.measure import compare_psnr

    # D = np.loadtxt('dltrainfiles/dl8_ycbcr_ds96_720pmoria.txt').T
    D = dl.dct_dict(256,m11)
    # print(D.shape)
    # exit()

    image = io.imread(fname)
    if image.shape[-1] == 4:
        image = color.rgba2rgb(image)
    img_train = color.rgb2gray(image) / 255.
    # img_train = image/255.

    nl, nc = img_train.shape
    img4 = img_train
    Ps = []
    for i in range(0, nl, m11):
        for j in range(0, nc, m22):
            if i + m11 <= nl and j + m22 <= nc:
                p = img4[i:(i+m11), j:(j+m22)]
                Ps.append(p)

    Ps = np.array(Ps).reshape(len(Ps), -1).T
    psnrs = []

    initial = np.asfortranarray(np.zeros(D.shape[1]))
    res = np.zeros((D.shape[1], Ps.shape[1]))
    A0 = spams.ssp.csc_matrix((res.shape[0],res.shape[1]),dtype=np.float)

    for i in range(1, int(Ps.shape[0])+1):
        mask = np.zeros(Ps.shape[0])
        choose = np.random.choice(Ps.shape[0], i, replace=False)
        # print(mask.shape)
        mask[choose] = 1
        # Dnew = np.asfortranarray((D * np.outer(mask, np.ones((1, D.shape[1])))),np.float)
        Dnew = np.asfortranarray(D[choose, :])
        Psnew = np.asfortranarray(Ps[choose, :])
        # res = np.array(dl.omp_mask(Ps, mask, D, 10, n_threads=4))
        # for j in range(Ps.shape[1]):
        #     # print(res[:,j].shape)
        #     res[:,j] = dl.sparse.iterative_hard_thresholding(Psnew[:,j], Dnew, 3, 0.1, initial, 12)
        res = spams.lasso(Psnew, Dnew, lambda1=0.000001, L=12).toarray()
        # res = spams.cd(Psnew, Dnew, A0, lambda1=0.00001)
        res2 = D.dot(res).T
        img1 = np.vstack((spl.reshape(-1,m11,m22).transpose(1,0,2).reshape(m11,-1) for spl in np.vsplit(res2, int(nl / m11))))
        # imgrgb = color.ycbcr2rgb(img1*255).clip(0,1)
        imgrgb = (img1)
        psnrs.append(compare_psnr(img_train*255, imgrgb*255))
        print("%d: %.3f" % (i, psnrs[-1]))
    # dl.ps
    # plt.subplot(2,1,1)
    # plt.plot(psnrs, 'o-')

    # print(imgrgb.min(), imgrgb.max())
    # plt.subplot(2,1,2)
    plt.imshow(imgrgb, cmap='gray')
    plt.show()


