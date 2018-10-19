import numpy as np
import glob
import spams
import time
from skimage import io, color
# from sklearn.feature_extraction import image
import matplotlib.pyplot as plt
import dictlearn as dl
from pathlib import Path
import time

fname = str(Path.home()) + '/Imagens/720pMoria.png'

m11, m22 = (8,8)

if __name__ == "__main__":
    from skimage.measure import compare_psnr

    D = np.loadtxt('dltrainfiles/dl8_ycbcr_ds96_720pmoria.txt').T
    # D = np.zeros((192,64))
    # D[::3] = dl.dct_dict(64, 8)
    # D[1::3] = dl.dct_dict(64, 8)
    # D[2::3] = dl.dct_dict(64, 8)
    # np.savetxt('dct8_ds64.txt', D)
    # D = dl.dct_dict(64,12)
    # print(D.shape)
    # exit()

    image = io.imread(fname)
    if image.shape[-1] == 4:
        image = color.rgba2rgb(image)
    img_train = color.rgb2ycbcr(image) / 255.
    # img_train = image/255.
    # Duse = color.ycbcr2rgb(D)

    nl, nc, _= img_train.shape
    img4 = img_train
    t0 = time.monotonic()
    Ps = []
    for i in range(0, nl, m11):
        for j in range(0, nc, m22):
            if i + m11 <= nl and j + m22 <= nc:
                p = img4[i:(i+m11), j:(j+m22)]
                Ps.append(p)

    Ps = np.array(Ps).reshape(len(Ps), -1).T

    psnrs = []

    for i in range(1, int(Ps.shape[0])):
        mask = np.zeros(Ps.shape[0])
        choose = np.random.choice(Ps.shape[0], i, replace=False)
        # print(mask.shape)
        mask[choose] = 1
        # Dnew = np.asfortranarray((D * np.outer(mask, np.ones((1, D.shape[1])))),np.float)
        Dnew = np.asfortranarray(Duse[choose, :])
        Psnew = np.asfortranarray(Ps[choose, :])
        # res = np.array(dl.omp_mask(Ps, mask, D, 10, n_threads=4))
        res = spams.lasso(Psnew, Dnew, lambda1=0.000001, L=14).toarray()
        res = D.dot(res).T
        img1 = np.vstack((spl.reshape(-1,m11,m22,3).transpose(1,0,2,3).reshape(m11,-1,3) for spl in np.vsplit(res, int(nl / m11))))
        imgrgb = color.ycbcr2rgb(img1*255).clip(0,1)
        # imgrgb = (img1+1)/2
        psnrs.append(compare_psnr(image, imgrgb))
        print("%d: %.3f" % (i, psnrs[-1]))
    # dl.ps
    plt.subplot(2,1,1)
    plt.plot(psnrs, 'o-')

    print(imgrgb.min(), imgrgb.max())
    plt.subplot(2,1,2)
    plt.imshow(imgrgb)
    plt.show()


