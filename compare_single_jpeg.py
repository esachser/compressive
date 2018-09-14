import numpy as np
import glob
from spams import omp, ompMask, lasso, lassoMask, somp, l1L2BCD, cd
import spams
import time
from skimage import io, color
from sklearn.feature_extraction import image
import matplotlib.pyplot as plt
from scipy.fftpack import idct
import matplotlib.pyplot as plt

m11, m22 = 8,8

if __name__ == '__main__':
    from skimage.measure import compare_psnr

    D = np.loadtxt('dltrain/dl8_yuv_ds96.txt')
    
    dir_images = "/home/eduardo/Imagens/*.png"
    images = glob.glob(dir_images)

    for i, img in enumerate(images):
        print(i, img)
    imgidx = int(input("Escolha do id da imagem: "))
    img_train = color.rgb2yuv(io.imread(images[imgidx])[:,:,:3])
    nl, nc, _= img_train.shape
    ml = nl % m11
    mc = nc % m22
    img_train = img_train[:(nl - ml), :(nc - mc), :].astype(float)

    nl, nc, _= img_train.shape

    img2 = io.imread('image.jpeg').astype(np.float)[:,:,:3] / 255.0

    # Recupera sx
    
    # sx = spams.ssp.load_npz('image.npz')

    # np.savez_compressed('img.npz', i0=id0, i1=id1, vs=vs, mptp=[minval, ptp], sh=sx.shape)
    # with np.load('img.npz') as f:
    #     id0, id1, vs, mptp, sh = f['i0'], f['i1'], f['vs'], f['mptp'], f['sh']
    # print(vs.shape)
    # vs = vs.astype(np.float)
    # # vs = np.exp2(vs) + mptp[0] - 1
    # vs = (vs*mptp[1] / 255.0) + mptp[0]
    # vs[:int(len(vs)/50)] *= mptp[2]
    # vs = idct(vs, norm='ortho')
    # sx = spams.ssp.csc_matrix((vs, (id0, id1)), shape=sh)
    with np.load('img.npz') as f:
        ids, vss, sh, mptp = f['ids'], f['vs'], f['sh'], f['mptp']

    t0 = time.monotonic()
    vmin = mptp[0]
    vptp = mptp[1]

    s = np.zeros(sh)
    for i in range(s.shape[0]):
        vs = vss[i]
        idss = ids[i, ids[i]!=0] - 1
        s[i, idss] = (vs[:idss.shape[0]] * vptp / 65535) + vmin
    
    newp = s.dot(D).reshape(-1, m11, m22, 3)
    # print(newp.shape)
    count = 0
    img1 = img_train.copy()
    for i in range(0, nl, m11):
        for j in range(0, nc, m22):
            if i + m11 <= nl and j + m22 <= nc:
                img1[i:(i+m11), j:(j+m22)] = newp[count]
                count += 1
    print("Recovery time: %f" % (time.monotonic() - t0))


    imgrgb = color.yuv2rgb(img1).clip(0,1)

    print(compare_psnr(img_train, img1))
    print(compare_psnr(img_train, color.rgb2yuv(img2)))
    io.imsave('rcv.png', imgrgb)
    # io.imshow(imgrgb)
    # io.show()







