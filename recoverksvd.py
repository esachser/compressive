import numpy as np
import glob
from spams import omp, ompMask, lasso, lassoMask, somp, l1L2BCD, cd
import spams
import time
from ksvd import KSVD_Encode
from skimage import io
from sklearn.feature_extraction import image
from sklearn.linear_model import lasso_path, orthogonal_mp
from scipy.fftpack import dct
import matplotlib.pyplot as plt


from joblib import Parallel, delayed
import multiprocessing
from threading import Thread

num_cores = multiprocessing.cpu_count()
par = Parallel(n_jobs=num_cores)
    

def clip(img):
    img = np.minimum(np.ones(img.shape), img)
    img = np.maximum(np.zeros(img.shape), img)
    return img

def recover_same_kron(p, kronprod):
    y = p.T
    print(y.shape, y.dtype)
    phikron = kronprod.T
    print(phikron.shape)
    sx = spams.omp(np.asfortranarray(y), np.asfortranarray(phikron), eps=0.001, L=6)
    # sx = orthogonal_mp(phikron, y, tol=0.01)
    # spams.ssp.save_npz('image.npz', (sx).astype(np.float16))
    arr = sx.toarray()
    ids = np.argwhere(arr)
    id0 = ids[:,0].astype(np.uint8)
    id1 = ids[:,1].astype(np.uint16)
    vs = arr[arr != 0.0]
    # vs2 = np.log2(vs - vs.min() + 1)1616
    # print(vs3.max(), vs3.min(), vs3.mean()
    # print(vs2[np.abs(vs2) < 0.1].shape)
    srtd = np.argsort(vs)[::-1]
    vs4 = vs[srtd]
    idd0 = id0[srtd]
    idd1 = id1[srtd]
    # vs3 = dct(vs4, norm='ortho')
    # plt.plot(np.log1p(np.abs(vs4)))
    # print(np.argmin(np.abs(vs4)))
    # plt.plot(np.abs(vs4))
    # plt.show()
    # np.savez_compressed('img.npz', i0=id0, i1=id1, vs=vs3, mptp=[minval, ptp, (mxinit / mxend)], sh=sx.shape)
    print(sx.count_nonzero() / y.ravel().shape[0])
    return sx
    # newp = np.matmul(sx.toarray().T, kronprod)
    # return newp


m11, m22 = 8,8


if __name__ == "__main__":
    from skimage.measure import compare_psnr

    Dr = np.loadtxt('ksvd8_r.txt')
    Dg = np.loadtxt('ksvd8_g.txt')
    Db = np.loadtxt('ksvd8_b.txt')

    dir_images = "/home/eduardo/Imagens/*.png"
    images = glob.glob(dir_images)

    for i, img in enumerate(images):
        print(i, img)
    imgidx = int(input("Escolha do id da imagem: "))

    # img_train = io.imread(images[imgidx], as_grey=True)
    img_train = io.imread(images[imgidx])[:,:,:3]
    nl, nc, _= img_train.shape
    ml = nl % m11
    mc = nc % m22
    img_train = img_train[:(nl - ml), :(nc - mc), :].astype(float) / 255.0

    io.imshow(img_train)
    io.show()

    nl, nc, _= img_train.shape
    img4 = img_train
    t0 = time.monotonic()
    Ps = []
    for i in range(0, nl, m11):
        for j in range(0, nc, m22):
            if i + m11 <= nl and j + m22 <= nc:
                p = img4[i:(i+m11), j:(j+m22)]
                Ps.append(p)

    tdivide = time.monotonic()
    print("Tempo para dividir: %.2f" % (tdivide - t0))

    Ps = np.array(Ps).transpose(0,3,1,2)
    tantes = time.monotonic()
    sr = recover_same_kron(Ps[:,0].reshape(Ps.shape[0],-1), Dr)
    sg = recover_same_kron(Ps[:,1].reshape(Ps.shape[0],-1), Dg)
    sb = recover_same_kron(Ps[:,2].reshape(Ps.shape[0],-1), Db)


    tdepois = time.monotonic()
    print("Tempo de processamento: %.2f" % (tdepois - tantes))

    i0 = []
    i1 = []
    sh = sr.shape
    vs = []
    dptps = []
    for sx in (sr, sg, sb):    
        ids = sx.nonzero()
        arr = sx.toarray()
        arr = arr[arr!=0.0]
        id0 = ids[0].astype(np.uint8)
        id1 = ids[1].astype(np.uint16)
        srtd = np.argsort(arr)[::-1]
        vs4 = arr[srtd]
        # plt.plot(vs4)
        idd0 = id0[srtd]
        idd1 = id1[srtd]
        print(idd0)
        print(idd1)
        vs4 = np.abs(vs4)
        ptp = np.ptp(vs4)
        vs4u = (vs4*255.0/ptp).astype(np.uint8)
        d = vs4u.argmin()
        i0.append(idd0)
        i1.append(idd1)
        vs.append(vs4u)
        dptps.append((d, ptp))

    np.savez_compressed('img.npz', i0=i0, i1=i1, vs=vs, ptp=dptps, sh=sh)

    # plt.plot(vs[0])
    # plt.plot(vs[1])
    # plt.plot(vs[2])
    # plt.plot(vs4*255/ptp)
    # plt.show()


    Ps[:,0] = sr.toarray().T.dot(Dr).reshape(-1, m11, m22)
    Ps[:,1] = sg.toarray().T.dot(Dg).reshape(-1, m11, m22)
    Ps[:,2] = sb.toarray().T.dot(Db).reshape(-1, m11, m22)
    Ps = Ps.transpose(0,2,3,1)
    # Ps = np.asarray(par(delayed(recover_same_kron)(ps, d) for ps, d in 
    #                 [(Ps[:,0].reshape(Ps.shape[0],-1), Dr), (Ps[:,1].reshape(Ps.shape[0],-1), Dg), (Ps[:,2].reshape(Ps.shape[0],-1), Db)]))
    # Ps = Ps.reshape(3, -1, m11, m22).transpose(1,2,3,0)
    count = 0
    img1 = img_train.copy()
    for i in range(0, nl, m11):
        for j in range(0, nc, m22):
            if i + m11 <= nl and j + m22 <= nc:
                img1[i:(i+m11), j:(j+m22)] = Ps[count].reshape(m11,m22,-1)
                count += 1
    # img1 = image.reconstruct_from_patches_2d(Ps.reshape(Ps.shape[0], m11, m22), img_train.shape)


    img1 = clip(img1)

    # io.imshow_collection([img1, img4])
    io.imshow(img1)
    io.imsave('generated.png', img1)
    io.imsave('image.jpeg', img4)
    io.show()
    print(compare_psnr(img_train, img1))
    # print(compare_psnr(img_train, img4))
    exit()