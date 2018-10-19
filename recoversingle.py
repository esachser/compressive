import numpy as np
import glob
import spams
import time
from skimage import io, color
from sklearn.feature_extraction import image
from sklearn.linear_model import orthogonal_mp_gram
import matplotlib.pyplot as plt

import dictlearn as dl

from joblib import Parallel, delayed
import multiprocessing
from threading import Thread

num_cores = multiprocessing.cpu_count()
par = Parallel(n_jobs=num_cores)
    

def recover_same_kron(p, kronprod, L):
    y = p.T
    # print(y.shape, y.dtype)
    phikron = kronprod.T
    print(phikron.shape, y.shape)
    sx = spams.omp(np.asfortranarray(y, dtype=np.float32), np.asfortranarray(phikron, dtype=np.float32), eps=0.001, L=L, numThreads=-1)
    # print(sx.count_nonzero() / y.ravel().shape[0])
    # sx = orthogonal_mp_gram(kronprod.dot(kronprod.T), kronprod.dot(y), L)
    return sx


m11, m22 = 8,8
sparsity = 16

if __name__ == "__main__":
    from skimage.measure import compare_psnr

    D = np.loadtxt('dltrainfiles/dl8_ycbcr_ds64_720pmoria.txt')
    # D = np.loadtxt('dl8_rgb_ds192.txt')
    # D = io.imread('dict.png').astype(float) / 65535
    # dl.visualize_dictionary(D, 16, 8)

    dir_images = "/home/eduardo/Imagens/*.png"
    images = glob.glob(dir_images)

    for i, img in enumerate(images):
        print(i, img)
    imgidx = int(input("Escolha do id da imagem: "))

    # img_train = io.imread(images[imgidx], as_grey=True)
    image = io.imread(images[imgidx])
    if image.shape[-1] == 4:
        image = color.rgba2rgb(image)

    img_train = color.rgb2ycbcr(image) / 255.
    # img_train = io.imread(images[imgidx])[:,:,:3]
    nl, nc, _= img_train.shape
    ml = nl % m11
    mc = nc % m22
    img_train = img_train[:(nl - ml), :(nc - mc), :].astype(float)
    # print(color.rgba2rgb(io.imread(images[imgidx]))[0,0])

    # io.imshow(color.yuv2rgb(img_train))
    # io.show()

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
    print("Tempo para dividir: %.3f" % (tdivide - t0))

    Ps = np.array(Ps)
    tantes = time.monotonic()
    # print(Ps.shape)
    print(img_train[0,0])
    print(Ps[0,0,0])
    Ps = Ps.reshape(Ps.shape[0],-1)
    print(Ps[0,0:3])
    s = recover_same_kron(Ps, D, sparsity)

    tdepois = time.monotonic()
    print("Tempo de processamento: %.3f" % (tdepois - tantes))
    # exit()

    ## Pensar num jeito de salvar
    spams.ssp.save_npz('image.npz', s.astype(np.float16))
    ss = s.transpose()
    vmin = ss.min()
    vptp = ss.max() - vmin
    vs = np.zeros((ss.shape[0], sparsity), dtype=np.uint16)
    ids = np.zeros_like(vs, dtype=np.uint8)
    for i in range(ss.shape[0]):
        nz = ss.getrow(i).toarray()
        idss = nz.nonzero()
        nz = nz[nz!=0.0]
        vs[i,:nz.shape[0]] = np.round((nz - vmin) * 65535.0 / vptp)
        ids[i,:nz.shape[0]] = idss[1]+1
    
    vals = np.unique(vs)
    print(vals.shape)
    # print(ids[:100])
    # print(vs.shape, ids.shape)
    np.savez_compressed('img.npz', ids=ids, vs=vs, mptp=[vmin, vptp], sh=ss.shape)
    # s = s.toarray()
    # Ps = D.T.dot(s).T
    Ps = s.transpose().dot(D)# .reshape(Ps.shape[0], m11, m22, -1)
    print(Ps[0,0:3])
    print(Ps.shape)
    # Ps = np.asarray(par(delayed(recover_same_kron)(ps, d) for ps, d in 
    #                 [(Ps[:,0].reshape(Ps.shape[0],-1), Dr), (Ps[:,1].reshape(Ps.shape[0],-1), Dg), (Ps[:,2].reshape(Ps.shape[0],-1), Db)]))
    # Ps = Ps.reshape(3, -1, m11, m22).transpose(1,2,3,0)
    # count = 0
    # img1 = img_train.copy()
    # for i in range(0, nl, m11):
    #     for j in range(0, nc, m22):
    #         if i + m11 <= nl and j + m22 <= nc:
    #             img1[i:(i+m11), j:(j+m22)] = Ps[count].reshape(m11,m22,-1)
    #             count += 1

    t0 = time.monotonic()
    img1 = np.vstack((spl.reshape(-1,m11,m22,3).transpose(1,0,2,3).reshape(m11,-1,3) for spl in np.vsplit(Ps, int(nl / m11))))
    # img1 = np.hstack((spl for spl in np.vsplit(newp, int(nl / m11)))).reshape(-1,m11,m22,3)#.reshape(nl,nc,3)
    print(img1.shape)
    print("Recovery time: %f" % (time.monotonic() - t0))
    # img1 = image.reconstruct_from_patches_2d(Ps.reshape(Ps.shape[0], m11, m22), img_train.shape)


    imgrgb = color.ycbcr2rgb(img1 * 255).clip(0,1)
    # imgrgb = (img1 / 255.).clip(0,1)

    # io.imshow_collection([img1, img4])
    # io.imshow(imgrgb)
    io.imsave('generated.png', imgrgb)
    io.imsave('image.jpeg', color.ycbcr2rgb(img4  * 255).clip(-1,1))
    io.imsave('dict.png', D)
    # io.show()
    # print(compare_psnr(img_train, imgrgb))
    # print(compare_psnr(img_train, img1.clip(-1,1)))
    print(compare_psnr(color.ycbcr2rgb(img_train * 255).clip(0,1), imgrgb))
    # print(compare_psnr(img_train/255., imgrgb))
    # print(compare_psnr(img_train, img4))
    exit()
