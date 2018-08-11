import numpy as np
import glob
from spams import omp, ompMask, lasso, lassoMask
import time
from ksvd import KSVD_Encode
from skimage import io


def clip(img):
    img = np.minimum(np.ones(img.shape), img)
    img = np.maximum(np.zeros(img.shape), img)
    return img

def recover_same_kron(p, kronprod, mask):
    # y = np.take(p, mask, axis=1).T
    # p[:,mask] = 0.0
    y = p.T
    print(y.shape)
    # # y = np.take(p, mask)
    # phikron = np.take(kronprod, mask, axis=0)
    phikron = kronprod.T
    # phikron /= np.linalg.norm(phikron, axis=0)
    # print(np.linalg.norm(phikron, axis=0))
    print(phikron.shape)
    
    # _,sx,_ = lasso_path(phikron, y, eps=0.001, n_alphas=1, return_n_iter=False)
    # print(y.shape)
    # print(msk.shape)
    # print((mask != 0))
    msk = np.array([mask for _ in range(y.shape[1])]).T
    # sx = ompMask(np.asfortranarray(y), np.asfortranarray(phikron), np.asfortranarray((msk != 0)), eps=0.01, return_reg_path=False).toarray()
    # sx = omp(np.asfortranarray(y), np.asfortranarray(phikron), eps=0.03).toarray()
    # sx = KSVD_Encode(np.asfortranarray(y), np.asfortranarray(phikron), 10)
    # sx = lasso(np.asfortranarray(y), np.asfortranarray(phikron), lambda1=0.03, return_reg_path=False).toarray()
    sx = lassoMask(np.asfortranarray(y), np.asfortranarray(phikron), np.asfortranarray((msk != 0)), lambda1=0.03).toarray()
    print(sx.shape)
    # sx = orthogonal_mp_gram(phikron.T.dot(phikron), phikron.T.dot(y))
    # print(sx.shape)
    # sx = orthogonal_mp(phikron, y, tol=0.0001)
    newp = np.matmul(sx.T, kronprod)
    return newp


m11, m22 = 12, 12


if __name__ == "__main__":
    from skimage.measure import compare_psnr

    kronprod = np.loadtxt('ksvdmean12.txt')

    dir_images = "/home/eduardo/Imagens/*.png"
    images = glob.glob(dir_images)

    for i, img in enumerate(images):
        print(i, img)
    imgidx = int(input("Escolha do id da imagem: "))

    img_train = io.imread(images[imgidx], as_grey=True)

    io.imshow(img_train)
    io.show()

    # img_train = img_train / 255.0
    nl, nc = img_train.shape
    img2 = img_train.copy()
    img2[1::2,::2] = 0.0
    img2[::2,1::2] = 0.0
    print(np.argwhere(img2.ravel()).shape, img2.ravel().shape)
    t0 = time.monotonic()
    img4 = img2.copy()
    # img4[2:-1:2, 1:-1:2] = (img4[2:-1:2, :-2:2] + img4[2:-1:2, 2::2] + img4[1:-2:2, 1:-1:2] + img4[3::2, 1:-1:2]) / 4.0
    # img4[1:-1:2, 2:-1:2] = (img4[1:-1:2, 1:-2:2] + img4[1:-1:2, 3::2] + img4[:-2:2, 2:-1:2] + img4[2::2, 2:-1:2]) / 4.0
    # img4[0, 1:-1:2] = (img4[0, :-2:2] + img4[0, 2::2] + img4[1, 1:-1:2]) / 3.0
    # img4[-1, 1:-1:2] = (img4[-1, :-2:2] + img4[-1, 2::2] + img4[-2, 1:-1:2]) / 3.0
    # img4[1:-1:2, 0] = (img4[1:-1:2, 1] + img4[:-2:2, 0] + img4[2::2, 0]) / 3.0
    # img4[2:-1:2, -1] = (img4[2:-1:2, -2] + img4[1:-2:2, -1] + img4[3::2, -1]) / 3.0
    img4[::2, 1:-1:2] = (img4[::2, :-2:2] + img4[::2, 2::2]) / 2.0
    if img4.shape[1] % 2 == 1: img4[::2,-1] = img4[::2,-2]
    img4[1:-1:2, ::2] = (img4[:-2:2,::2] + img4[2::2,::2]) / 2.0
    if img4.shape[0] % 2 == 1: img4[-1,::2] = img4[-2,::2]

    #### Define a Máscara ####
    mask = np.ones((m11,m22),dtype=int)
    mask[1:-1:2,2:-1:2] = 0
    mask[2:-1:2,1:-1:2] = 0
    # mask = np.argwhere(mask.ravel()).ravel()
    mask = mask.reshape(m11*m22)
    ##########################


    Ps = []
    for i in range(0, nl, m11):
        for j in range(0, nc, m22):
            if i + m11 <= nl and j + m22 <= nc:
                p = img4[i:(i+m11), j:(j+m22)]
                Ps.append(p)

    tdivide = time.monotonic()
    print("Tempo para dividir: %.2f" % (tdivide - t0))

    nps = np.array([p.ravel() for p in Ps])
    tantes = time.monotonic()
    Ps = recover_same_kron(nps, kronprod, mask)
    count = 0
    img1 = img_train.copy()
    for i in range(0, nl, m11):
        for j in range(0, nc, m22):
            if i + m11 <= nl and j + m22 <= nc:
                img1[i:(i+m11), j:(j+m22)] = Ps[count].reshape(m11,m22)
                count += 1



    tdepois = time.monotonic()
    print("Tempo de processamento: %.2f" % (tdepois - tantes))

    img1 = clip(img1)

    io.imshow_collection([img1, img4])
    io.show()
    print(compare_psnr(img_train, img1))
    print(compare_psnr(img_train, img4))
    exit()