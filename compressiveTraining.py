# Implementacao do paper


import numpy as np
import scipy.sparse as sp
import pickle
from sklearn.linear_model import lasso_path, orthogonal_mp, Lasso, orthogonal_mp_gram
from spgl1 import spg_bpdn
from spams import lasso, omp, ompMask, lassoMask
import glob
import time
import cv2
from joblib import Parallel, delayed
import multiprocessing
from threading import Thread

num_cores = multiprocessing.cpu_count()
# num_cores = 1
par = Parallel(n_jobs=num_cores)
parar = False

def generate_random_square_matrices(k, shape):
    return [np.random.random(shape) for i in range(k)]

def eliminamenores(S, m1, m2, eliminar):
    s = np.absolute(S)
    el = np.argpartition(s.ravel(), eliminar)[:eliminar]
    S[el] = 0.0
    return S.reshape(m1,m2)

def elimina(S, k, m1, m2, eliminar):
    return np.apply_along_axis(eliminamenores, 1, S.reshape(k, m1*m2), m1, m2, eliminar)

def processa_S(U, V, m1, m2, eliminar, p):
    s = np.matmul(np.matmul(U.transpose(0,2,1), p), V).reshape(k, m1*m2)
    # Tenho que eliminar os (m1*m2 - t) menores valores, colocando-os como 0
    return np.apply_along_axis(eliminamenores, 1, s, m1, m2, eliminar)

def update_UV(Ps, u, v, s, m):
    z0 = np.sum(np.matmul(
        np.matmul(
            np.multiply(Ps.transpose(1,2,0), m).transpose(2,0,1),
            v),
        s.transpose(0,2,1)), axis=0)

    z1 = np.sum(np.matmul(
        np.matmul(
            np.multiply(Ps.transpose(2,1,0), m).transpose(2,0,1),
            u),
        s), axis=0)

    svdU = np.linalg.svd(z0)
    svdV = np.linalg.svd(z1)
    return np.matmul(svdU[0], svdU[2]), np.matmul(svdV[0], svdV[2])

def update_M(U, V, beta, p, s):
    npmusv2 = np.sum((p - np.matmul(np.matmul(U, s), V.transpose(0,2,1)))**2, axis=(1,2))
    vs = np.nan_to_num(np.exp((-beta) * npmusv2))
    soma = vs.sum()
    ret = vs
    if soma > 0: 
        ret /= soma
    return ret

def train_data(P, k, m1, m2, t, upbeta, initial_beta, beta_increment, tolerance):# error = 0.0001
    global parar
    # stop = error * error
    error = upbeta
    beta = initial_beta
    stop = tolerance
    eliminar = m1*m2 - t
    nt = len(P)

    Urandom = generate_random_square_matrices(k, (m1, m1))
    U = np.array([np.linalg.qr(m)[0]  for m in Urandom])

    Vrandom = generate_random_square_matrices(k, (m2, m2))
    V = np.array([np.linalg.qr(m)[0]  for m in Vrandom])

    M = np.zeros((nt, k))
    M.fill(1.0 / k)
    # return zip(U,V)

    continuar = True
    # Aqui daremos voltas
    S = np.zeros((nt, k,m1,m2))
    # S = sp.DOK((nt, k, m1, m2))
    # S = np.empty((nt, k), dtype=sp.coo.coo_matrix)
    # S = da.zeros((nt, k, m1, m2))
    Ps = np.array(P)
    prosd = delayed(processa_S)
    upuvd = delayed(update_UV)
    upmd = delayed(update_M)
    while continuar:
        # Calcula matriz S
        t0 = time.monotonic()
        # for i in range(nt):
        #     S[i] = func(Ps[i])
        S = np.array(par(prosd(U, V, m1, m2, eliminar, p) for p in Ps))

        t1 = time.monotonic()

        # Atualizacao em U[a] e V[a]
        # for a in range(k):
        #     U[a], V[a] = update_UV(Ps, U[a], V[a], S[:,a], M[:,a])
        uv = par(upuvd(Ps, U[a], V[a], S[:,a], M[:,a]) for a in range(k))
        u, v = zip(*uv)
        U, V = np.array(u), np.array(v)

        t2  = time.monotonic()

        lastM = np.copy(M)
        # Atualizacao de M
        # for i in range(nt):
        #     M[i] = update_M(U,V,beta,Ps[i],S[i])
        M = np.array(par(upmd(U, V, beta, Ps[i], S[i]) for i in range(nt)))
        # print(np.abs(M - Maux).max())

        t3 = time.monotonic()
        # t4 = time.monotonic()
        print("Tempos: %.3f %.3f %.3f" % (t1-t0, t2-t1, t3-t2))
        # print(M)
        merror = np.abs(M - lastM).max()
        print("%.8f %.3f %.3f" % (merror, beta, beta_increment))
        if merror < error:
            # beta += beta_increment
            # beta_increment *= 1.5
            if beta_increment > 1.01: beta_increment -= 0.01
            if beta > 500000 : beta_increment = 1.001
            beta *= beta_increment
            print ("Beta Increment to %.3f!" % beta)

        # Testa M, deve ser próximo a 0 ou a 1
        test0 = np.isclose(M, 0, atol=stop)
        test1 = np.isclose(M, 1.0, atol=stop)
        test = np.logical_or(test0, test1)
        print(M[0, test[0] == False])
        # print(M[-1, test[-1] == False])
        print(M[test[:,:] == False])
        print(test[test[:,:] == False].shape)
        print()
        
        continuar = not(np.alltrue(test) or parar)
        print()

    return zip(U,V)


k = 18
m11 = 12
m22 = 12
t = 10
m = 72

def treina():
    dir_images = "/home/eduardo/Imagens/*.png"
    images = glob.glob(dir_images)
    for i, img in enumerate(images):
        print(i, img)
    imgidx = int(input("Escolha do id da imagem: "))

    # img_train = cv2.color.rgb2gray(cv2.io.imread(images[imgidx]))
    img_train = cv2.cvtColor(cv2.imread(images[imgidx]), cv2.COLOR_BGR2GRAY)
    cv2.imshow("Imagem escolhida", img_train)
    cv2.waitKey()
    cv2.destroyAllWindows()

    img_train = img_train / 255.0

    # print img_train[0:12,0:12]

    nl, nc = img_train.shape
    Ps = []
    for i in range(0, nl, m11):
        for j in range(0, nc, m22):
            if i + m11 <= nl and j + m22 <= nc:
                aux = img_train[i:(i+m11), j:(j+m22)].ravel()
                if not np.allclose(aux, aux[0]):
                    Ps.append(img_train[i:(i+m11), j:(j+m22)])

    # Ps = generate_random_square_matrices(1000, (m11,m22))

    print(len(Ps))
    UV = train_data(Ps[::4], k, m11, m22, t, 0.0007, 0.5, 1.5,  0.01)

    kronprod = [np.kron(U,V) for U, V in UV]

    with open("kronprod3.pkl", 'wb') as fp:
        pickle.dump(kronprod, fp)

    print ("Geradas bases")
    exit()


def eliminamenores2(S, eliminar):
    s = np.absolute(S)
    el = np.argpartition(s, eliminar)[:eliminar]
    S[el] = 0.0
    return S

def elimina2(S, k, eliminar):
    return np.apply_along_axis(eliminamenores2, 1, S, eliminar)

def compute_best_index(p, t, s, kron):
    k = len(kron)
    psize = p.shape[0]
    xs = np.matmul(kron.transpose(0,2,1), p)
    xs = elimina(xs, k, psize, 1, int(psize-s))
    # for a in range(k):
    #     x = np.dot(kron[a].T, p)
    #     idxs = np.argpartition(x, int(psize-s-1))[int(psize-s):]
    #     idxs = idxs[np.argsort(x[idxs])]
    #     y = np.zeros(x.shape)
    #     while (z[a] <= s) and (e[a] > t):
    #         y[idxs[int(-z[a])]] = x[idxs[int(-z[a])]]
    #         e[a] = np.sum(np.square(p - np.matmul(kron[a], y)))
    #         z[a] += 1
            # t0 = time.monotonic()
            # print(np.matmul(kron[a], y).shape)
            # t1 = time.monotonic()
        # print(a, "T:", e[a])
    # print(np.square(p - np.matmul(kron, xs).reshape(k,psize)).shape)
    err = np.sum(np.square(p - np.matmul(kron, xs).reshape(k,psize)), axis=1)

    # a = np.argmin(z) if np.min(z)!=(s+1) else np.argmin(e)
    # print(a, np.argmin(err))
    return np.argmin(err)

def recover(p, kronprod, mask):
    # t0 = time.monotonic()
    # m1, m2 = p.shape
    # a = 18
    a = compute_best_index(p.ravel(), 0.001, 20, kronprod)
    # print(a)
    # t1 = time.monotonic()
    # mask = np.random.choice(range(m1 * m2), m1 * m2 - remover, replace = False)
    # y = np.take(p, mask, axis=1).T
    # y = np.take(p, mask)
    y = np.expand_dims(p, 1)
    # print(y.shape)
    # phikron = np.take(kronprod[a], mask, axis=0)
    phikron = kronprod[a]
    # np.savetxt('phik.txt', phikron)
    # y = np.take(p.ravel(),mask)
    # phikron = np.take(kronprod[a], mask, axis=0)#[mask, ...]
    # t2 = time.monotonic()
    # sx,_,_,_ = spg_bpdn(phikron, y.ravel(), 0.1)
    # sx = lasso(np.asfortranarray(y.ravel()), np.asfortranarray(phikron), lambda1=0.02, return_reg_path=False).toarray()
    # _,sx,_ = lasso_path(phikron, y, eps=0.0001, n_alphas=1, return_n_iter=False)
    # print(y.shape)
    msk = np.expand_dims((mask != 0),0)
    # sx = ompMask(np.asfortranarray(y), np.asfortranarray(phikron), np.asfortranarray(msk), lambda1=0.01, return_reg_path=False).toarray()
    # sx = omp(np.asfortranarray(y), np.asfortranarray(phikron), eps=0.03, return_reg_path=False).toarray()
    # sx = lassoMask(np.asfortranarray(y), np.asfortranarray(phikron), np.asfortranarray(msk), lambda1=0.01).toarray()
    sx = orthogonal_mp(phikron, y, tol=0.03)
    # print(sx.shape)
    # t3 = time.monotonic()
    # t4 = time.monotonic()
    newp = np.matmul(kronprod[a], sx)
    # print(newp.shape)
    # print("Tempos: %.5f %.5f %.5f" % (t1 - t0, t2 - t1, t3 - t2), a)
    return newp

def recover_same_kron(p, kronprod, mask):
    # y = np.take(p, mask, axis=1).T
    # p[:,mask] = 0.0
    y = p.T
    # print(y.shape)
    # # y = np.take(p, mask)
    # phikron = np.take(kronprod, mask, axis=0)
    phikron = kronprod
    # phikron /= np.linalg.norm(phikron, axis=0)
    # print(np.linalg.norm(phikron, axis=0))
    
    # _,sx,_ = lasso_path(phikron, y, eps=0.001, n_alphas=1, return_n_iter=False)
    # print(y.shape)
    msk = np.array([mask for _ in range(y.shape[1])]).T
    # print(msk.shape)
    # print((mask != 0))
    # sx = ompMask(np.asfortranarray(y), np.asfortranarray(phikron), np.asfortranarray((msk != 0)), eps=0.01, return_reg_path=False).toarray()
    sx = omp(np.asfortranarray(y), np.asfortranarray(phikron), eps=0.03, return_reg_path=False).toarray()
    # sx = lasso(np.asfortranarray(y), np.asfortranarray(phikron), lambda1=0.01, return_reg_path=False).toarray()
    # sx = lassoMask(np.asfortranarray(y), np.asfortranarray(phikron), np.asfortranarray((msk != 0)), lambda1=0.01).toarray()
    # sx = orthogonal_mp_gram(phikron.T.dot(phikron), phikron.T.dot(y))
    # print(sx.shape)
    # sx = orthogonal_mp(phikron, y, tol=0.0001)
    newp = np.matmul(kronprod, sx).T
    return newp

from recovering import recover_cython, recover_patches

m11, m22 = 12, 12

def testa():
    from skimage.measure import compare_psnr

    with open('kr12.pkl', 'rb') as fp:
        kronprod = np.array(pickle.load(fp))


    dir_images = "/home/eduardo/Imagens/*.png"
    images = glob.glob(dir_images)

    for i, img in enumerate(images):
        print(i, img)
    imgidx = int(input("Escolha do id da imagem: "))

    img_train = cv2.cvtColor(cv2.imread(images[imgidx]), cv2.COLOR_BGR2GRAY)
    cv2.imshow("Imagem escolhida", img_train)
    cv2.waitKey()
    cv2.destroyAllWindows()

    img_train = img_train / 255.0
    nl, nc = img_train.shape
    img2 = img_train.copy()
    img2[1::2,:] = 0.0
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
    img4[::2, 1:-1:2] = (img4[::2, :-3:2] + img4[::2, 2::2]) / 2.0
    if img4.shape[1] % 2 == 1: img4[::2,-1] = img4[::2,-2]
    img4[1:-1:2, :] = (img4[:-2:2,:] + img4[2::2,:]) / 2.0
    if img4.shape[0] % 2 == 1: img4[-1,:] = img4[-2,:]

    #### Define a Máscara ####
    mask = np.ones((m11,m22),dtype=int)
    # mask[2:-1:2,2:-1] = 0
    # mask[1:-1:2,1:-1:2] = 0
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
    bestidxs = np.apply_along_axis(compute_best_index, 1, nps, 0.001, 20, kronprod)
    best = np.bincount(bestidxs).argmax()
    print(set(bestidxs), best)
    tantes = time.monotonic()
    # func = lambda p: recover(p, kronprod, m)
    # Ps = par(delayed(recover_cython)(p.ravel(), kronprod, mask) for p in Ps)
    Ps = recover_same_kron(nps, kronprod[best], mask)
    # Ps = [recover(p.ravel(), kronprod, mask) for p in Ps]
    # Ps = [recover_cython(p.ravel(), kronprod, mask).reshape(m11,m22) for p in Ps]
    # for i in range(len(Ps)):
    #     # t0 = time.monotonic()
    #     Ps[i] = recover_cython(Ps[i].ravel(), kronprod, mask).reshape(m11,m22)
    #     # print(time.monotonic() - t0)
    # recover_patches(Ps, kronprod, mask)
    # Ps, Psmais = zip(*results)
    count = 0


    img1 = img_train.copy()
    for i in range(0, nl, m11):
        for j in range(0, nc, m22):
            if i + m11 <= nl and j + m22 <= nc:
                img1[i:(i+m11), j:(j+m22)] = Ps[count].reshape(m11,m22)
                count += 1



    tdepois = time.monotonic()
    print("Tempo de processamento: %.2f" % (tdepois - tantes))
    # img3 = cv2.GaussianBlur(img2.copy().astype("float32"), (3,3), 0)

    cv2.imshow("Imagem Recuperada", img1)
    # cv2.imshow("Imagem inicial", img2)
    cv2.imshow("Media", img4)
    cv2.waitKey()
    cv2.destroyAllWindows()
    print(compare_psnr(img_train, img1))
    print(compare_psnr(img_train, img4))
    exit()

def testaparada():
    global parar
    while not parar:
        if str(input()) == 'q':
            parar = True

# Thread(daemon=True, target=testaparada).start()
# treina()
testa()

