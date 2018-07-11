# Implementacao do paper


import numpy as np
import pickle
from sklearn.linear_model import Lasso as Lasso
from spgl1 import spg_bpdn
import glob
import time
import cv2
from joblib import Parallel, delayed
import multiprocessing

def generate_random_square_matrices(k, shape):
    return [np.random.random(shape) for i in range(k)]

def eliminamenores(S, m1, m2, eliminar):
    s = np.absolute(S)
    el = np.argpartition(s.ravel(), eliminar)[:eliminar]
    S[el] = 0.0
    return S.reshape(m1,m2)

def elimina(S, k, m1, m2, eliminar):
    return np.apply_along_axis(eliminamenores, 1, S.reshape(k, m1*m2), m1, m2, eliminar)

def train_data(P, k, m1, m2, t, upbeta, initial_beta, beta_increment, tolerance):# error = 0.0001
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
    S = np.zeros((nt, k, m1, m2))
    Ps = np.array(P)
    while continuar:
        # Calcula matriz S
        t0 = time.clock()
        for i in range(nt):
            S[i] = np.matmul(np.matmul(U.transpose(0,2,1), Ps[i]), V)

        t1 = time.clock()
        # Tenho que eliminar os (m1*m2 - t) menores valores, colocando-os como 0
        # S = np.apply_along_axis(elimina, 1, S.reshape(nt, k*m1*m2), k, m1, m2, eliminar)
        for i in range(nt):
            S[i] = np.apply_along_axis(eliminamenores, 1, S[i].reshape(k, m1*m2), m1, m2, eliminar)

        t2  = time.clock()

        # Atualizacao em U[a] e V[a]
        for a in range(k):
            z0 = np.sum(np.matmul(
                np.matmul(
                    np.multiply(Ps.transpose(1,2,0), M[:,a]).transpose(2,0,1),
                    V[a]),
                S[:,a].transpose(0,2,1)), axis=0)

            z1 = np.sum(np.matmul(
                np.matmul(
                    np.multiply(Ps.transpose(2,1,0), M[:,a]).transpose(2,0,1),
                    U[a]),
                S[:,a]), axis=0)

            svdU = np.linalg.svd(z0)
            svdV = np.linalg.svd(z1)
            U[a] = np.matmul(svdU[0], svdU[2])
            V[a] = np.matmul(svdV[0], svdV[2])

            

        t3 = time.clock()
        lastM = np.copy(M)

        # Atualizacao de M
        # npmusv = np.linalg.norm((Ps - np.matmul(np.matmul(U, S), V.transpose(0,2,1))
        #                          .transpose(1,0,2,3)).transpose(1,0,2,3),
        #                         axis=(2,3))

        # vss = np.exp((-beta) * npmusv * npmusv).T
        # soma = vss.sum(axis=0)
        # M = (vss / soma).T
        # print(vss.shape)
        for i in range(nt):
            npmusv2 = np.sum((Ps[i] - np.matmul(np.matmul(U, S[i]), V.transpose(0,2,1)))**2, axis=(1,2))
            vs = np.nan_to_num(np.exp((-beta) * npmusv2))
            soma = vs.sum()
            # if i==0:
            #     print(P[i])
            #     print(S[i])
            #     print(npmusv2)
            #     print(vs)
            M[i] = vs
            if soma > 0: M[i] = M[i] / soma
        # print(np.abs(M - Maux).max())

        t4 = time.clock()
        print("Tempos:", t1-t0, t2-t1, t3-t2, t4-t3)
        # print(M)
        merror = np.abs(M - lastM).max()
        print("%f %.3f" % (merror, beta))
        if merror < error:
            beta += beta_increment
            beta_increment += 1.0
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
        
        continuar = not np.alltrue(test)
        print()

    return zip(U,V)


k = 32
m11 = 12
m22 = 12
t = 20
m = 32

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
    UV = train_data(Ps, k, m11, m22, t, 0.003, 0.1, 1,  0.01)

    kronprod = [np.kron(U,V) for U, V in UV]

    with open("kronprod3.pkl", 'wb') as fp:
        pickle.dump(kronprod, fp)

    print ("Geradas bases")
    exit()


def compute_best_index(p, t, s, kron, m1, m2):
    k = len(kron)
    e = np.array([np.infty for a in range(k)])
    z = np.ones(k)
    for a in range(k):
        x = np.dot(kron[a].T, p)

        while (z[a] <= s) and (e[a] > t):
            # Eliminar m1*m2 - z[a]
            eliminar = int(m1 * m2 - z[a])
            y = x.copy()
            el = np.argpartition(y, eliminar)[:eliminar]
            y[el] = 0.0

            # print y
            e[a] = np.linalg.norm(p - np.dot(kron[a], y))**2.0
            z[a] += 1

    a = np.argmin(z) if np.min(z)!=(s+1) else np.argmin(e)
    # print(a)
    return a

def recover(p, kronprod, remover):
    m1, m2 = p.shape
    a = compute_best_index(p.ravel(), 0.0001, 5, kronprod, m1, m2)
    phi = np.identity(m1 * m2)
    mask = np.random.choice(range(m1 * m2), m1 * m2 - remover, replace = False)
    phi = phi[mask, ...]
    pnew = p.reshape(m1*m2,1)
    y = pnew.T.flat[mask]
    y2 = np.zeros(pnew.shape)
    y2[mask] = pnew[mask]
    y = np.expand_dims(y, axis=1) 
    phikron = np.dot(phi, kronprod[a])
    sx,_,_,_ = spg_bpdn(phikron, y.ravel(), 0.1)
    newp = np.dot(kronprod[a], sx).reshape(m1, m2)
    return newp, y2.reshape(m1,m2)



def testa():
    from skimage.measure import compare_psnr

    with open('kronmaps.pkl', 'rb') as fp:
        kronprod = pickle.load(fp)


    dir_images = "/home/eduardo/Imagens/*.png"
    images = glob.glob(dir_images)

    for i, img in enumerate(images):
        print(i, img)
    imgidx = int(input("Escolha do id da imagem: "))

    img_train = cv2.cvtColor(cv2.imread(images[imgidx]), cv2.COLOR_BGR2GRAY)
    cv2.imshow("Imagem escolhida", img_train)
    cv2.waitKey()
    cv2.destroyAllWindows()

    t0 = time.clock()
    img_train = img_train / 255.0
    nl, nc = img_train.shape
    Ps = []
    for i in range(0, nl, m11):
        for j in range(0, nc, m22):
            if i + m11 <= nl and j + m22 <= nc:
                Ps.append(img_train[i:(i+m11), j:(j+m22)])
    tdivide = time.clock()
    print("Tempo para dividir: %.2f" % (tdivide - t0))

    Psmais = []
    tantes = time.clock()
    num_cores = multiprocessing.cpu_count()
    print("Cores: %d" % num_cores)
    func = lambda p: recover(p, kronprod, m)
    results = Parallel(n_jobs=num_cores)(delayed(func)(p) for p in Ps)
    # r = [recover(p, kronprod) for p in Ps]
    Ps, Psmais = zip(*results)
    tdepois = time.clock()
    print("Tempo de processamento: %.2f" % (tdepois - tantes))
    count = 0
    
    img2 = img_train.copy()
    img1 = img_train.copy()
    for i in range(0, nl, m11):
        for j in range(0, nc, m22):
            if i + m11 <= nl and j + m22 <= nc:
                img1[i:(i+m11), j:(j+m22)] = Ps[count]
                img2[i:(i+m11), j:(j+m22)] = Psmais[count]
                count += 1

    img3 = cv2.medianBlur(img2.copy().astype("float32"), 3)
            
    cv2.imshow("Imagem Recuperada", img1)
    cv2.imshow("Imagem inicial", img2)
    cv2.imshow("Median Blured", img3)
    cv2.waitKey()
    cv2.destroyAllWindows()
    print(compare_psnr(img_train, img1))
    print(compare_psnr(img_train, img3))
    exit()


# treina()
testa()


