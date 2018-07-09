# Implementacao do paper


import numpy as np
import pickle
from sporco.admm import bpdn
from sklearn.linear_model import Lasso as Lasso
from spgl1 import spg_bpdn
import glob
import time
import cv2

def generate_random_square_matrices(k, shape):
    return [np.random.random(shape) for i in range(k)]

def eliminamenores(S, m1, m2, eliminar):
    s = np.absolute(S)
    el = np.argpartition(s.flatten(), eliminar)[:eliminar]
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
            svdU = np.linalg.svd(z0)

            z1 = np.sum(np.matmul(
                np.matmul(
                    np.multiply(Ps.transpose(2,1,0), M[:,a]).transpose(2,0,1),
                    U[a]),
                S[:,a]), axis=0)
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
            npmusv = np.linalg.norm(Ps[i] - np.matmul(np.matmul(U, S[i]), V.transpose(0,2,1)), axis=(1,2))
            vs = np.exp((-beta) * npmusv * npmusv)
            soma = vs.sum()
            M[i] = vs / soma
        # print(np.abs(M - Maux).max())

        t4 = time.clock()
        print("Tempos:", t1-t0, t2-t1, t3-t2, t4-t3)
        # print(M)
        merror = np.abs(M - lastM).max()
        print(merror)
        if merror < error:
            beta += beta_increment
            print ("Beta Increment to {}!".format(beta))

        continuar = merror >= stop
        print()

    return zip(U,V)


k = 32
m11 = 12
m22 = 12
t = 32
m = 20

def treina():
    dir_images = "C:/Users/eduardo.sachser/Pictures/*.png"
    images = glob.glob(dir_images)
    imgidx = int(input(images))

    img_train = cv2.cvtColor(cv2.imread(images[imgidx]), cv2.COLOR_BGR2GRAY)
    cv2.imshow("Bla", img_train)
    cv2.waitKey()
    cv2.destroyAllWindows()

    img_train = img_train / 255.0

    # print img_train[0:12,0:12]

    nl, nc = img_train.shape

    Ps = []

    for i in range(0, nl, m11):
        for j in range(0, nc, m22):
            if i + m11 <= nl and j + m22 <= nc:
                Ps.append(img_train[i:(i+m11), j:(j+m22)])

    # Ps = generate_random_square_matrices(1000, (m11,m22))

    print(len(Ps))
    UV = train_data(Ps, k, m11, m22, t, 0.01, 1.0, 3.0,  0.0001)

    kronprod = [np.kron(U,V) for U, V in UV]

    with open("kronprod2.pkl", 'wb') as fp:
        pickle.dump(kronprod, fp)

    print ("Geradas bases")
    exit()


computeidx=1
def compute_best_index(p, t, s, kron, m1, m2):
    global computeidx
    k = len(kron)
    e = np.array([np.infty for a in range(k)])
    z = np.ones(k)
    print("Computing...", computeidx, "...", end=' ')
    computeidx += 1
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
    print(a)
    return a


def testa():
    with open('kronprod2.pkl', 'rb') as fp:
        kronprod = pickle.load(fp)

    # Ps = generate_random_square_matrices(20, (m11,m22))

    dir_images = "C:/Users/eduardo.sachser/Pictures/*.png"
    images = glob.glob(dir_images)
    # print(images)
    # exit()

    img_train = cv2.cvtColor(cv2.imread(images[-2]), cv2.COLOR_BGR2GRAY)
    cv2.imshow("Bla", img_train/255.0)
    cv2.waitKey()
    cv2.destroyAllWindows()

    img_train = img_train / 255.0
    nl, nc = img_train.shape
    Ps = []
    for i in range(0, nl, m11):
        for j in range(0, nc, m22):
            if i + m11 <= nl and j + m22 <= nc:
                Ps.append(img_train[i:(i+m11), j:(j+m22)])

    best_indexes = [compute_best_index(p.ravel(), 0.0001, 5, kronprod, m11, m22) for p in Ps]
    Psmais = []
    tantes = time.clock()
    for i in range(len(Ps)):
        a = best_indexes[i]

        # t0 = time.clock()
        phi = np.identity(m11 * m22)
        mask = sorted(np.random.choice(range(m11 * m22), m11 * m22 - m, replace = False))
        # phi[mask, mask] = 0
        phi = phi[mask, ...]
        # t1 = time.clock()
        # p = np.ndarray((m11 * m22,1), buffer = Ps[i].flatten())
        p = Ps[i].reshape(m11*m22,1)
        y = p.T.flat[mask]
        y2 = p.copy()
        y2[mask] = 0.0
        y = np.expand_dims(y, axis=1)   
        # y = np.dot(phi, p)
        phikron = np.dot(phi, kronprod[a])
        # print(y.shape)
        # t2 = time.clock()

        # b = bpdn.BPDN(phikron, y, 0.001)
        # x = b.solve()

        # sb = Lasso(0.001, normalize=False)
        # sb.fit(phikron, y)
        # sx = 10 * sb.coef_

        # sx = spg_bpdn(phikron, y, 0.001)
        # print(y.shape)
        sx,resid,grad,info = spg_bpdn(phikron, y.ravel(), 0.1)

        # print(x)
        # print(sx)
        # exit()
        newp = np.dot(kronprod[a], sx).reshape(m11, m22)
        # t3 = time.clock() 
        # newp = np.ndarray((m11,m22), buffer = np.dot(kronprod[a], x))
        Ps[i] = newp
        Psmais.append(y2.reshape(m11,m22))

    tdepois = time.clock()

    print("Tempo de processamento: %.2f" % (tdepois - tantes))
    count = 0
    img2 = img_train.copy()
    for i in range(0, nl, m11):
        for j in range(0, nc, m22):
            if i + m11 <= nl and j + m22 <= nc:
                img_train[i:(i+m11), j:(j+m22)] = Ps[count]
                img2[i:(i+m11), j:(j+m22)] = Psmais[count]
                count += 1
    
    cv2.imshow("Bla", img_train)
    cv2.imshow("Bla2", img2)
    cv2.waitKey()
    cv2.destroyAllWindows()
    exit()


# treina()
testa()


