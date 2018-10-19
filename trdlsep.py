import numpy as np
import glob
# import cv2
from skimage import color
import scipy.misc as io
# from sklearn.feature_extraction import image
# import gc
import spams

dict_size = 48
target_sparsity = 6
patch_size = (8,8)

if __name__ == "__main__":
    dir_images = "/home/eduardo/Documentos/compressive/trainframes/*"
    images = glob.glob(dir_images)
    # for i, img in enumerate(images):
    #     print(i, img)
    # imgidx = int(input("Escolha do id da imagem: "))

    # img_trains = [color.rgba2rgb(io.imread(img)) for img in images]
    img_trains = [io.imread(img) for img in images]
    print(images[0])
    print(img_trains[0][0,0])
    input()
    # nl, nc, _= img_train.shape
    # ml = nl % patch_size[0]
    # mc = nc % patch_size[1]
    # img_train = img_train[:(nl - ml), :(nc - mc), :]

    # io.imshow(img_train)
    # io.show()

    param = {'K':dict_size,
             'lambda1':0.01,
             'iter':1000
    }

    patchesy = []
    patchescb = []
    patchescr = []
    for img_train in img_trains:
        img_train = color.rgb2ycbcr(img_train) / 255.
        for i in range(0, img_train.shape[0], patch_size[0]):
            for j in range(0, img_train.shape[1], patch_size[1]):
                if i + patch_size[0] <= img_train.shape[0] and j + patch_size[1] <= img_train.shape[1]:
                    p = img_train[i:(i+patch_size[0]), j:(j+patch_size[1])]
                    if(np.linalg.norm(p[:,:,0])>0.0): patchesy.append(p[:,:,0])
                    if(np.linalg.norm(p[:,:,1])>0.0): patchescb.append(p[:,:,1])
                    if(np.linalg.norm(p[:,:,2])>0.0): patchescr.append(p[:,:,2])
    patchesy = np.array(patchesy)
    patchescb = np.array(patchescb)
    patchescr = np.array(patchescr)


    print(patchesy.shape)
    X = patchesy.reshape(patchesy.shape[0], -1)[::1]
    print(X.max())
    print(X.shape, X.dtype)
    D = spams.trainDL(np.asfortranarray(X.T), **param).T
    print(D.shape)
    np.savetxt('dltrainfiles/dldiff8_y_ds48_720pBunny.txt', D)

    print(patchescb.shape)
    X = patchescb.reshape(patchescb.shape[0], -1)[::1]
    print(X.max())
    print(X.shape, X.dtype)
    D = spams.trainDL(np.asfortranarray(X.T), **param).T
    print(D.shape)
    np.savetxt('dltrainfiles/dldiff8_cb_ds48_720pBunny.txt', D)

    print(patchescr.shape)
    X = patchescr.reshape(patchescr.shape[0], -1)[::1]
    print(X.max())
    print(X.shape, X.dtype)
    D = spams.trainDL(np.asfortranarray(X.T), **param).T
    print(D.shape)
    np.savetxt('dltrainfiles/dldiff8_cr_ds48_720pBunny.txt', D)
