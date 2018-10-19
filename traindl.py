import numpy as np
import glob
# import cv2
from skimage import color
import scipy.misc as io
# from sklearn.feature_extraction import image
# import gc
import spams
import dictlearn as dl

dict_size = 64
target_sparsity = 12
patch_size = (8,8)

if __name__ == "__main__":
    dir_images = "/home/eduardo/Documentos/compressive/trainframes/*"
    images = glob.glob(dir_images)
    # for i, img in enumerate(images):
    #     print(i, img)
    # imgidx = int(input("Escolha do id da imagem: "))

    # img_trains = [color.rgba2rgb(io.imread(img)) for img in images]
    img_trains = [io.imread(img) for img in images]
    # print(images[0])
    # print(img_trains[0][0,0])
    # input()
    # nl, nc, _= img_train.shape
    # ml = nl % patch_size[0]
    # mc = nc % patch_size[1]
    # img_train = img_train[:(nl - ml), :(nc - mc), :]

    # io.imshow(img_train)
    # io.show()
    # dl.visualize_dictionary(dl.dct_dict(dict_size, 4), 5, 4)
    # print(dl.dct_dict(dict_size, 4).shape)

    iniD = np.zeros((patch_size[0]*patch_size[1]*3, dict_size))
    iniD[::3] = dl.dct_dict(dict_size, patch_size[0])
    iniD[-2::-3] = dl.dct_dict(dict_size, patch_size[0])
    iniD[2::3] = dl.dct_dict(dict_size, patch_size[0])

    param = {'D':np.asfortranarray(iniD),
             'lambda1':target_sparsity,
             'iter':1000,
             'mode':3
    }

    # Treinando tudo junto
    # img_train = color.rgb2yuv(img_train)
    # patches = image.extract_patches_2d(color.rgb2yuv(img_trains[0]), patch_size)
    # patches = np.hstack((image.extract_patches_2d(color.rgb2yuv(img_train), patch_size)) for img_train in img_trains)
    patches = []
    for img_train in img_trains:
        # img_train = color.rgb2ycbcr(img_train) / 255.
        img_train = img_train / 255.
        for i in range(0, img_train.shape[0], patch_size[0]):
            for j in range(0, img_train.shape[1], patch_size[1]):
                if i + patch_size[0] <= img_train.shape[0] and j + patch_size[1] <= img_train.shape[1]:
                    p = img_train[i:(i+patch_size[0]), j:(j+patch_size[1])]
                    patches.append(p)
    patches = np.array(patches)
    print(patches.shape)
    X = patches.reshape(patches.shape[0], -1)[::1]
    print(X.max())
    print(X.shape, X.dtype)
    input()
    # X /= .01

    # p1 = np.vsplit(img_train, range(patch_size[0], img_train.shape[0], patch_size[0]))
    # p2 = []
    # for p in p1:
    #     p2.extend(np.hsplit(p, range(patch_size[1], img_train.shape[1], patch_size[1])))
    # X = np.vstack([p.reshape(1, -1) for p in p2])
    # print(X.shape, X.dtype)
    D = spams.trainDL(np.asfortranarray(X.T), **param).T
    # D = np.array(dl.ksvd(X.T, iniD, 1000, target_sparsity)).T
    print(D.shape)

    np.savetxt('dltrainfiles/dl8_rgb_ds64_720pBunny.txt', D)
