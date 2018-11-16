import numpy as np
import glob
# import cv2
from skimage import color
import scipy.misc as io
from sklearn.feature_extraction import image
# import gc
import spams
import dictlearn as dl

dict_size = 16
target_sparsity = 5
patch_size = (4,4)

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
    iniD[::3] = 1.0
    # iniD[-2::-3] = dl.dct_dict(dict_size, patch_size[0])
    iniD[2::3] = 0.3

    param = {'D':np.asfortranarray(iniD),
             'model': {'A': np.asfortranarray(np.zeros((dict_size, dict_size))), 
                       'B': np.asfortranarray(np.zeros_like(iniD)), 
                       'iter':0},
             'lambda1':target_sparsity,
             'iter':500,
             'mode':3,
             'return_model': True
    }

    # Treinando tudo junto
    # img_train = color.rgb2yuv(img_train)
    # patches = image.extract_patches_2d(color.rgb2yuv(img_trains[0]), patch_size)
    # patches = np.hstack((image.extract_patches_2d(color.rgb2yuv(img_train), patch_size)) for img_train in img_trains)
    patches = []
    for i in np.random.choice(len(img_trains), len(img_trains), replace=False):
        img_train = img_trains[i]
        if img_train.shape[-1] == 4:
            img_train = color.rgba2rgb(img_train)
        img_train = img_train.astype(np.float)
        if img_train.max() > 1.0: img_train /= 255
        # for i in range(0, img_train.shape[0], patch_size[0]):
        #     for j in range(0, img_train.shape[1], patch_size[1]):
        #         if i + patch_size[0] <= img_train.shape[0] and j + patch_size[1] <= img_train.shape[1]:
        #             p = img_train[i:(i+patch_size[0]), j:(j+patch_size[1])]
        #             patches.append(p)
        # patches = np.array(patches)
        patches = image.extract_patches_2d(img_train, patch_size)
        print(patches.shape)        
        X = patches.reshape(patches.shape[0], -1)[::1]
        print(X.shape)
        # D = spams.trainDL(np.asfortranarray(X.T), **param)
        D, model = spams.trainDL(np.asfortranarray(X.T), **param)
        param['D'] = D
        param['model'] = model

    # patches = np.array(patches)
    # print(patches.shape)
    # X = patches.reshape(patches.shape[0], -1)[::1]
    # m = np.mean(X)
    # m = 0.0
    # s = np.std(X)
    # s = 1.0
    # X -= m
    # X /= s
    # print(X.max())
    # print(X.shape, X.dtype)
    # input()
    # X /= .01

    # p1 = np.vsplit(img_train, range(patch_size[0], img_train.shape[0], patch_size[0]))
    # p2 = []
    # for p in p1:
    #     p2.extend(np.hsplit(p, range(patch_size[1], img_train.shape[1], patch_size[1])))
    # X = np.vstack([p.reshape(1, -1) for p in p2])
    # print(X.shape, X.dtype)
    # D, model = spams.trainDL(np.asfortranarray(X.T), **param)
    D = D.T
    # D = np.array(dl.ksvd(X.T, iniD, 1000, target_sparsity)).T
    print(D.shape)
    print(model['A'].shape)
    print(model['B'].shape)
    print(model['iter'])

    np.savetxt('dltrainfiles/dl4_rgb_ds16_cartoon.txt', D)
