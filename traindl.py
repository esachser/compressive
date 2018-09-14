import numpy as np
import glob
from skimage import io, color
from sklearn.feature_extraction import image
import gc
import spams

dict_size = 96
target_sparsity = 24
patch_size = (8,8)

if __name__ == "__main__":
    dir_images = "/home/eduardo/Imagens/*.png"
    images = glob.glob(dir_images)
    for i, img in enumerate(images):
        print(i, img)
    imgidx = int(input("Escolha do id da imagem: "))

    img_train = io.imread(images[imgidx])[:,:,:-1]
    nl, nc, _= img_train.shape
    ml = nl % patch_size[0]
    mc = nc % patch_size[1]
    img_train = img_train[:(nl - ml), :(nc - mc), :]
    # io.imshow(img_train)
    # io.show()

    param = {'K':dict_size,
             'lambda1':0.01,
             'iter':1000
    }
    # Treinar R, G e B separadamente
    # ### R ###
    # patches = image.extract_patches_2d(img_train[:,:,0], patch_size)
    # X = patches.reshape(patches.shape[0], -1)[::100]
    # print(X.shape)

    # D, Gamma = KSVD(X, dict_size, target_sparsity, 1000,
    #                 print_interval = 80,
    #                 enable_printing = True, enable_threading = True)

    # np.savetxt('ksvd8_r_ds32.txt', D)
    # gc.collect()

    # ### G ###
    # patches = image.extract_patches_2d(img_train[:,:,1], patch_size)
    # X = patches.reshape(patches.shape[0], -1)[::100]
    # print(X.shape)

    # D, Gamma = KSVD(X, dict_size, target_sparsity, 1000,
    #                 print_interval = 80,
    #                 enable_printing = True, enable_threading = True)

    # np.savetxt('ksvd8_g_ds32.txt', D)
    # gc.collect()

    # ### B ###
    # patches = image.extract_patches_2d(img_train[:,:,2], patch_size)
    # X = patches.reshape(patches.shape[0], -1)[::100]
    # print(X.shape)

    # D, Gamma = KSVD(X, dict_size, target_sparsity, 1000,
    #                 print_interval = 80,
    #                 enable_printing = True, enable_threading = True)

    # Treinando tudo junto
    img_train = color.rgb2yuv(img_train)
    patches = image.extract_patches_2d(img_train, patch_size)
    X = patches.reshape(patches.shape[0], -1)[::1].astype(float)
    print(X.max())
    print(X.shape, X.dtype)
    # X /= .01

    # p1 = np.vsplit(img_train, range(patch_size[0], img_train.shape[0], patch_size[0]))
    # p2 = []
    # for p in p1:
    #     p2.extend(np.hsplit(p, range(patch_size[1], img_train.shape[1], patch_size[1])))
    # X = np.vstack([p.reshape(1, -1) for p in p2])
    # print(X.shape, X.dtype)
    D = spams.trainDL(np.asfortranarray(X.T), **param).T
    print(D.shape)

    np.savetxt('dltrain/dl8_yuv_ds96.txt', D)