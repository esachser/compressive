from ksvd import KSVD
import numpy as np
import glob
from skimage import io
from sklearn.feature_extraction import image
import gc
import spams

dict_size = 128
target_sparsity = 32
patch_size = (8,8)

if __name__ == "__main__":
    dir_images = "/home/eduardo/Imagens/*.png"
    images = glob.glob(dir_images)
    for i, img in enumerate(images):
        print(i, img)
    imgidx = int(input("Escolha do id da imagem: "))

    img_train = io.imread(images[imgidx])[:,:,:-1]
    io.imshow(img_train)
    io.show()

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
    patches = image.extract_patches_2d(img_train, patch_size)
    X = patches.reshape(patches.shape[0], -1)[::100]
    print(X.shape)

    D, Gamma = KSVD(X, dict_size, target_sparsity, 1000,
                    print_interval = 80,
                    enable_printing = True, enable_threading = True)

    np.savetxt('ksvd8_rgb_ds128.txt', D)