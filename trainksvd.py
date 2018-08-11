from ksvd import KSVD
import numpy as np
import glob
from skimage import io
from sklearn.feature_extraction import image

dict_size = 256
target_sparsity = 10
patch_size = (12,12)

if __name__ == "__main__":
    dir_images = "/home/eduardo/Imagens/*.png"
    images = glob.glob(dir_images)
    for i, img in enumerate(images):
        print(i, img)
    imgidx = int(input("Escolha do id da imagem: "))

    img_train = io.imread(images[imgidx], as_grey=True)
    io.imshow(img_train)
    io.show()

    patches = image.extract_patches_2d(img_train, patch_size, max_patches=10000)
    X = patches.reshape(patches.shape[0], -1)
    print(X.shape)
    input()

    D, Gamma = KSVD(X, dict_size, target_sparsity, 5000,
                    print_interval = 100,
                    enable_printing = True, enable_threading = True)

    np.savetxt('ksvdmean12.txt', D)