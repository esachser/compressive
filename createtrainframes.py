import numpy as np
import cv2

import os
import shutil

def removefiles(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

video = '../Videos/big_buck_bunny_720p24.y4m'
dirtrain = os.path.abspath('../trainframes/')

# Quantos segundos entre cada aquisição
secinterval = 10

def main():
    cap = cv2.VideoCapture(video)
    cnt = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = fps*secinterval
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) / interval
    
    removefiles(dirtrain)

    while True:
        # for _ in range(int(interval)): cap.grab()
        actual = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, actual + interval)
        ret, frame = cap.read()
        cnt+=1
        if not ret: break
        cv2.imwrite(os.path.join(dirtrain, 'frame_%d.png' % (cnt)), frame)
        print("%d/%d frames generated" % (cnt, total))

if __name__ == '__main__':
    main()