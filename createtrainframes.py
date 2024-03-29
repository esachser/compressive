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


vnames = ['tennis_sif.y4m', 'stefan_sif.y4m']

dirtrain = os.path.abspath('../trainframes/')

# Quantos segundos entre cada aquisição
secinterval = 1

def main():
    removefiles(dirtrain)
    for videoname in vnames:
        video = '../Videos/' + videoname
        cap = cv2.VideoCapture(video)
        cnt = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        interval = fps*secinterval
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT) // interval

        while True:
            actual = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, actual + interval)
            ret, frame = cap.read()
            cnt+=1
            if not ret: break
            cv2.imwrite(os.path.join(dirtrain, '%s_%d.png' % (videoname.split('.')[0],cnt)), frame)
            print("%d/%d frames generated" % (cnt, total))

if __name__ == '__main__':
    main()