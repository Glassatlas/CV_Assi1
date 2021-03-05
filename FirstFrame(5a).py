import cv2
import Hardcoded
import numpy as np
from matplotlib import image as img
from matplotlib import pyplot as plt
from PIL import Image
from numpy import asarray
import numpy as np
import os
cwd = os.getcwd()


Dataset='Dataset/DatasetC.mpg'

vid = cv2.VideoCapture(Dataset)


first=True
while(vid.isOpened()):
    ret, frame = vid.read()

    if frame is not None:
        # to gray
        gray = Hardcoded.BGR2GRAY(frame)

        #To take only first frame
        if first :
            reference=gray #store first frame gray as ref
        first=False

        # saturated difference
        res=Hardcoded.diff(gray,reference)

        #show
        cv2.imshow('frame', res)

    key=cv2.waitKey(1)
    if key & 0xFF == ord('q') or key == 27 or 'x' == chr(key & 255):
        break
vid.release()
cv2.destroyAllWindows()


