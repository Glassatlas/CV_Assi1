import cv2

import Hardcoded
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

Dataset='Dataset/DatasetC.mpg'

vid = cv2.VideoCapture(Dataset)

#set font for text
font = cv2.FONT_HERSHEY_SIMPLEX


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])




#Parameters

# max distance for a point to be considered a belonging to a cluster
dst=180

#minimum number of points in a cluster to be considred an object
cluser_size=5000



first=True
nb_frames=0 #number of frames
count_per_frame=[]
while(vid.isOpened()):
    ret, frame = vid.read()

    if frame is not None:
        nb_frames+=1 #frame count 
        # to gray
        gray = Hardcoded.BGR2GRAY(frame)

        #To take previous
        if first : #if first is true
            #at first frame the reference is zero
            reference,previous=np.zeros_like(gray),np.zeros_like(gray) #return 2 zeroes array 
            #1ref 1 previous
            first=False
        else :
            # after reference is previous, and previous will store present frame
            reference,previous=previous,gray

        # saturated difference
        res=Hardcoded.diff(gray,reference)


        
        #Comment this section for fast video feed, the count won't be done
        #count objects,
        count=Hardcoded.count_obj(res, dst, cluser_size)
        count_per_frame.append(count)
        cv2.putText(res, str(count), (10, 50), font, 1, (255, 255, 255), 2)

   
        # show
        cv2.imshow('frame', res)

    key=cv2.waitKey(1)
    if key & 0xFF == ord('q') or key == 27 or 'x' == chr(key & 255):
        break




#plotting
bars = [str(i+1) for i in range(len(count_per_frame))]

plt.bar(bars, count_per_frame)
# Displaying the bar plot
plt.show()



vid.release()
cv2.waitKey(0)
cv2.destroyAllWindows()