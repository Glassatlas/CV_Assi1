import cv2

import Hardcoded
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

Dataset='Dataset/DatasetC.mpg'

vid = cv2.VideoCapture(Dataset)


#set font for text
font = cv2.FONT_HERSHEY_SIMPLEX

#Parameters

# max distance for a point to be considered a belonging to a cluster
dst=280

#minimum number of points in a cluster to be considred an object
cluser_size=20000


first=True
nb_frames=0
count_per_frame=[]
while(vid.isOpened()):
    ret, frame = vid.read()

    if frame is not None:
        # to gray
        gray = Hardcoded.BGR2GRAY(frame)

        #To take previous
        if first:
            reference=np.zeros_like(gray)
            first=False

        else :
            #after reference is previous and previous will store present frame
            #reference=cv2.add(reference,gray)/counter
            reference = np.mean([reference,np.array(gray)], axis=0) #compute mean along row of the two matrix
            reference = reference.astype(np.uint8) #only positive value


        # saturated difference
        res=Hardcoded.diff(gray,reference)

       
        # Comment this section for fast video feed which does not exclude counting
        # count objects
        count = Hardcoded.count_obj2(res, dst, cluser_size)
        count_per_frame.append(count)
        cv2.putText(res, str(count), (10, 50), font, 1, (255, 255, 255), 2)
     

        #show
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
cv2.waitKey()
cv2.destroyAllWindows()