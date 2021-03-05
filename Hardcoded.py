import cv2
import numpy as np


def BGR2GRAY (image):
    #Formula used : gray = 0.21R + 0.72G + 0.07B
    if image is not None :
        coef = np.array([[[0.07, 0.72, 0.21]]])
        gray = cv2.convertScaleAbs(np.sum(image * coef, axis=2))

        return(gray)


def diff(image,reference):
    #function to do pixel wise diff

    if image is not None and reference is not None :
        res=cv2.absdiff(image, reference)
        return res

def count_obj(image, dst_threshold, Number_of_elements):
    #clustering function to count the objects
    image=np.array(image)

    #get non zero indices
    non_zeros=np.nonzero(image)

    centers = []

    #number on non zero elements
    num_elts=len(non_zeros[0]) #return length of non zeroes indice(row)
    if num_elts == 0 : 
        return 0
    else :
        print(num_elts)
        #loop throug indices
        for i in range(0,num_elts-1,100):
            #print("i    :",i)
            point=(non_zeros[0][i],non_zeros[1][i])
            if len(centers)==0:
                center=point
                centers.append(([point],center,1))
            else :
                for k,seed in enumerate(centers) :
                    center=(seed[1][0],seed[1][1])
                    if abs(point[0]-center[0])<=dst_threshold and abs(point[1]-center[1])<=dst_threshold: #if less than threshold
                        #to append new point
                        temp=seed[0]
                        temp.append(point)

                        #calculate new center
                        center=np.mean(temp,axis=0) 

                        #number of point
                        nb=seed[2]
                        centers[k]=(temp,center,nb+1)
                    else :
                        center=point
                        centers.append(([point],point,1))

    return sum([1 for elt in centers if elt[3]>Number_of_elements])


def count_obj2(image, dst_threshold, Number_of_elements):
    #Fast clustering. Faster than the first but  weak with noise
    #if a point is close to one of the points encountered by dst, ignored it and add 1 to the cluster counter
    #if a point is not close to any previous point by dst consider it a new seed and append
    #consider only clusters with points > number_of_elements as object

    image=np.array(image)

    #get non zero indices
    non_zeros=np.nonzero(image)

    centers = []

    #number on non zero elements
    num_elts=len(non_zeros[0])
    if num_elts == 0 :
        return 0
    else :
        points=[(non_zeros[0][j],non_zeros[1][j]) for j in range(num_elts) ]
        # print("len  point :", len(points))
        points=list(set(points))
        # print("len  set of point :", len(points))
        # print(num_elts)
    
        for point in points:
            #print("i    :",i)
            # point=(non_zeros[0][i],non_zeros[1][i])
            #print(point)
            if len(centers)==0:
                center=point
                centers.append((point,1))
            else :
                belongs_to_cluster=False
                for k,seed in enumerate(centers) :
                    #print("seed :",seed)
                    center=(seed[0][0],seed[0][1])
                    #print(seed)
                    dst=np.sqrt((center[0]-point[0])**2+(center[1]-point[1])**2)
                    # if abs(point[0]-center[0])<=dst_threshold and abs(point[1]-center[1])<=dst_threshold:
                    if dst<=dst_threshold:
                        centers[k]=(centers[k][0],centers[k][1]+1)

                        belongs_to_cluster=True
                if not belongs_to_cluster:
                    centers.append((point,1))
    # for m in centers :
    #     print("centers ",m)

    return sum([1 for elt in centers if elt[1]>Number_of_elements])
    #return len(centers)




