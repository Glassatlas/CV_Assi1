#!/usr/bin/env python
# coding: utf-8


from matplotlib import image as img
from matplotlib import pyplot as plt
from numpy import asarray
import numpy as np
import glob, os
import cv2
cwd = os.getcwd()
os.chdir(cwd + "/Generated Histogram Data")



# Load all histograms 

faces_2x2 = []
cars_2x2 = []

faces_1x2 = []
cars_1x2 = []

faces_4x4 = []
cars_4x4 = []
 
for file in glob.glob("*.npy"):
    if '2x2' in file:
        print(file)
        if 'face' in file:
            faces_2x2.append(np.load(file).ravel().astype('float32')) #switch t
        elif 'car' in file:
            cars_2x2.append(np.load(file).ravel().astype('float32'))
    elif '1x2' in file:
        print(file)
        if 'face' in file:
            faces_1x2.append(np.load(file).ravel().astype('float32'))
        elif 'car' in file:
            cars_1x2.append(np.load(file).ravel().astype('float32'))
    elif '4x4' in file:
        print(file)
        if 'face' in file:
            faces_4x4.append(np.load(file).ravel().astype('float32'))
        elif 'car' in file:
            cars_4x4.append(np.load(file).ravel().astype('float32'))


print("Decreased 1x2 window\n")
compare1 = cv2.compareHist(cars_1x2[0], cars_1x2[1], cv2.HISTCMP_CORREL)

compare2 = cv2.compareHist(cars_1x2[0], cars_1x2[2], cv2.HISTCMP_CORREL)

compare3 = cv2.compareHist(cars_1x2[1], cars_1x2[2], cv2.HISTCMP_CORREL)

print('Average cars histogram correlation in 1x2 window:\n', (compare1 + compare2 + compare3)/3)

compare1 = cv2.compareHist(faces_1x2[0], faces_1x2[1], cv2.HISTCMP_CORREL)

compare2 = cv2.compareHist(faces_1x2[0], faces_1x2[2], cv2.HISTCMP_CORREL)

compare3 = cv2.compareHist(faces_1x2[1], faces_1x2[2], cv2.HISTCMP_CORREL)

print('Average faces histogram correlation in 1x2 window:\n', (compare1 + compare2 + compare3)/3)

list = []
for i in range(3):
    for t in range(3):
        list.append(cv2.compareHist(cars_1x2[i], faces_1x2[t], cv2.HISTCMP_CORREL))
        
print('Average faces vs cars histogram correlation in 1x2 window:\n', sum(list)/len(list))



print("2x2 window\n")
compare1 = cv2.compareHist(cars_2x2[0], cars_2x2[1], cv2.HISTCMP_CORREL)

compare2 = cv2.compareHist(cars_2x2[0], cars_2x2[2], cv2.HISTCMP_CORREL)

compare3 = cv2.compareHist(cars_2x2[1], cars_2x2[2], cv2.HISTCMP_CORREL)

print('Average cars histogram correlation in 2x2 window:\n', (compare1 + compare2 + compare3)/3)

compare1 = cv2.compareHist(faces_2x2[0], faces_2x2[1], cv2.HISTCMP_CORREL)

compare2 = cv2.compareHist(faces_2x2[0], faces_2x2[2], cv2.HISTCMP_CORREL)

compare3 = cv2.compareHist(faces_2x2[1], faces_2x2[2], cv2.HISTCMP_CORREL)

print('Average faces histogram correlation in 2x2 window:\n', (compare1 + compare2 + compare3)/3)

list = []
for i in range(3):
    for t in range(3):
        list.append(cv2.compareHist(cars_2x2[i], faces_2x2[t], cv2.HISTCMP_CORREL))
        
print('Average faces vs cars histogram correlation in 2x2 window:\n', sum(list)/len(list))



print("Increased 4x4 window\n")
compare1 = cv2.compareHist(cars_4x4[0], cars_4x4[1], cv2.HISTCMP_CORREL)

compare2 = cv2.compareHist(cars_4x4[0], cars_4x4[2], cv2.HISTCMP_CORREL)

compare3 = cv2.compareHist(cars_4x4[1], cars_4x4[2], cv2.HISTCMP_CORREL)

print('Average cars histogram correlation in 4x4 window:\n', (compare1 + compare2 + compare3)/3)

compare1 = cv2.compareHist(faces_4x4[0], faces_4x4[1], cv2.HISTCMP_CORREL)

compare2 = cv2.compareHist(faces_4x4[0], faces_4x4[2], cv2.HISTCMP_CORREL)

compare3 = cv2.compareHist(faces_4x4[1], faces_4x4[2], cv2.HISTCMP_CORREL)

print('Average faces histogram correlation in 4x4 window:\n', (compare1 + compare2 + compare3)/3)

list = []
for i in range(3):
    for t in range(3):
        list.append(cv2.compareHist(cars_4x4[i], faces_4x4[t], cv2.HISTCMP_CORREL))
        
print('Average faces vs cars histogram correlation in 4x4 window:\n', sum(list)/len(list))


