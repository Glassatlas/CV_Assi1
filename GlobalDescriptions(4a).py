#!/usr/bin/env python
# coding: utf-8

from matplotlib import image as img
from matplotlib import pyplot as plt
from PIL import Image
from numpy import asarray
import numpy as np
import os
cwd = os.getcwd()



#Set
image_name = 'car-1'
horizontal_divide_count = 2
vertical_divide_count = 2

#Importing image
image_path = os.path.join(cwd, 'Dataset/DatasetA/' + image_name + '.jpg')
image_file = img.imread(image_path)

print(image_file.dtype)
print(image_file.shape)
plt.imshow(image_file)
plt.show()


# Converting to grayscale
grey_image = np.dot(image_file[...,:3], [0.2989, 0.5870, 0.1140])

plt.imshow(grey_image, cmap=plt.get_cmap('gray'))
print(grey_image.shape)
plt.show()

#Question 4a: Divide image into windows

#Division @ I seperate it into 4 window

window_height = int(256 / horizontal_divide_count) #divided by 2
window_width = int(256 / vertical_divide_count) #divided by 2
h, w = grey_image.shape  #retrns # of row and column
windows = grey_image.reshape(h//window_height, window_height, -1, window_width).swapaxes(1,2).reshape(-1, window_height, window_width)
#gives(2, 128,-1, 128)--> 128, 2 
#swap axes--> (2,2,128,128)
#reshape to (4,128,128)


#Display 
f, ax = plt.subplots(1,len(windows))

f.set_figheight(15)
f.set_figwidth(15)

for i in range(len(windows)):
    ax[i].imshow(windows[i], cmap=plt.get_cmap('gray'))
    ax[i].set_title('Window ' + str(i + 1))

print(windows[0].shape)
plt.show()

#Calculate LBP over windows

LBP_windows = [] #creart blank array

for index in range(len(windows)): #gives 4
    
    imgLBP = np.zeros_like(windows[index]) #same size as window
    neighbor = 3 #
    for ih in range(0,windows[index].shape[0] - neighbor): #go through matrix
        for iw in range(0,windows[index].shape[1] - neighbor): #go through matrix
            imgl          = windows[index][ih:ih+neighbor,iw:iw+neighbor] #image layer
            center       = imgl[1,1] 
            img01        = (imgl >= center)*1.0 #return true ==1
            img01_vector = img01.T.flatten() #Transpose flatten and turn to vector 
            
            img01_vector = np.delete(img01_vector,4) #since 4 is the centre
            
            where_img01_vector = np.where(img01_vector)[0] #get the index of 1
            if len(where_img01_vector) >= 1: #if there is 
                num = np.sum(2**where_img01_vector) #convert to decimal
            else:
                num = 0
            imgLBP[ih+1,iw+1] = num #
    LBP_windows.append(imgLBP) #list of 4 seperated window

#Display part
f, ax = plt.subplots(1,len(LBP_windows))

f.set_figheight(15)
f.set_figwidth(15)

for i in range(len(LBP_windows)):
    ax[i].imshow(LBP_windows[i], cmap=plt.get_cmap('gray'))
    ax[i].set_title('Window ' + str(i + 1) +' LBP')

print(LBP_windows[0].shape)
plt.show()


#Normalized histogram of windows

f, ax = plt.subplots(len(LBP_windows), constrained_layout=True)

histograms = [] #blank for histogram's'

for window_index in range(len(LBP_windows)): #window 0-4
    values = LBP_windows[window_index].flatten() #flatten each
    
    histogram = []
    
    for i in range(256):
        histogram.append(np.count_nonzero(values == i))  #return non zeroes where values == i
        
    histogram = np.asarray(histogram)

    # Normalized counts
    histogram = histogram / histogram.sum()
    
    histograms.append(histogram)
    
    ax[window_index].bar(range(256), histogram)
    ax[window_index].set_title('Window ' + str(window_index + 1) + ' normalized histogram')
    
f.set_figwidth(15)
f.set_figheight(10)
plt.show()

#Concatenate Histogram (Global Description)

concatenated_histogram = np.concatenate(histograms)

plt.style.use('default')
plt.bar(range(len(concatenated_histogram)), concatenated_histogram)
plt.title('Concatenated histogram')
plt.show()

#Save Histogram data

from tempfile import TemporaryFile
outfile = TemporaryFile()

file_name = image_name + '_' + str(horizontal_divide_count) + 'x' + str(vertical_divide_count) + '_histogram'

np.save(os.path.join(cwd + '\HistogramData', file_name), concatenated_histogram)


#Read Data

