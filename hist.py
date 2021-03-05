import numpy as np
import cv2
import os
from matplotlib import pyplot as plt


def calc_histogram(image):
    # calculate histogram bin for each color channel
    hist = []
    for i in range(3):
        im = image[:, :, i]
        h = np.zeros((256,))
        for bin in range(256):
            h[bin] = np.sum(im == bin)
        hist.append(h)
    return np.array(hist)


def visualize_save(hist,idx):
    # visualize and save histogram by matplotlib
    color = ('b','g','r')
    fig = plt.figure()
    for i, col in enumerate(color):
        plt.plot(hist[i], color=col)
        plt.xlim([0,256])
    plt.show()
    fig.savefig(f'output/hist{idx}.png')

def getFrame(video_path, start_frame, end_frame):
    #save frames to the file path
    cap = cv2.VideoCapture(video_path)
    total_frames = cap.get(7) #use this to check how many frames there are in the video
    for i in range(start_frame, end_frame+1): #if you want everything then just set it to total_frames+1
        cap.set(1, i)
        ret, frame = cap.read()
        cv2.imwrite(f'output/frame{i}.jpg', frame)
    

def construct_histogram(video_path):
    # construct histogram sequences from video path
    cap = cv2.VideoCapture(video_path)
    total_frames = cap.get(7)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    histseq = []
    for i in range(int(total_frames+1)):
        ret, frame = cap.read() #obtain return value and get frame in the video
        hist = calc_histogram(frame)
        #visualize_save(hist, i)  # visualize histogram
        histseq.append(hist)

    return histseq, width * height * 3

q, width * height * 3


def calc_intersection_sequences(histseq):
    # calculate the intersection between consecutive frames
    def calc_intersection(hist1, hist2):
        hist_min = np.minimum(hist1, hist2)
        sum = np.sum(hist_min)
        return sum

    intersection = []
    for i in range(len(histseq) - 1):
        val = calc_intersection(histseq[i], histseq[i+1])
        intersection.append(val)
    intersection = np.array(intersection)
    return intersection


def draw_intersection(intersection, img_size):
    # visualize and save intersection by matplotlib
    def draw_inter(inter, output_path):
        fig = plt.figure()
        plt.plot(inter)
        plt.show()
        fig.savefig(output_path)

    norm_inter = intersection / img_size
    draw_inter(intersection, 'intersection.png')
    draw_inter(norm_inter, 'norm_intersection.png')


if __name__ == '__main__':
    # video path
    input_path = '../Dataset/DatasetB.avi'
    histseq, img_size = construct_histogram(input_path)
    intersection = calc_intersection_sequences(histseq)
    draw_intersection(intersection, img_size)

