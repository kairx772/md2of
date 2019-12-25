# coding: utf8
import matplotlib.pyplot as plt
from VisualSti import GenerateVisualStimuli as gvs
from VisualSti import MotionDetection as md
from VisualSti import Reichardt as rh
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import os
import gc
import numpy as np
import sys
from scipy import signal
import time
import cv2


def basic_stimulus_null(inputfilen, outputfilen):
    with open(inputfilen, 'rb') as input:
        st = pickle.load(input).movie
    sw2 = rh.RH(inputframe = st[...,0])
    rharr = np.empty_like(st)
    for i in range(0,120):
        rharr[...,i] = sw2.run(st[...,120-i])
        print ('frame', i)
    rharr = rharr*0.05
    plt.xlabel('frame')
    plt.ylabel('response')
    plt.plot(st[40,40,1:120], color='black', label='Image')
    plt.plot(rharr[40,40,1:120], color='blue', label='Reichardt')
    plt.legend(loc='best')
    plt.savefig(outputfilen)
    plt.clf()

def basic_stimulus(inputfilen, outputfilen):
    with open(inputfilen, 'rb') as input:
        st = pickle.load(input).movie
    sw2 = rh.RH(inputframe = st[...,0])
    rharr = np.empty_like(st)
    for i in range(0,120):
        rharr[...,i] = sw2.run(st[...,i])
        print ('frame', i)
    rharr = rharr*0.05
    plt.xlabel('frame')
    plt.ylabel('response')
    plt.plot(st[40,40,1:120], color='black', label='Image')
    plt.plot(rharr[40,40,1:120], color='blue', label='Reichardt')
    plt.legend(loc='best')
    plt.savefig(outputfilen)
    plt.clf()

def dispOpticalFlow(Image,Flow_h,Flow_v,Divisor,name):
    PictureShape = np.shape(Image)
    #determine number of quiver points there will be
    Imax = int(PictureShape[0]/Divisor)
    Jmax = int(PictureShape[1]/Divisor)
    #create a blank mask, on which lines will be drawn.
    mask = np.zeros_like(Image)
    for i in range(1, Imax):
        for j in range(1, Jmax):
            X1 = (i)*Divisor
            Y1 = (j)*Divisor
            X2 = int(X1 + Flow_h[X1,Y1])
            Y2 = int(Y1 + Flow_v[X1,Y1])
            X2 = np.clip(X2, 0, PictureShape[0])
            Y2 = np.clip(Y2, 0, PictureShape[1])
            #add all the lines to the mask
            mask = cv2.line(mask, (Y1,X1),(Y2,X2), [255, 255, 255], 1)
    #superpose lines onto image
    img = cv2.add(Image,mask)
    #print image
    cv2.imshow(name,img)
    return []




def main():
    pass



if __name__ == '__main__':

    foldername = './gabor/Yosemite/'
    img = cv2.imread(foldername+'frame07.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sw2_h = rh.RH(inputframe = img,
                A1_Lambda = 5.0,
                A2_Lambda = 5.0)
    sw2_v = rh.RH(inputframe = img,
                A1_theta = 0,
                A2_theta = 0,
                A1_Lambda = 5.0,
                A2_Lambda = 5.0)
    # rharr = np.empty_like(st)
    norm = mpl.colors.Normalize(vmin=-200, vmax=200)

    for frame in range(7,15):
        img = cv2.imread(foldername+'frame{0:02d}.png'.format(frame))
        img00 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('ss',img)
        k = cv2.waitKey(0) & 0xff

        if k == 27:
            break

        # dispOpticalFlow(img,sw2_h.run(img00)*0.001,sw2_v.run(img00)*0.001,100,'of00')

        # cmap = plt.cm.gray
        # imgplot = plt.imshow(sw2_h.run(img)*0.001,cmap = plt.cm.bwr , norm = norm)
        # imgplot = plt.imshow(sw2_v.run(img)*0.001,cmap = plt.cm.bwr , norm = norm)
        # plt.colorbar(imgplot)
        # plt.savefig(foldername+'RH_h/frame{0:02d}.png'.format(frame), dpi=100)
        # plt.clf()

        # plt.show()
        # print ('frame', i)