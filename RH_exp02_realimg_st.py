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
import math

from os.path import splitext
import numpy as np

# from skimage import transform,data

def draw_hsv(h, w, fx, fy):
    # h, w = flow.shape[:2]
    # fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


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



def main():
    pass



if __name__ == '__main__':

    foldername = './gabor/Army/'
    img = cv2.imread(foldername+'frame07.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fheight, fwidth = img.shape
    sw2 = rh.RH(inputframe = img,
                A1_Lambda = 15.0,
                A2_Lambda = 15.0)
    sw3 = rh.RH(inputframe = img,
                A1_theta = 0,
                A2_theta = 0,
                A1_Lambda = 15.0,
                A2_Lambda = 15.0)
    # rharr = np.empty_like(st)
    norm = mpl.colors.Normalize(vmin=-400, vmax=400)
    # sw2.print_gabor('gh0', 'gh1')
    # sw3.print_gabor('gv0', 'gv1')

    for frame in range(7,15):
        img = cv2.imread(foldername+'frame{0:02d}.png'.format(frame))
        imgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        flowx = sw2.run(imgb)*0.00003
        # flowy = np.zeros_like(flowx)
        flowy = sw3.run(imgb)*0.00003
        # print (flowy[10, 10])
        # flowx = flowy = np.zeros_like(flowy)



        print (img.shape)

        cmap = plt.cm.gray
        # imgplot = plt.imshow(flowx,cmap = plt.cm.bwr , norm = norm)
        imgplot = plt.imshow(flowx,cmap = plt.cm.bwr , norm = norm)
        plt.colorbar(imgplot)
        plt.savefig('messi/Army/plot/fx/frame{0:02d}.png'.format(frame), dpi=100)
        plt.clf()

        imgplot = plt.imshow(flowy,cmap = plt.cm.bwr , norm = norm)
        plt.colorbar(imgplot)
        plt.savefig('messi/Army/plot/fy/frame{0:02d}.png'.format(frame), dpi=100)
        plt.clf()

        imgf = img
        imgf = cv2.resize(imgf, (imgf.shape[1]*5, imgf.shape[0]*5), interpolation=cv2.INTER_CUBIC)

        # transform.rescale(imgf, 10).shape

        # cv2.imshow('hsv optical flow', draw_hsv(img.shape[0],img.shape[1],flowx,flowy))

        ch = cv2.waitKey(5)
        if ch == 27:
            break

        for y in range(0, fheight, 8):
            for x in range(0, fwidth, 8):
                # print (flowx[x, y])
                if flowy[y, x] > 0:
                    cv2.arrowedLine(imgf, (x*5, y*5), (x*5+int(flowx[y, x]), y*5+int(flowy[y, x])), ( 0, 255, 0 ), thickness=2, shift=0)
                if flowy[y, x] < 0:
                    cv2.arrowedLine(imgf, (x*5, y*5), (x*5+int(flowx[y, x]), y*5+int(flowy[y, x])), ( 0, 255, 0), thickness=2, shift=0)
        
        # out.write(img)

        cv2.imwrite('./messi/Army/messigray{}.png'.format(frame),imgf)

        # cmap = plt.cm.gray
        # imgplot = plt.imshow(imgf,cmap = plt.cm.bwr , norm = norm)
        # plt.colorbar(imgplot)
        # plt.savefig(foldername+'RH/frame{0:02d}.png'.format(frame), dpi=100)
        # plt.clf()
        # plt.show()
        print ('frame', frame)