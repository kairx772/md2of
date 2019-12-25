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


def basic_stimulus(inputfilen, writefile):
    with open(inputfilen, 'rb') as input:
        st = pickle.load(input).movie
    sw2 = rh.RH(inputframe = st[...,0])
    rharr = np.empty_like(st)
    for i in range(0,120):
        rharr[...,i] = sw2.run(st[...,i])
        print ('frame', i)
    with open(writefile, 'a') as f:
        f.write(str(np.mean(rharr)))

def basic_stimulus_nf(inputfilen, writefile):
    with open(inputfilen, 'rb') as input:
        st = pickle.load(input).movie
    sw2 = md.NormalFlow_x()
    sw2.input = st
    sw2.run()
    # for i in range(0,120):
    #     rharr[...,i] = sw2.run(st[...,i])
    #     print ('frame', i)
    with open(writefile, 'a') as f:
        f.write(str((sw2.mean)))

def basic_stimulus_bs(inputfilen, writefile):
    with open(inputfilen, 'rb') as input:
        st = pickle.load(input).movie
    sw2 = md.MD_borst()
    sw2.input = st
    sw2.run()
    # for i in range(0,120):
    #     rharr[...,i] = sw2.run(st[...,i])
    #     print ('frame', i)
    with open(writefile, 'a') as f:
        f.write(str((sw2.mean)))
    # rharr = rharr*0.05
    # plt.xlabel('frame')
    # plt.ylabel('response')
    # plt.plot(st[40,40,1:120], color='black', label='Image')
    # plt.plot(rharr[40,40,1:120], color='blue', label='Reichardt')
    # plt.legend(loc='best')
    # plt.savefig(outputfilen)
    # plt.clf()

# def basic_stimulus(inputfilen, outputfilen):
#     with open(inputfilen, 'rb') as input:
#         st = pickle.load(input).movie
#     sw2 = rh.RH(inputframe = st[...,0])
#     rharr = np.empty_like(st)
#     for i in range(0,120):
#         rharr[...,i] = sw2.run(st[...,i])
#         print ('frame', i)
#     rharr = rharr*0.05
#     plt.xlabel('frame')
#     plt.ylabel('response')
#     plt.plot(st[40,40,1:120], color='black', label='Image')
#     plt.plot(rharr[40,40,1:120], color='blue', label='Reichardt')
#     plt.legend(loc='best')
#     plt.savefig(outputfilen)
#     plt.clf()

def main():
    pass



if __name__ == '__main__':

    # basic_stimulus('./data/basic/gradient.sti', 'result/rh_sti/gradient.png')
    # basic_stimulus_null('./data/basic/gradient.sti', 'result/rh_sti/gradient_null.png')

    # basic_stimulus('./data/basic/peak.sti', 'result/rh_sti/peak.png')
    # basic_stimulus_null('./data/basic/peak.sti', 'result/rh_sti/peak.png')

    # basic_stimulus('./data/basic/peak_minus.sti', 'result/rh_wxwt/test.png')
    # basic_stimulus_null('./data/basic/peak_minus.sti', 'result/rh_sti/peak_minus.png')

    # basic_stimulus('./data/basic/peak_minus.sti', 'result/rh_sti/peak_minus.png')
    # basic_stimulus_null('./data/basic/peak_minus.sti', 'result/rh_sti/peak_minus.png')

    # basic_stimulus('./data/data/05pad/peak_minus.sti', 'result/rh_wxwt/test.png')


    wx = float(sys.argv[1])
    velo = float(sys.argv[2])
    filename = sys.argv[3]


    print (wx, type(wx))
    print (velo, type(velo))




    wt = wx + velo
    if (wt >= -10 and wt < -1):

        print ('wx = ', wx)
        print ('v = ', velo)
        print ('wt = ', wt)

        basic_stimulus('./data/05pad/<wx>{}<v>{}.sti'.format(wx, velo), filename)

    with open(filename, 'a') as f:
        f.write(',')

