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


def main():
    pass



if __name__ == '__main__':


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

        basic_stimulus_bs('./data/05pad/<wx>{}<v>{}.sti'.format(wx, velo), filename)

    with open(filename, 'a') as f:
        f.write(',')
