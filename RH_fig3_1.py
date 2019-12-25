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

def basic_stimulus(inputfilen, outputfilen, preferdir = True, plot = True):
    with open(inputfilen, 'rb') as input:
        st = pickle.load(input).movie
    sw2 = rh.RH(inputframe = st[...,0],
                # A1_phase = 'cos',
                # A2_phase = 'sin',
                A1_sigma = 5,
                A2_sigma = 5,
                # A1_Lambda = 4.0,
                # A2_Lambda = 4.0,
                fl = 0.25,
                fh = 0.25,
                # winsize = 2
                )
    # sw2.print_gabor()
    rharr = np.empty_like(st)
    A2B1 = np.empty_like(st)
    A1B2 = np.empty_like(st)
    A1 = np.empty_like(st)
    A2 = np.empty_like(st)
    B1 = np.empty_like(st)
    B2 = np.empty_like(st)
    # sw2.print_gabor('gab3')
    frame_range = 120
    if (preferdir):
        for i in range(0,frame_range):
            rharr[...,i] = sw2.run(st[...,i])
            A2B1[...,i] = sw2.A2B1
            A1B2[...,i] = sw2.A1B2
            A1[...,i] = sw2.convA1
            A2[...,i] = sw2.convA2
            B1[...,i] = sw2.lowpassA1
            B2[...,i] = sw2.lowpassA2
            print ('frame', i)
    else :
        for i in range(0,frame_range):
            rharr[...,i] = sw2.run(st[...,frame_range-i])
            A2B1[...,i] = sw2.A2B1
            A1B2[...,i] = sw2.A1B2
            A1[...,i] = sw2.convA1
            A2[...,i] = sw2.convA2
            B1[...,i] = sw2.lowpassA1
            B2[...,i] = sw2.lowpassA2
            print ('frame', i)
    rharr = rharr*1
    A2B1 = A2B1*1
    A1B2 = A1B2*1
    fig, ax1 = plt.subplots()
    fig.subplots_adjust(right=0.88)
    ax2 = ax1.twinx()
    ax1.set_ylim([-2,2])
    ax2.set_ylim([-45,45])
    sampleloc = 45
    p1, = ax1.plot(st[sampleloc,sampleloc,1:frame_range], color='black', label='input image')
    # p2, = ax2.plot(rharr[sampleloc,sampleloc,1:frame_range], color='blue', label='Reichardt')
    # p3, = ax2.plot(A2B1[sampleloc,sampleloc,1:frame_range], '--', color='green', label='A2B1')
    # p4, = ax2.plot(A1B2[sampleloc,sampleloc,1:frame_range], '--', color='red', label='A1B2')
    p5, = ax2.plot(A1[sampleloc,sampleloc,1:frame_range], color='green', label='A1')
    p6, = ax2.plot(A2[sampleloc,sampleloc,1:frame_range], color='red', label='A2')
    p7, = ax2.plot(B1[sampleloc,sampleloc,1:frame_range], '--', color='green', label='B1')
    p8, = ax2.plot(B2[sampleloc,sampleloc,1:frame_range], '--', color='red', label='B2')

    ax1.set_xlabel('frame number')
    ax1.set_ylabel('image intensity', color='black')
    ax2.set_ylabel('response', color='blue')
    lines = [p1, p5, p6, p7, p8]
    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc='best')
    if (plot):
        plt.savefig(outputfilen+'.png')
    else:
        plt.show()
    plt.clf()


    fig2, ax3 = plt.subplots()
    fig2.subplots_adjust(right=0.88)
    sampleloc = 45
    # p1, = ax3.plot(st[sampleloc,sampleloc,1:frame_range], color='black', label='Input image')
    
    p3, = ax3.plot(A2B1[sampleloc,sampleloc,1:frame_range], '--', color='green', label='A2B1')
    p4, = ax3.plot(A1B2[sampleloc,sampleloc,1:frame_range], '--', color='red', label='A1B2')
    p2, = ax3.plot(rharr[sampleloc,sampleloc,1:frame_range], color='blue', label='output')
    # p5, = ax3.plot(A1[sampleloc,sampleloc,1:frame_range], color='green', label='A1')
    # p6, = ax3.plot(A2[sampleloc,sampleloc,1:frame_range], color='red', label='A2')

    ax3.set_xlabel('frame number')
    # ax1.set_ylabel('image intensity', color='black')
    ax3.set_ylabel('response', color='blue')
    lines = [p3, p4, p2]
    labs = [l.get_label() for l in lines]
    ax3.legend(lines, labs, loc='best')
    if (plot):
        plt.savefig(outputfilen+'_out.png')
    else:
        plt.show()
    plt.clf()

def main():
    pass



if __name__ == '__main__':

    # basic_stimulus('./data/basic/valley.sti', 'result/fig3_1/valley' ,preferdir = True)
    # basic_stimulus('./data/basic/valley.sti', 'result/fig3_1/null_valley', preferdir = False)
    # basic_stimulus('./data/basic/peak.sti', 'result/fig3_1/null_peak', preferdir = True)
    # basic_stimulus('./data/basic/peak.sti', 'result/fig3_1/peak', preferdir = True)
    # basic_stimulus('./data/basic/peak.sti', 'result/fig3_1/peak_null', preferdir = False)
    # basic_stimulus('./data/basic/gradient.sti', 'result/fig3_1/gradient', preferdir = True)
    # basic_stimulus('./data/basic/gradient.sti', 'result/fig3_1/null_gradient', preferdir = False)
    # basic_stimulus('./data/basic/gradient_minus.sti', 'result/fig3_1/gradient_minus', preferdir = True)
    # basic_stimulus('./data/basic/gradient_minus.sti', 'result/fig3_1/null_gradient_minus', preferdir = False)

    basic_stimulus('./data/05pad/<wx>-5.0<v>-1.0.sti', 'result/fig3_1/singrattest', preferdir = True)
    basic_stimulus('./data/05pad/<wx>-5.0<v>-1.0.sti', 'result/fig3_1/null_singrattest', preferdir = False)
