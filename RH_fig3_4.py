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


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

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


def basic_stimulus(inputfilen, outputfilen, preferdir = True, plot = True):

    frame_range = 240

    with open(inputfilen, 'rb') as input:
        st = pickle.load(input).movie

    sw_nf = md.NormalFlow_x()
    sw_nf.input = st
    sw_nf.run()
    print (sw_nf.result.shape)

    sw_bs = md.MD_borst()
    sw_bs.input = st
    sw_bs.run()
    print (sw_bs.result.shape)
    
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


    SMALL_SIZE = 14
    fig, host = plt.subplots()
    fig.subplots_adjust(right=0.75)

    par1 = host.twinx()
    par2 = host.twinx()
    par3 = host.twinx()


    par2.spines["right"].set_position(("axes", 1.1))
    par3.spines["right"].set_position(("axes", 1.2))

    make_patch_spines_invisible(par2)

    par2.spines["right"].set_visible(True)


    
    sampleloc = 45
    p1, = host.plot(st[sampleloc,sampleloc,1:frame_range], color='black', label='input image')
    # p2, = ax3.plot(rharr[sampleloc,sampleloc,1:frame_range], color='blue', label='st-Reichardt')
    # p3, = ax3.plot(sw_bs.result[sampleloc+1,sampleloc+1,0:frame_range], color='green', label='Borst')
    # p4, = ax4.plot(sw_nf.result[sampleloc,sampleloc,1:frame_range], color='red', label='LK-method')

    p2, = par1.plot(rharr[sampleloc,sampleloc,1:frame_range], color='blue', label='st-Reichardt')
    p3, = par2.plot(sw_bs.result[sampleloc+1,sampleloc+1,0:frame_range], '--', color='red', label='Borst')
    p4, = par3.plot(sw_nf.result[sampleloc,sampleloc,1:frame_range], '--', color='green', label='LK-method')


    # host.set_xlim(0, 2)
    host.set_ylim(-0.5, 2)
    par1.set_ylim(-50, 200)
    par2.set_ylim(-2, 8)
    par3.set_ylim(-0.5, 2)

    host.set_xlabel('frame number', size=SMALL_SIZE)
    host.set_ylabel('image intensity', color='black', size=SMALL_SIZE)
    par1.set_ylabel('st-Reichardt response', color='blue', size=SMALL_SIZE)
    par2.set_ylabel('Borst response', color='red', size=SMALL_SIZE)
    par3.set_ylabel('optical flow value', color='green', size=SMALL_SIZE)

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    par2.yaxis.label.set_color(p3.get_color())
    par3.yaxis.label.set_color(p4.get_color())

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes


    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
    # par3.tick_params(axis='y', colors='red', **tkw)
    host.tick_params(axis='x', **tkw)
    fig.set_size_inches(15.5, 7.5)

    lines = [p1, p2, p3, p4]

    host.legend(lines, [l.get_label() for l in lines])
    if (plot):
        plt.savefig(outputfilen+'.png')
    else:
        plt.show()
    plt.clf()


    # fig2, ax3 = plt.subplots()
    # fig2.subplots_adjust(right=0.88)
    # sampleloc = 45
    # # p1, = ax3.plot(st[sampleloc,sampleloc,1:frame_range], color='black', label='Input image')
    
    # p3, = ax3.plot(A2B1[sampleloc,sampleloc,1:frame_range], '--', color='green', label='A2B1')
    # p4, = ax3.plot(A1B2[sampleloc,sampleloc,1:frame_range], '--', color='red', label='A1B2')
    # p2, = ax3.plot(rharr[sampleloc,sampleloc,1:frame_range], color='blue', label='output')
    # # p5, = ax3.plot(A1[sampleloc,sampleloc,1:frame_range], color='green', label='A1')
    # # p6, = ax3.plot(A2[sampleloc,sampleloc,1:frame_range], color='red', label='A2')

    # ax3.set_xlabel('frame number')
    # # ax1.set_ylabel('image intensity', color='black')
    # ax3.set_ylabel('response', color='blue')
    # lines = [p3, p4, p2]
    # labs = [l.get_label() for l in lines]
    # ax3.legend(lines, labs, loc='best')
    # if (plot):
    #     plt.savefig(outputfilen+'_out.png')
    # else:
    #     plt.show()
    # plt.clf()

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

    basic_stimulus('./data/05pad/<wx>-5.0<v>-1.0.sti', 'result/fig3_3/test', preferdir = True, plot = True)
    # basic_stimulus('./data/basic/singrat.sti', 'result/fig3_1/null_singrat', preferdir = False)
