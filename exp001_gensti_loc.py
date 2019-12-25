# coding: utf8
import matplotlib.pyplot as plt
from VisualSti import GenerateVisualStimuli as gvs
from VisualSti import MotionDetection as md
import matplotlib.pyplot as plt
import pickle
import os
import gc
import numpy as np
import sys

def main():
    pass

# This file is to generate visual stumuls data

if __name__ == '__main__':

    # vrange = np.arange(-6,4.5,.5)
    # wxrange = np.arange(-8,-1.5,.5)
    # (wt >= -10 and wt < -1)

    wx = float(sys.argv[1])
    velo = float(sys.argv[2])


    print (wx, type(wx))
    print (velo, type(velo))

    wt = wx + velo
    if (wt >= -10 and wt < -1):

        print ('wx = ', wx)
        print ('v = ', velo)
        print ('wt = ', wt)

        sw0 = gvs.sti()
        sw0.V = velo
        sw0.wlen = 2.0 ** (-(wx))
        sw0.sec = 4
        sw0.singrat()
        sw0.genavi('./data/01pad/<wx>{}<v>{}.avi'.format(wx, velo))
        sw0.savpickle('./data/01pad/<wx>{}<v>{}.sti'.format(wx, velo))
        del sw0
        gc.collect()

    main()