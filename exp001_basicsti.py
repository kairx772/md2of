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

    sw0 = gvs.sti()
    sw0.wlen = 2**(4)
    sw0.fps = 120
    sw0.sec = 2
    sw0.degr = 0
    sw0.V = 2**(-1)
    sw0.contrast = 1
    
    sw0.peak()
    sw0.genavi('./data/basic/peak.avi')
    sw0.savpickle('./data/basic/peak.sti')

    sw0.peak_minus()
    sw0.genavi('./data/basic/valley.avi')
    sw0.savpickle('./data/basic/valley.sti')
    
    sw0.gradient()
    sw0.genavi('./data/basic/gradient.avi')
    sw0.savpickle('./data/basic/gradient.sti')

    sw0.gradient_minus()
    sw0.genavi('./data/basic/gradient_minus.avi')
    sw0.savpickle('./data/basic/gradient_minus.sti')

    sw0.singrat()
    sw0.genavi('./data/basic/singrat.avi')
    sw0.savpickle('./data/basic/singrat.sti')

    # sw = sw0.gradient_minus()
    # sw0.savpickle('./data/basic/gradient_minus.sti')

    # sw = sw0.gradient_minus()
    # sw0.savpickle('./data/basic/gradient_minus.sti')

    # sw = sw0.peak()
    # sw0.genavi_null('peak.avi')
    # sw0.savpickle('./data/basic/peak.sti')

    # sw = sw0.peak_minus()
    # sw0.genavi_null('peak_minus.avi')
    # sw0.savpickle('./data/basic/peak_minus.sti')

    # sw = sw0.peak_avg_bg()
    # sw0.savpickle('./data/basic/peak_avg_bg.sti')

    # sw = sw0.peak_avg_bg_minus()
    # sw0.savpickle('./data/basic/peak_avg_bg_minus.sti')
    # main()