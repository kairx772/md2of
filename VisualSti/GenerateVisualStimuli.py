# coding: utf8
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from numba import autojit
# from VisualSti import borst as bs
import pandas as pd
import pickle

class sti:
    def __init__(self):
        self.width = 320
        self.height = 240
        self.fps = 120
        self.sec = 10
        self.contrast = 1.0
        # self.maxt = self.fps*self.sec
        # self.movie = np.zeros((self.height, self.width, self.maxt))

        # sine grating parameter
        self.wlen = 80
        self.degr = 0
        self.angl = (self.degr/180)*np.pi
        self.V = 1
        self.V = self.wlen/self.fps
        # self.V_t = np.ones(self.maxt) * self.V
        # self.degr = 45
        # self.angl = (self.degr/180)*np.pi
        self.Vdgr = self.angl
        self.Vy = self.V*np.cos(self.Vdgr)
        self.Vx = self.V*np.sin(self.Vdgr)

        # genavi pareter
        self.fname = 'out.avi'
        self.classfname = 'test.sti'

    @autojit
    def singrat(self):
        self.Vx = self.V*np.cos(self.Vdgr)
        self.Vy = self.V*np.sin(self.Vdgr)
        self.angl = (self.degr/180)*np.pi
        self.maxt = self.fps*self.sec
        [xv, yv, tt] = np.meshgrid(range(0, self.width), range(0, self.height), range(0, self.maxt))
        self.movie = np.cos(2.0*np.pi*((np.sin(self.angl)/self.wlen)*(yv-(self.Vy*tt)) + (np.cos(self.angl)/(self.wlen))*(xv-(self.Vx*tt))))
        self.movie = self.movie*self.contrast*0.5+0.5
        return self.movie

    def genavi(self):
        normalizedImg = np.zeros((self.height, self.width))
        out = cv.VideoWriter(self.fname, cv.VideoWriter_fourcc(*'XVID'), self.fps, (self.width, self.height))
        for i in range(self.maxt):
            normalizedImg = self.movie[:,:,i] * 255 + 0.5
            # normalizedImg = cv.normalize(self.movie[:,:,i],  normalizedImg, 0, 255, cv.NORM_MINMAX)
            normalizedImg = normalizedImg.astype(np.uint8)
            normalizedImg = cv.cvtColor(normalizedImg,cv.COLOR_GRAY2RGB)
            out.write(normalizedImg)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    def savpickle(self):
        with open(self.classfname, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

def main():
    pass

if __name__ == '__main__': 

    sw0 = sti()
    sw0.fps = 120
    sw0.sec = 4
    sw0.degr = 45
    sw0.classfname = 'test2.sti'
    sw = sw0.singrat()
    sw0.fname = 'test2.avi'
    sw0.genavi()
    sw0.savpickle()

    with open('test.sti', 'rb') as input:
        sw0 = pickle.load(input)
    print (sw0.degr)


    main()


