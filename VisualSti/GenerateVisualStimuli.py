# coding: utf8
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from numba import autojit
# from VisualSti import borst as bs
import pandas as pd

class gen:
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

    @autojit
    def singrat(self):
        self.Vx = self.V*np.cos(self.Vdgr)
        self.Vy = self.V*np.sin(self.Vdgr)
        self.angl = (self.degr/180)*np.pi
        self.maxt = self.fps*self.sec
        # self.movie = np.zeros((self.height, self.width, self.maxt))
        [xv, yv, tt] = np.meshgrid(range(0, self.width), range(0, self.height), range(0, self.maxt))
        self.movie = np.cos(2.0*np.pi*((np.sin(self.angl)/self.wlen)*(yv-(self.Vy*tt)) + (np.cos(self.angl)/(self.wlen))*(xv-(self.Vx*tt))))
        # [xv, yv] = np.meshgrid(range(0, self.width), range(0, self.height))
        # for t in range(self.maxt):
        #     print ("frame:", t)
        #     self.movie[:,:,t] = np.cos(2.0*np.pi*((np.sin(self.angl)/self.wlen)*(yv-(self.Vy*t)) + (np.cos(self.angl)/(self.wlen))*(xv-(self.Vx*t))))
            # for i in range(self.height):
            #     for j in range(self.width):
            #         self.movie[i,j,t] = np.cos(2.0*np.pi*((np.sin(self.angl)/self.wlen)*(i-(self.Vx*t)) + (np.cos(self.angl)/(self.wlen))*(j-(self.Vy*t))))
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

def main():
    pass

if __name__ == '__main__': 

    sw0 = gen()
    sw0.fps = 120
    sw0.sec = 10
    sw0.degr = 30
    sw = sw0.singrat()
    sw0.fname = 'test.avi'
    sw0.genavi()

    # plt.imshow(sw[:,:,0], cmap=plt.gray())
    # plt.show()

    main()


