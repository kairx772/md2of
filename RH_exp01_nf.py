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
    sw2 = md.NormalFlow_x()
    sw2.input = st
    sw2.run()
    rharr = np.empty_like(st)
    # for i in range(0,120):
    #     rharr[...,i] = sw2.run(st[...,i])
    #     print ('frame', i)
    with open('./result/fig3_2/nf_tfq.csv', 'a') as f:
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


    print (wx, type(wx))
    print (velo, type(velo))




    wt = wx + velo
    if (wt >= -10 and wt < -1):

        print ('wx = ', wx)
        print ('v = ', velo)
        print ('wt = ', wt)

        basic_stimulus('./data/05pad/<wx>{}<v>{}.sti'.format(wx, velo), './result/rh_wxwt/<wx>{}<v>{}.png'.format(wx, velo))

    with open('./result/fig3_2/nf_tfq.csv', 'a') as f:
        f.write(',')


# sw0 = rh.RH(inputframe = st[...,0])
# rh = np.empty_like(st)
# for i in range(0,120):
#     rh[...,i] = sw0.run(st[...,i])
#     print ('frame', i)

# rh = rh*0.05
# plt.plot(st[40,40,1:120], color='black', label='Image')
# plt.plot(rh[40,40,1:120], color='blue', label='Reichardt')
# plt.legend(loc='best')
# plt.show()
# timestr = time.strftime("%Y%m%d-%H%M%S")
# plt.savefig('result/rh_sti/{}.png'.format(timestr))





    # for wx in wxrange:
    #     for velo in vrange:2

    #         sw0 = gvs.sti()
    #         # sw0.V = 2.0 ** velo
    #         # sw0.wlen = 2.0 ** (-(wx))
    #         # sw0.sec = 1
    #         # sw0.singrat()

    #         fname = './data/05pad/<wx>{}<v>{}.sti'.format(wx, velo)

    #         with open(fname, 'rb') as input:
    #             sw0 = pickle.load(input)

    #         print ('V = ', sw0.V)
    #         print ('wlen = ', sw0.wlen)
        

    #         # plt.plot(wxrange, resp, linewidth=1)

    #         lowpass1 = sw0.movie[...,0]
    #         lowpass2 = sw0.movie[...,0]
    #         bandpass = np.empty_like(sw0.movie)
    #         conv = np.empty_like(sw0.movie)


    #         for i in range(0,120):
    #             conv[:,:,i] = signal.correlate2d(sw0.movie[:,:,i], gb, "same")

    #         lowpass1 = conv[...,0]
    #         lowpass2 = conv[...,0]
    #         bandpass = np.empty_like(conv)

    #         for i in range(1,120):
    #             temp1 = (1-fh)*lowpass1 + fh*conv[...,i]
    #             temp2 = (1-fl)*lowpass2 + fl*conv[...,i]
    #             lowpass1 = temp1
    #             lowpass2 = temp2
    #             bandpass[:,:,i] = lowpass1 - lowpass2
    #         resp.append(np.amax(bandpass[50,50,15:-15])-np.amin(bandpass[50,50,15:-15]))
    #         #resp_org.append(np.amax(sw0.movie[50,50,100:-100])-np.amin(sw0.movie[50,50,100:-100]))
    #     resp_v.append(resp)
    #     resp = []


    # plt.xlabel('velocity (power of 2)')
    # plt.ylabel('motion detection response')
    # plt.plot(vrange, resp_v[0], label=r'${\lambda}=2^{7}$', linewidth=1)

    # plt.plot(vrange, resp_v[1], label=r'${\lambda}=2^{6}$', linewidth=1)

    # plt.plot(vrange, resp_v[2], label=r'${\lambda}=2^{5}$', linewidth=1)

    # plt.plot(vrange, resp_v[3], label=r'${\lambda}=2^{4}$', linewidth=1)

    # plt.plot(vrange, resp_v[4], label=r'${\lambda}=2^{3}$', linewidth=1)

    # plt.plot(vrange, resp_v[5], label=r'${\lambda}=2^{2}$', linewidth=1)


    # plt.legend(loc='best')
    # plt.title('GF Lambda = {}, fl = {}, fh = {}'.format(Lambda, fl, fh))
    # # plt.plot(wxrange, resp_org, linewidth=1)
    # plt.savefig('result/test/<fl>{}<fh>{}<gabor>{}_cos.png'.format(fl, fh, Lambda), dpi=100)
    # # plt.plot(sw0.movie[50,50,:], linewidth=1)
    # plt.show()
    # plt.clf()


    # imgplot = plt.imshow(sw0.movie[...,0])
    # plt.colorbar(imgplot)
    # plt.show()
    # plt.clf()

    # imgplot = plt.imshow(sw0.movie[...,1])
    # plt.colorbar(imgplot)
    # plt.show()
    # plt.clf()

    # imgplot = plt.imshow(sw0.movie[...,2])
    # plt.colorbar(imgplot)
    # plt.show()
    # plt.clf()

    # imgplot = plt.imshow(sw0.movie[...,3])
    # plt.colorbar(imgplot)
    # plt.show()
    # plt.clf()

    # imgplot = plt.imshow(sw0.movie[...,4])
    # plt.colorbar(imgplot)
    # plt.show()
    # plt.clf()

    # sw0.genavi('./data/test.avi')

    # del sw0
    # gc.collect()

