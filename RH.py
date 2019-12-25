# coding: utf8
import matplotlib.pyplot as plt
from VisualSti import GenerateVisualStimuli as gvs
from VisualSti import MotionDetection as md
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import os
import gc
import numpy as np
from scipy import signal

def gabor_fn_cos(sigma, theta, Lambda, psi, gamma):
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 3 # Number of standard deviation sigma
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    print ('xmax', xmax)
    xmax = np.ceil(max(1, xmax))
    print ('xmax', xmax)
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation 
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    return gb

degre = np.pi/2
sigma = 5.0
Lambda = 15.0
degree = 180*degre/np.pi

gb_cos = gabor_fn_cos(sigma, degre, Lambda, 0, 1.0)

def gabor_fn_sin(sigma, theta, Lambda, psi, gamma):
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 3 # Number of standard deviation sigma
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    print ('xmax', xmax)
    xmax = np.ceil(max(1, xmax))
    print ('xmax', xmax)
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation 
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.sin(2 * np.pi / Lambda * x_theta + psi)
    return gb

degre = np.pi/2
sigma = 5.0
Lambda = 15.0
degree = 180*degre/np.pi

gb_sin = gabor_fn_sin(sigma, degre, Lambda, 0, 1.0)

degre = -np.pi/2

gb_sin_NULL = gabor_fn_sin(sigma, degre, Lambda, 0, 1.0)


def main():
    pass

# This file is to generate visual stumuls data

if __name__ == '__main__':


    norm = mpl.colors.Normalize(vmin=-1, vmax=1)

    imgplot = plt.imshow(gb_cos, cmap = plt.cm.gray , norm = norm)
    plt.title('degree = {}, sigma = {}'.format(degree, sigma))
    plt.colorbar(imgplot)
    plt.show()
    plt.clf()

    imgplot = plt.imshow(gb_sin, cmap = plt.cm.gray , norm = norm)
    plt.title('degree = {}, sigma = {}'.format(degree, sigma))
    plt.colorbar(imgplot)
    plt.show()
    plt.clf()

    imgplot = plt.imshow(gb_sin_NULL, cmap = plt.cm.gray , norm = norm)
    plt.title('degree = {}, sigma = {}'.format(-90, sigma))
    plt.colorbar(imgplot)
    plt.show()
    plt.clf()

    fl = 0.2
    fh = 0.4
    
    wx = -5.0
    velo = 0.0

    resp = []
    resp_org = []
    resp_v = []

    print ('wx', wx)
    print ('v', velo)

    wxrange = np.arange(-6.0, -1.0)
    vrange = np.arange(-4.0,5.0)

    sw0 = gvs.sti()
    fname = './data/05pad/<wx>{}<v>{}.sti'.format(wx, velo)
    print ('load file')
    with open(fname, 'rb') as input:
        sw0 = pickle.load(input)

    lowpass1 = sw0.movie[...,0]
    lowpass2 = sw0.movie[...,0]
    Reichardt = np.empty_like(sw0.movie)
    ReichardtNull = np.empty_like(sw0.movie)

    convR = np.empty_like(sw0.movie)
    convL = np.empty_like(sw0.movie)
    convL_null = np.empty_like(sw0.movie)

    print ('sptial filter')
    for i in range(0,120):
        print ('spf ', i)
        convR[:,:,i] = signal.correlate2d(sw0.movie[:,:,i], gb_cos, "same")
        convL[:,:,i] = signal.correlate2d(sw0.movie[:,:,i], gb_sin, "same")
        convL_null[:,:,i] = signal.correlate2d(sw0.movie[:,:,i], gb_sin_NULL, "same")

    lowpass1 = convR[...,0]
    lowpass2 = convL[...,0]

    # Reichardt = np.empty_like(conv)


    print ('time filter')
    for i in range(1,120):
        print ('spf ', i)
        temp1 = (1-fh)*lowpass1 + fh*convR[...,i]
        temp2 = (1-fl)*lowpass2 + fl*convL[...,i]
        lowpass1 = temp1
        lowpass2 = temp2
        A2B1 = np.multiply(lowpass1, convL[...,i])
        A1B2 = np.multiply(lowpass2, convR[...,i])
        Reichardt[:,:,i] = A2B1 - A1B2

    lowpass1 = convR[...,0]
    lowpass3 = convL_null[...,0]

    for i in range(1,120):
        print ('spf NULL', i)
        temp1 = (1-fh)*lowpass1 + fh*convR[...,i]
        temp2 = (1-fl)*lowpass2 + fl*convL_null[...,i]
        lowpass1 = temp1
        lowpass2 = temp2
        A2B1 = A2B1 = np.multiply(lowpass1, convL_null[...,i])
        A1B2 = np.multiply(lowpass2, convR[...,i])
        ReichardtNull[:,:,i] = A2B1 - A1B2

    ReichardtNull = ReichardtNull*0.05
    Reichardt = Reichardt*0.05

    fig, ax = plt.subplots()
    # ax.axis([1, 150, -0.5, 5])
    plt.plot(sw0.movie[40,40,1:120], color='black', label='Image')
    # plt.plot(convR[40,40,1:120], color='blue', label='normal flow')
    # plt.plot(convL[40,40,1:120], color='red', label='Borst')
    plt.plot(Reichardt[40,40,1:120], color='blue', label='Reichardt')
    plt.plot(ReichardtNull[40,40,1:120], color='green', label='Reichardt(NuLL)')
    plt.legend(loc='best')

    plt.show()

    for wx in wxrange:
        for velo in vrange:

            sw0 = gvs.sti()
            # sw0.V = 2.0 ** velo
            # sw0.wlen = 2.0 ** (-(wx))
            # sw0.sec = 1
            # sw0.singrat()

            fname = './data/05pad/<wx>{}<v>{}.sti'.format(wx, velo)

            with open(fname, 'rb') as input:
                sw0 = pickle.load(input)

            print ('V = ', sw0.V)
            print ('wlen = ', sw0.wlen)
        

            # plt.plot(wxrange, resp, linewidth=1)

            lowpass1 = sw0.movie[...,0]
            lowpass2 = sw0.movie[...,0]
            bandpass = np.empty_like(sw0.movie)
            conv = np.empty_like(sw0.movie)


            for i in range(0,120):
                conv[:,:,i] = signal.correlate2d(sw0.movie[:,:,i], gb, "same")

            lowpass1 = conv[...,0]
            lowpass2 = conv[...,0]
            bandpass = np.empty_like(conv)

            for i in range(1,120):
                temp1 = (1-fh)*lowpass1 + fh*conv[...,i]
                temp2 = (1-fl)*lowpass2 + fl*conv[...,i]
                lowpass1 = temp1
                lowpass2 = temp2
                bandpass[:,:,i] = lowpass1 - lowpass2
            resp.append(np.amax(bandpass[50,50,15:-15])-np.amin(bandpass[50,50,15:-15]))
            #resp_org.append(np.amax(sw0.movie[50,50,100:-100])-np.amin(sw0.movie[50,50,100:-100]))
        resp_v.append(resp)
        resp = []


    plt.xlabel('velocity (power of 2)')
    plt.ylabel('motion detection response')
    plt.plot(vrange, resp_v[0], label=r'${\lambda}=2^{7}$', linewidth=1)

    plt.plot(vrange, resp_v[1], label=r'${\lambda}=2^{6}$', linewidth=1)

    plt.plot(vrange, resp_v[2], label=r'${\lambda}=2^{5}$', linewidth=1)

    plt.plot(vrange, resp_v[3], label=r'${\lambda}=2^{4}$', linewidth=1)

    plt.plot(vrange, resp_v[4], label=r'${\lambda}=2^{3}$', linewidth=1)

    plt.plot(vrange, resp_v[5], label=r'${\lambda}=2^{2}$', linewidth=1)


    plt.legend(loc='best')
    plt.title('GF Lambda = {}, fl = {}, fh = {}'.format(Lambda, fl, fh))
    # plt.plot(wxrange, resp_org, linewidth=1)
    plt.savefig('result/test/<fl>{}<fh>{}<gabor>{}_cos.png'.format(fl, fh, Lambda), dpi=100)
    # plt.plot(sw0.movie[50,50,:], linewidth=1)
    plt.show()
    plt.clf()


    imgplot = plt.imshow(sw0.movie[...,0])
    plt.colorbar(imgplot)
    plt.show()
    plt.clf()

    imgplot = plt.imshow(sw0.movie[...,1])
    plt.colorbar(imgplot)
    plt.show()
    plt.clf()

    imgplot = plt.imshow(sw0.movie[...,2])
    plt.colorbar(imgplot)
    plt.show()
    plt.clf()

    imgplot = plt.imshow(sw0.movie[...,3])
    plt.colorbar(imgplot)
    plt.show()
    plt.clf()

    imgplot = plt.imshow(sw0.movie[...,4])
    plt.colorbar(imgplot)
    plt.show()
    plt.clf()

    sw0.genavi('./data/test.avi')

    del sw0
    gc.collect()


    main()