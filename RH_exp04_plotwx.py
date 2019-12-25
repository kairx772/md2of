# coding: utf8
import matplotlib.pyplot as plt
from VisualSti import GenerateVisualStimuli as gvs
from VisualSti import MotionDetection as md
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd

def main():
    pass

if __name__ == '__main__':

    # data = 'result/fig3_2/rh_wx.csv'
    # plot = './result/fig3_2/rh_wx.png'

    # data = './result/fig3_2/bs_wx.csv'
    # plot = './result/fig3_2/bs_wx.png'

    data = 'result/fig3_2/rh_wx.csv'
    plot = 'result/fig3_2/rh_wx_t.png'

    df = pd.read_csv(data)


    # df = df.T
    # df.to_csv('./result/fig3_2/bs_wx.csv')
    # tfq_nf_test2
    # tfq

    print (df[df.columns[0]])
    fig, ax = plt.subplots()

    # for i in (0,14):
    #     plt.plot(df[df.columns[0]], df[df.columns[i+1]], label=r'${\lambda}=2^{}$',linewidth=1)

    plt.plot(df[df.columns[0]], df[df.columns[1]], label=r'${\lambda}=2^{8}$',linewidth=1)
    # plt.plot(df[df.columns[0]], df[df.columns[2]], label=r'${\lambda}=2^{7.5}$',linewidth=1)
    plt.plot(df[df.columns[0]], df[df.columns[3]], label=r'${\lambda}=2^{7}$',linewidth=1)
    # plt.plot(df[df.columns[0]], df[df.columns[4]], label=r'${\lambda}=2^{6.5}$',linewidth=1)
    plt.plot(df[df.columns[0]], df[df.columns[5]], label=r'${\lambda}=2^{6}$',linewidth=1)
    # plt.plot(df[df.columns[0]], df[df.columns[6]], label=r'${\lambda}=2^{5.5}$',linewidth=1)
    plt.plot(df[df.columns[0]], df[df.columns[7]], label=r'${\lambda}=2^{5}$',linewidth=1)
    # plt.plot(df[df.columns[0]], df[df.columns[8]], label=r'${\lambda}=2^{4.5}$',linewidth=1)
    plt.plot(df[df.columns[0]], df[df.columns[9]], label=r'${\lambda}=2^{4}$',linewidth=1)
    # plt.plot(df[df.columns[0]], df[df.columns[10]], label=r'${\lambda}=2^{3.5}$',linewidth=1)
    plt.plot(df[df.columns[0]], df[df.columns[11]], label=r'${\lambda}=2^{3}$',linewidth=1)
    # plt.plot(df[df.columns[0]], df[df.columns[12]], label=r'${\lambda}=2^{2.5}$',linewidth=1)
    plt.plot(df[df.columns[0]], df[df.columns[13]], label=r'${\lambda}=2^{2}$',linewidth=1)
    # plt.plot(df[df.columns[0]], df[df.columns[14]], label=r'${\lambda}=2^{1.5}$',linewidth=1)
    plt.plot(df[df.columns[0]], df[df.columns[15]], label=r'${\lambda}=2^{1}$',linewidth=1)

    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 16

    plt.xlabel('velocity pixel/frame (power of 2)', size=SMALL_SIZE)
    # plt.ylabel('optic flow value', size=SMALL_SIZE)
    plt.ylabel('motion detection response', size=SMALL_SIZE)


    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE) 
    plt.rc('font', size=14)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    # fig.set_size_inches(18.5, 10.5)
    plt.legend(loc='best')
    plt.savefig(plot, dpi=100)
    plt.show()


    main()