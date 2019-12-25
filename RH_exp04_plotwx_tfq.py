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

    df = pd.read_csv('./result/rh_tf.csv')


    # df = df1.T
    # df.to_csv('./result/rh_wx.csv')
    # tfq_nf_test2
    # tfq

    print (df[df.columns[0]])
    fig, ax = plt.subplots()

    plt.xlabel('time frequency (power of 2)')
    plt.ylabel('motion detection response')

    # for i in (0,14):
    #     plt.plot(df[df.columns[0]], df[df.columns[i+1]], label=r'${\lambda}=2^{}$',linewidth=1)

    plt.plot(df[df.columns[0]], df[df.columns[1]], label=r'${\lambda}=2^{8}$',linewidth=1)
    plt.plot(df[df.columns[0]], df[df.columns[2]], label=r'${\lambda}=2^{7.5}$',linewidth=1)
    plt.plot(df[df.columns[0]], df[df.columns[3]], label=r'${\lambda}=2^{7}$',linewidth=1)
    plt.plot(df[df.columns[0]], df[df.columns[4]], label=r'${\lambda}=2^{6.5}$',linewidth=1)
    plt.plot(df[df.columns[0]], df[df.columns[5]], label=r'${\lambda}=2^{6}$',linewidth=1)
    plt.plot(df[df.columns[0]], df[df.columns[6]], label=r'${\lambda}=2^{5.5}$',linewidth=1)
    plt.plot(df[df.columns[0]], df[df.columns[7]], label=r'${\lambda}=2^{5}$',linewidth=1)
    plt.plot(df[df.columns[0]], df[df.columns[8]], label=r'${\lambda}=2^{4.5}$',linewidth=1)
    plt.plot(df[df.columns[0]], df[df.columns[9]], label=r'${\lambda}=2^{4}$',linewidth=1)
    plt.plot(df[df.columns[0]], df[df.columns[10]], label=r'${\lambda}=2^{3.5}$',linewidth=1)
    plt.plot(df[df.columns[0]], df[df.columns[10]], label=r'${\lambda}=2^{3}$',linewidth=1)
    plt.plot(df[df.columns[0]], df[df.columns[10]], label=r'${\lambda}=2^{2.5}$',linewidth=1)
    plt.plot(df[df.columns[0]], df[df.columns[10]], label=r'${\lambda}=2^{2}$',linewidth=1)
    plt.plot(df[df.columns[0]], df[df.columns[10]], label=r'${\lambda}=2^{1.5}$',linewidth=1)
    plt.plot(df[df.columns[0]], df[df.columns[10]], label=r'${\lambda}=2^{1}$',linewidth=1)

    # plt.figure.set_size_inches(18.5, 10.5)
    # plt.savefig('test2png.png', dpi=100)
    plt.legend(loc='best')
    plt.savefig('result/plot/rh_tf.png', dpi=100)
    plt.show()


    main()