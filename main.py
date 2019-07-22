# coding: utf8
import matplotlib.pyplot as plt
from VisualSti import GenerateVisualStimuli as gvs
from VisualSti import MotionDetection as md
import matplotlib.pyplot as plt
import pickle

def main():
    pass

if __name__ == '__main__':
    sw0 = gvs.gen()
    sw0.V = 10
    sw = sw0.singrat()
    sw0.fname = 'out3.avi'
    sw0.genavi()

    test1 = md.MD_borst()
    test1.input = sw
    test1.run()

    print (test1.result.shape)

    plt.plot(sw[10,10,:], color='black', label='local',linewidth=1)
    plt.plot(test1.result[10,10,:], color='red', label='local',linewidth=1)
    plt.savefig('test.png', bbox_inches='tight')
    # plt.show()

    main()