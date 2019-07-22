from VisualSti import borst as bs
import numpy as np
import pickle

class MD_borst(object):
    """docstring for ClassName"""
    def __init__(self):

        self.input = np.zeros((2,2,1))
        self.result = np.zeros((2,2,1))

        self.lptau = 5.0
        self.hptau = 25.0

        self.Eexc = +50.0
        self.Einh = -20.0
        self.gleak = 1.0
        self.mean = 0.0
        
    def run(self):

        noff = self.input.shape[1]

        lp = bs.lowpass(self.input, self.lptau)
        hp = bs.highpass(self.input, self.hptau)

        Mi9 = bs.rect(1.0-lp[:,0:noff-2,:],0)
        Mi1 = bs.rect(hp[:,1:noff-1,:],0)
        Mi4 = bs.rect(lp[:,2:noff-0,:],0)

        gexc = bs.rect(Mi1,0)
        ginh = bs.rect(Mi9+Mi4,0)

        # self.result = (self.Eexc*gexc+self.Einh*ginh)/(gexc+ginh+self.gleak)
        self.result = bs.rect((self.Eexc*gexc+self.Einh*ginh)/(gexc+ginh+self.gleak),0)
        self.mean = np.mean(self.result)
        # T4a_rect=bs.rect(T4a,0)
        # self.result = T4a
        return self.result

    def savpickle(self, clsfname = 'test.bs'):
        self.classfname = clsfname
        with open(self.classfname, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def loadpicke(self, ldfname = 'load.sti'):
        self.loadfname = ldfname
        with open(self.loadfname, 'rb') as input:
            self.input.movie = pickle.load(input)