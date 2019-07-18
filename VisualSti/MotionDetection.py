from VisualSti import borst as bs
import numpy as np

class MD_borst(object):
    """docstring for ClassName"""
    def __init__(self):

        self.input = np.zeros((2,2,1))
        self.result = np.zeros((2,2,1))

        self.lptau = 5.0
        self.hptau = 25.0

        self.Eexc=+50.0
        self.Einh=-20.0
        self.gleak=1.0
        
    def run(self):

        noff = self.input.shape[0]

        lp = bs.lowpass(self.input, self.lptau)
        hp = bs.highpass(self.input, self.hptau)

        Mi9 = bs.rect(1.0-lp[:,0:noff-2,:],0)
        Mi1 = bs.rect(hp[:,1:noff-1,:],0)
        Mi4 = bs.rect(lp[:,2:noff-0,:],0)

        gexc = bs.rect(Mi1,0)
        ginh = bs.rect(Mi9+Mi4,0)

        T4a = (self.Eexc*gexc+self.Einh*ginh)/(gexc+ginh+self.gleak)
        # T4a_rect=bs.rect(T4a,0)

        self.result = T4a
        return T4a