from arcane.synthesis import moog
from arcane.utils import utils
import pandas
import matplotlib.pyplot as plt
import iofiles
import numpy as np
import unittest
from arcane.synthesis import readvald


hd122563 = iofiles.readspip('./DATA/HD122563plsp.op')

linelist = pandas.read_csv('./vald/stellar_short.csv',dtype={'moog_species':str})
vald_stellar = readvald.read_valdshort('./vald//Vald_stellar_short')
vald_stellar_long = readvald.read_valdlong('./vald/Vald_stellar_long')
vald_long = readvald.read_valdlong('./vald/240122Valdlong_3600_7000_solar0.001_hfssolar.txt')
class TestLinelist(unittest.TestCase):
    def test_linelist_moog(self):
        wvl0, flx0 = moog.synth(linelist=linelist,teff=4636,logg=1.418,vt=2.05, feh= - 2.60,A_6=5.220,A_56=-1.80,
            run_id='test_moog0_csv',workdir='./output')
        wvl1,flx1 = moog.synth(linelist=vald_stellar,teff=4636,logg=1.418,vt=2.05, feh= - 2.60,A_6=5.220,A_56=-1.80,
            run_id='test_moog1_vald',workdir='./output')
        self.assertTrue(np.allclose(wvl0,wvl1,atol=1.0e-4))
        self.assertTrue(np.allclose(flx0,flx1,atol=1.0e-3))
        wvl2,flx2 = moog.synth(linelist=vald_stellar_long,teff=4636,logg=1.418,vt=2.05, feh= - 2.60,A_6=5.220,A_56=-1.80,
            run_id='test_moog2_vald_long',workdir='./output')
        self.assertTrue(np.allclose(wvl1,wvl2,atol=1.0e-4))
        self.assertTrue(np.allclose(flx1,flx2,atol=1.0e-4))

def test_synth_CH(outfile,synth_code='moog'):
    fig,ax = plt.subplots(1,1,figsize=(5,5))

    synth_func = globals()[synth_code].synth #synth_func = moog.synth for example,

    wvl0,flx0 = synth_func(linelist=linelist,teff=4636,logg=1.418,vt=2.05, feh= - 2.60,A_6=5.220,A_56=-1.80,
        run_id=f'test_{synth_code}0',workdir='./output')
    flx_smoothed0 = utils.smooth_spectrum(wvl0,flx0,vfwhm=9.478)

#    if synth_code == 'moog':
#        isotpes = {'I_106_00112':0.5,'I_106_00113':0.5}
#    else:
#        isotpes = {'I_6_12':0.5,'I_6_13':0.5}

#    wvl1,flx1 = synth_func(linelist=chlinelist,teff=4636,logg=1.418,vt=2.05, feh= - 2.60,A_6=5.220,\
#        run_id=f'test_{synth_code}I',workdir='./output',**isotpes)
#    flx_smoothed1 = utils.smooth_spectrum(wvl1,flx1,vfwhm=9.478)
##    ax.plot(wvl0,flx_smoothed0,'C0-',label='Fiducial')
#    ax.plot(wvl1,flx_smoothed1,'C1--',label='12C/13C = 1')
#    ax.plot(hd122563['wvl'],hd122563['flx'],'ko',ms=1.)
#    ax.set(xlim=(4320,4327),ylim=(0.7,1.1))
#    ax.legend()
#    fig.savefig(outfile)

#class TestMoog(unittest.TestCase):
#    def test_synth_CH(self):
#        test_synth_CH('./output/moog_CH_HD122563.pdf',synth_code='moog')

def test_longline(outfile,synth_code='moog'):
    fig,ax = plt.subplots(1,1,figsize=(5,5))

    synth_func = globals()[synth_code].synth #synth_func = moog.synth for example,

    wvl0,flx0 = synth_func(linelist=vald_long,teff=4636,logg=1.418,vt=2.05, feh= - 2.60,A_6=5.220,A_56=-1.80,
        run_id=f'test_{synth_code}_optical',workdir='./output')
    flx_smoothed0 = utils.smooth_spectrum(wvl0,flx0,vfwhm=9.478)



#def test_moog():
#    test_synth_CH('./output/moog_CH_HD122563.pdf',synth_code='moog')

if __name__ == '__main__':
    unittest.main()