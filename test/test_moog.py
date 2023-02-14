from arcane_dev.synthesis import moog
from arcane_dev.utils import utils
import pandas
import matplotlib.pyplot as plt
import iofiles
import numpy as np

arcturus = iofiles.readspip('./HD122563plsp.op')
chlinelist = pandas.read_csv('./CHlinelists2.csv',dtype={'species':str})
chlinelist = chlinelist.rename(columns={'ep':'chi','gf':'loggf','species':'moog_species'})
chlinelist.loc[:,'wavelength'] = np.abs(chlinelist.loc[:,'wavelength'])

def plot_synth_CH(synth_func,ismoog,outfile):
    fig,ax = plt.subplots(1,1,figsize=(5,5))
    wvl0,flx0 = synth_func(linelist=chlinelist,teff=4636,logg=1.418,vt=2.05, feh= - 2.60,A_6=5.220)
    flx_smoothed0 = utils.smooth_spectrum(wvl0,flx0,vfwhm=9.478)

    if ismoog:
        wvl1,flx1 = synth_func(linelist=chlinelist,teff=4636,logg=1.418,vt=2.05, feh= - 2.60,A_6=5.220,\
            I_106_00112=0.5,I_106_00113=0.5)
    else:
        pass
    flx_smoothed1 = utils.smooth_spectrum(wvl1,flx1,vfwhm=9.478)
    ax.plot(wvl0,flx_smoothed0,'C0-',label='Fiducial')
    ax.plot(wvl1,flx_smoothed1,'C1--',label='12C/13C = 1')
    ax.plot(arcturus['wvl'],arcturus['flx'],'ko',ms=1.)
    ax.set(xlim=(4320,4327),ylim=(0.7,1.1))
    ax.legend()
    fig.savefig(outfile)

def test_moog():
    plot_synth_CH(moog.synth,True,'./moog_CH_HD122563.png')


if __name__ == '__main__':
    test_moog()