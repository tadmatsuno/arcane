from arcane.synthesis import moog,grid,readvald,turbospectrum
from arcane.mdlatm import marcs
from arcane.utils import utils
import numpy as np
import matplotlib.pyplot as plt
import os

moog.set_moogsilent_path("/mnt/d/MOOG/mymoog17scat/MOOGSILENT/")

datadir = os.path.join(os.path.dirname(__file__),'data')
valdlinelist = readvald.readvald(os.path.join(datadir,'Vald_stellar_short_hfs'))

hd122563model = os.path.join(datadir,'hd122563_marcs.mod')
if not os.path.exists(hd122563model):
    model = marcs.get_marcs_mod(teff=4616,logg=1.5,mh=-2.7)
    model.write(hd122563model)
#model = marcs.read_marcs(hd122563model)

wvl_ts, flx_ts = turbospectrum.synth(\
    linelist=valdlinelist,
    run_id='test_ts_marcs',
    marcs_mod_file=hd122563model,
    workdir='output',vt=2.,feh=-2.7)
wvl_moog, flx_moog = moog.synth(\
    linelist=valdlinelist,
    run_id='test_moog_marcsmod',
    marcs_mod_file=hd122563model,vt=2.,feh=-2.7,
    scat=1,
    workdir='output')
wvl_moog_noscat, flx_moog_noscat = moog.synth(\
    linelist=valdlinelist,
    run_id='test_moog_marcsmod',
    marcs_mod_file=hd122563model,vt=2.,feh=-2.7,
    scat=0,
    workdir='output')

plt.plot(wvl_ts,flx_ts,label='Turbospectrum')
plt.plot(wvl_moog,flx_moog,label='MOOG')
plt.plot(wvl_moog_noscat,flx_moog_noscat,label='MOOG no scat',linestyle='dashed')
plt.xlim(5853,5860)
#plt.ylim(0.4,0.6)
plt.legend()