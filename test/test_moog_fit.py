from arcane_dev.spectrum import model
from arcane_dev.synthesis import moog
import iofiles
import matplotlib.pyplot as plt
import pandas
import numpy as np

hd122563 = iofiles.readspip('./DATA/HD122563plsp.op')
chlinelist = pandas.read_csv('./DATA/CHlinelists2.csv',
            dtype={'species':str})
chlinelist = chlinelist.rename(columns={'ep':'chi','gf':'loggf','species':'moog_species'})
chlinelist.loc[:,'wavelength'] = np.abs(chlinelist.loc[:,'wavelength'])


def test_synth():
    synth_model = model.LineSynth1param(moog.synth,\
        synth_parameters={\
            'teff': 4636, 'logg': 1.418, 'feh': -2.60, 'vt': 2.05,
            'A_6':8.43-2.60, 'workdir': './output', 'run_id': 'fit_CH',
            'linelist': chlinelist},
        parameter_to_fit='A_6',niterate=1,
        vfwhm_in=9.478, kw_fit_control={'fix_vFWHM':True},
        samples=[[4322.5,4324.5]])
    synth_model.fit(hd122563['wvl'].values,hd122563['flx'].values)
    a_c0,vFWHM0 = synth_model.synth_parameters['A_6'],\
          synth_model.model_parameters['vFWHM']
    print(a_c0,vFWHM0)
    synth_model = model.LineSynth1param(moog.synth,\
        synth_parameters={\
            'teff': 4636, 'logg': 1.418, 'feh': -2.60, 'vt': 2.05,
            'A_6':8.43-2.60, 'workdir': './output', 'run_id': 'fit_CH',
            'linelist': chlinelist},
        parameter_to_fit='A_6',niterate=1,grid_size=0.05,
        vfwhm_in=9.478, kw_fit_control={'fix_vFWHM':True},
        samples=[[4322.5,4324.5]])
    synth_model.fit(hd122563['wvl'].values,hd122563['flx'].values)
    a_c02,vFWHM02 = synth_model.synth_parameters['A_6'],\
          synth_model.model_parameters['vFWHM']
    print(a_c02,vFWHM02)


    synth_model = model.LineSynth1param(moog.synth,\
        synth_parameters={\
            'teff': 4636, 'logg': 1.418, 'feh': -2.60, 'vt': 2.05,
            'A_6':8.43-2.60, 'workdir': './output', 'run_id': 'fit_CH',
            'linelist': chlinelist},
        snr=200.,
        parameter_to_fit='A_6',niterate=1,
        vfwhm_in=9.478, kw_fit_control={'fix_vFWHM':False},
        samples=[[4322.5,4324.5]])
    synth_model.fit(hd122563['wvl'].values,hd122563['flx'].values)
    a_c1,vFWHM1 = synth_model.synth_parameters['A_6'],\
          synth_model.model_parameters['vFWHM']
    print(a_c1,vFWHM1)


    synth_model = model.LineSynth1param(moog.synth,\
        synth_parameters={\
            'teff': 4636, 'logg': 1.418, 'feh': -2.60, 'vt': 2.05,
            'A_6':8.43-2.60, 'workdir': './output', 'run_id': 'fit_CH',
            'linelist': chlinelist},
        parameter_to_fit='A_6',niterate=1,grid_size=0.0,
        vfwhm_in=9.478, kw_fit_control={'fix_vFWHM':True},
        samples=[[4322.5,4324.5]])
    synth_model.fit(hd122563['wvl'].values,hd122563['flx'].values)
    a_c2,vFWHM2 = synth_model.synth_parameters['A_6'],\
          synth_model.model_parameters['vFWHM']
    print(a_c2,vFWHM2)

    synth_modelmulti = model.LineSynth(moog.synth,\
        synth_parameters={\
            'teff': 4636, 'logg': 1.418, 'feh': -2.60, 'vt': 2.05,
            'A_6':8.43-2.60, 'workdir': './output', 'run_id': 'fit_CH',
            'linelist': chlinelist},
        parameters_to_fit=['A_6'],niterate=1,
        vfwhm_in=9.478, kw_fit_control={'fix_vFWHM':True},
        samples=[[4322.5,4324.5]])
    synth_modelmulti.fit(hd122563['wvl'].values,hd122563['flx'].values)
    a_c3,vFWHM3 = synth_modelmulti.synth_parameters['A_6'],\
          synth_modelmulti.model_parameters['vFWHM']
    print(a_c3,vFWHM3)


    fig,ax = plt.subplots(1,1,figsize=(5,5))
    plot_mask = (4320<hd122563['wvl'])&(hd122563['wvl']<4328)
    synth_model.synth_parameters['A_6'] = a_c0
    synth_model.model_parameters['vFWHM'] = vFWHM0
    ax.plot(hd122563[plot_mask]['wvl'],
            synth_model.evaluate(hd122563[plot_mask]['wvl'].values),\
            'C0-',label='Fix, interp')
    ax.plot(hd122563[plot_mask]['wvl'],
            synth_model.evaluate(hd122563[plot_mask]['wvl'].values,force_recompute=True),\
            'C7--',label='Fix, recomputed')

    synth_model.synth_parameters['A_6'] = a_c1
    synth_model.model_parameters['vFWHM'] = vFWHM1
    ax.plot(hd122563[plot_mask]['wvl'],
            synth_model.evaluate(hd122563[plot_mask]['wvl'].values),\
            'C1-',label='Fit, interp')
    synth_model.synth_parameters['A_6'] = a_c2
    synth_model.model_parameters['vFWHM'] = vFWHM2
    ax.plot(hd122563[plot_mask]['wvl'],
            synth_model.evaluate(hd122563[plot_mask]['wvl'].values,force_recompute=True),\
            'C2-',label='Fix, recompute')
    ax.plot(hd122563['wvl'],hd122563['flx'],'ko',ms=1.)
    ax.legend()
    ax.set(xlim=(4320,4327),ylim=(0.7,1.1))
    fig.savefig('output/test_moog_fit.png',dpi=300)


if __name__ == '__main__':
    test_synth()
