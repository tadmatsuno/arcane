import numpy as np
import pandas
import shapely
from arcane_dev.utils import utils
from scipy.io import readsav
import os
import warnings
from astropy.constants import N_A
from arcane_dev.mdlatm import marcs
from arcane_dev.mdlatm.base import ModelAtm

data_dir = '/mnt/d/model_atm/AVG3D/' ## CHANGE THIS 

allmodels = readsav(os.path.join(data_dir,'avg3d.sav'))['a']


def read_model(teff,logg,mh):
    
    model = {}
    model['modeltype'] = 'avg3d'
    model['comment'] = ''
    model['input_parameters'] = (teff,logg,mh)

    tgmodels = allmodels[(allmodels['TEFF']==teff)&(allmodels['LOGG']==logg)]
    if len(tgmodels)==0:
        raise ValueError(f'Teff = {teff} logg = {logg} combinations do not exist')
    _mh = np.clip(mh,np.min(tgmodels['FEH']),np.max(tgmodels['FEH']))
    if mh != mh:
        warnings.warn(f'[Fe/H]={mh} is outside of grid. Extrapolated.')
    m1 = tgmodels[tgmodels['FEH']==_mh][0]
    model['modelname'] = m1['NAME'].decode()
    model['teff'] = m1['TEFF']
    model['logg'] = m1['LOGG']
    model['gravity'] = 10.**m1['LOGG']
    model['mass'] = 0.0 # TBC
    model['geometry'] = 'plane-parallel' # TBC
    model['m_h'] = m1['FEH']
    model['alpha_m'] = 0.0 # TBC
    
    model['ndepth'] = len(m1['LGTROSS'])
    model['lgTauR'] = m1['LGTROSS']
    model['T'] = m1['T']
    model['Pg'] = m1['PG']
    model['Pe'] = m1['PE']
    model['Mu'] = m1['MU']*N_A.value
    model['AlphaTauR'] = m1['ALPHA_ROSS']
    model['KappaRoss'] = m1['KAPPA_ROSS']
    model['Density'] = m1['RHO']
    model['Pturb'] = m1['PTURB']
    model['Prad'] = m1['PRAD']

    return model
 
def get_model(teff, logg, mh, alphafe=None, 
    outofgrid_error=False, check_interp=False):

    if not alphafe is None:
        warnings.warn('[Alpha/Fe] variation is not available in this model')

    outside = 'nearest'
    if outofgrid_error:
        outside = 'error'
    
    try:
        teff1, teff2, t_success = utils.get_grid_value(allmodels['TEFF'],teff,outside=outside)
    except ValueError:
        raise ValueError('teff out of range')
    try:
        logg1, logg2, g_success = utils.get_grid_value(allmodels['LOGG'],logg,outside=outside)
    except ValueError:
        raise ValueError('logg out of range')
    try:
        mh1, mh2, m_success = utils.get_grid_value(allmodels['FEH'],mh,outside=outside)
    except ValueError:
        raise ValueError('mh out of range')
    params = {'111':[teff1,logg1,mh1],
                '112':[teff1,logg1,mh2],
                '121':[teff1,logg2,mh1],
                '122':[teff1,logg2,mh2],
                '211':[teff2,logg1,mh1],
                '212':[teff2,logg1,mh2],
                '221':[teff2,logg2,mh1],
                '222':[teff2,logg2,mh2],}
    models = {}
    for grid_key in params.keys():
        models[grid_key] = read_model(*params[grid_key])

    ## Interpolation in mh
    alpha_values = {'T':1.-(teff/4000.)**2.0,\
        'Pe':1-(teff/3500)**2.5,
        'Pg':1-(teff/4100)**4.,
        'KappaRoss':1.0-(teff/3700.)**3.5}
    if mh1==mh2:
        for grid_key in ['11','12','21','22']:
            models[f'{grid_key}0'] = models[f'{grid_key}1'].copy()   
    else:
        mw2 = (mh-mh1)/(mh2-mh1)
        for grid_key in ['11','12','21','22']:
            models[f'{grid_key}0'] = \
                marcs.interp_model2(models[f'{grid_key}1'],models[f'{grid_key}2'],mw2,alpha=alpha_values)

    # Interpolation in logg
    alpha_values = {'T':0.3,\
        'Pe':0.05,
        'Pg':0.06,
        'KappaRoss':-0.12}
    if logg1==logg2:
        for grid_key in ['1','2']:
          models[f'{grid_key}00'] = models[f'{grid_key}10'].copy()   
    else:
        gw2 = (logg-logg1)/(logg2-logg1)
        for grid_key in ['1','2']:
            models[f'{grid_key}00'] = \
                marcs.interp_model2(models[f'{grid_key}10'],models[f'{grid_key}20'],gw2,alpha=alpha_values)

    # Interpolation in teff
    alpha_values = {'T':0.15,\
        'Pe':0.3,
        'Pg':-0.4,
        'KappaRoss':-0.15}
    if teff1==teff2:
        models[f'000'] = models[f'100'].copy()   
    else:
        tw2 = (teff-teff1)/(teff2-teff1)
        models[f'000'] = \
            marcs.interp_model2(models[f'100'],models[f'200'],tw2,alpha=alpha_values)

    if check_interp:
        return models
    else:
        return models['000']
    

class Avg3d(ModelAtm):
    '''
    Class for average 3D model atmospheres
    '''
    def __init__(self,*args,**kwargs):
        super(Avg3d,self).__init__(*args,**kwargs)
    def write(self,filename):
        marcs.write_marcs(filename,self)
    def resample(self,lgTauR):
        return marcs.resample(self,lgTauR)