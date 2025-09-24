import numpy as np
import pandas
from arcane.utils import utils
from scipy.interpolate import CubicSpline
from arcane.mdlatm.base import ModelAtm
import os
import json
import shutil

#data_dir = '/mnt/d/model_atm/MARCS/' ## CHANGE THIS 

src_path = os.path.expanduser("~/.arcanesrc")

def find_data_dir():
    global data_dir
    # This function reads the location of MOOGSILENT from .arcanesrc
    arcane_config = json.load(open(src_path,"r"))
    if not "marcs_dir" in arcane_config.keys():
        print("marcs_dir is not set in the .arcanesrc file")
        print("Call set_marcs_path to set the path")
        return
    data_dir  = arcane_config["marcs_dir"]
    if not os.path.exists(data_dir):
        print("MARCS data directory does not exist at the specified path:{0:s}".format(data_dir))
        print("Call set_marcs_path to set the path")
        return
    print("MARCS data directory location: {0:s}".format(data_dir))
    return

def set_marcs_path(marcs_path):
    if os.path.exists(marcs_path):
        shutil.copy(src_path,src_path+"_old")
        arcane_setup = json.load(open(src_path,"r"))
        arcane_setup["marcs_dir"] = marcs_path
        json.dump(arcane_setup,open(src_path,"w"))
    find_data_dir()

find_data_dir()

with open(data_dir+'MARCS_avai.dat') as fout:
  grid_value = {}
  for line in fout.readlines():
    key = line[0:10].strip()
    if key == 'teff':
      val = np.array([int(val) for val in line[10:].split()])
    else:
      val = np.array([float(val) for val in line[10:].split()])
    grid_value[key] = val

  
grid = pandas.read_csv(data_dir+'MARCS_grid.csv',index_col=None)

def get_filename1(geometry,teff,logg,mh,alpha=None):
  assert geometry in ['p','s'], 'Geometry has to be either p or s.'
  if alpha is None:
    g1 = grid[(grid['geometry']==geometry)&(grid['teff']==teff)&(grid['logg']==logg)&\
      (grid['mh']==mh)&(grid['comp']=='st')]
  elif isinstance(alpha, str):
    g1 = grid[(grid['geometry']==geometry)&(grid['teff']==teff)&(grid['logg']==logg)&\
      (grid['mh']==mh)&(grid['comp']==alpha)]
  else:
    g1 = grid[(grid['geometry']==geometry)&(grid['teff']==teff)&(grid['logg']==logg)&\
      (grid['mh']==mh)&(grid['alphafe']==alpha)]
  assert len(g1)<=1,f'Cannot identify unique model {teff} {logg} {mh} {alpha}'
  assert len(g1)>=1,f'No model exist {teff} {logg} {mh} {alpha}'
  return data_dir+g1.iloc[0]['filename']


def resample_model(model,lgtauRnew):
  new_model = model.copy()
  new_model['ndepth'] = len(lgtauRnew)
  new_model['lgTauR'] = lgtauRnew
  for key in model.keys():
    if (type(model[key]) is np.ndarray) and (key != 'lgTauR'):
      cs = CubicSpline(model['lgTauR'],model[key])
      new_model[key] = cs(new_model['lgTauR'])
  return new_model
      


def interp_model2(model1,model2,w2,alpha={},\
  interp_in_log=['Pe','Pg','Prad','Pturb','KappaRoss','Density','RHOX'],
  skip_keys = ['filename','modeltype','modelname','last_iteration','ndepth','lgTauR','geometry','comment','input_parameters']):

  '''
    model1 needs to be ''the inferior'' model 
    alpha is not alpha abundance!! 
  ''' 
  assert model1['geometry']==model2['geometry'],\
    'Interpolation error, model geometry mismatch'
  assert model1['modeltype']==model2['modeltype'],\
    'Interpolation error, modeltype mismatch'
  model_new = MARCS()
  model_new['comment'] = ''
  model_new['comment'] += model1['comment']
  model_new['comment'] += model2['comment']
  model_new['modeltype'] = model1['modeltype']
  model_new['geometry'] = model1['geometry']

  ## No need to resample if lgTauR are the same
  tauR1 = model1['lgTauR']
  tauR2 = model2['lgTauR']
  is_resample = True
  if len(tauR1)==len(tauR2):
    if np.all(tauR1==tauR2):
      is_resample = False
      model_new['lgTauR'] = tauR1
  if is_resample:
    print('Resampling required')
    model_new['lgTauR'] = tauR1[(np.min(tauR2)<=tauR1)&(tauR1<=np.max(tauR2))]
    model2 = resample_model(model2,model_new['lgTauR'])
  model_new['ndepth'] = len(model_new['lgTauR'])

  def get_weight(key):
    if key in alpha.keys():
      a = alpha[key]
    else:
      a = 0.0
    return w2**(1.0-a) 

  model_new['filename'] = ''
  model_new['modelname'] = 'interp_'+model1['modelname']+'_'+model2['modelname']
  model_new['last_iteration'] = 'interp'
  for key in model1.keys():
    ww = get_weight(key)
    if key in skip_keys:
      continue
    elif key == 'logg':
      model_new['logg'] = (1.-ww)*model1['logg'] + ww*model2['logg']
      model_new['gravity'] = 10.**model_new['logg']
    elif key == 'abundance':
      model_new['abundance'] = {}
      for key2 in model1[key].keys():
        model_new[key][key2] = (1.-ww)*model1[key][key2]+ww*model2[key][key2]
    else:
      if key in interp_in_log:
        model_new[key] = 10.**( (1.-ww)*np.log10(model1[key]) + \
                                ww*np.log10(model2[key]) )
      else:
        model_new[key] = (1.-ww)*model1[key] + ww*model2[key]
  return model_new

def get_marcs_mod(teff, logg, mh, alphafe=None, outofgrid_error=False, check_interp=False):
  '''
  Get marcs file name
  * logg<3.5 then spherical
  * mass is fixed to 1 for spherical
  * vt is fixed to 2 for spherical model and 1 for plane parallel models\
  
  Parameters
  ----------
  alphafe : float or string
    if string, it should be one of 'st','ae','an','ap'
    if None, comp is assumed to be 'st'
    else, the model will also be interpolated in alpha direction. 
    Otherwise, standard composition
  '''
  if logg <= 3.5:
    geometry = 's'
  else:
    geometry = 'p'
  
  outside = 'nearest'
  if outofgrid_error:
    outside = 'error'

  try:
    teff1, teff2, t_success = utils.get_grid_value(grid_value['teff'],teff,outside=outside)
  except ValueError:
    raise ValueError('teff out of range')
  try:
    logg1, logg2, g_success = utils.get_grid_value(grid_value['logg'],logg,outside=outside)
  except ValueError:
    raise ValueError('logg out of range')
  try:
    mh1, mh2, m_success = utils.get_grid_value(grid_value['mh'],mh,outside=outside)
  except ValueError:
    raise ValueError('mh out of range')

  grid_small = grid[((grid['teff']==teff1)|(grid['teff']==teff2))&\
    ((grid['logg']==logg1)|(grid['logg']==logg2))&\
    ((grid['mh']==mh1)|(grid['mh']==mh2))]
  
  params = {'111':[teff1,logg1,mh1],
            '112':[teff1,logg1,mh2],
            '121':[teff1,logg2,mh1],
            '122':[teff1,logg2,mh2],
            '211':[teff2,logg1,mh1],
            '212':[teff2,logg1,mh2],
            '221':[teff2,logg2,mh1],
            '222':[teff2,logg2,mh2],}
  models = {}
  if alphafe is None:
    for grid_key in params.keys():
      models[grid_key] = read_marcs(get_filename1(geometry,*params[grid_key]))
  elif isinstance(alphafe,str):
    for grid_key in params.keys():
      models[grid_key] = read_marcs(get_filename1(geometry,*params[grid_key],alpha=alphafe))
  else:
    alphafe_grid1 = grid_small[grid_small['mh']==mh1]['alphafe'].values
    try:
      alpha1z1, alpha2z1, a_success1 = utils.get_grid_value(alphafe_grid1,alphafe,outside=outside)
    except ValueError:
      raise ValueError('Alpha_fe out of range')
    alphafe_grid2 = grid_small[grid_small['mh']==mh2]['alphafe'].values
    try:
      alpha1z2, alpha2z2, a_success2 = utils.get_grid_value(alphafe_grid2,alphafe,outside=outside)
    except ValueError:
      raise ValueError('Alpha_fe out of range')

    # First interpolate in [alpha/fe]
    try:
      for grid_key in ['111','121','211','221']:
        modela1 = read_marcs(get_filename1(geometry,*params[grid_key],alpha1z1))
        modela2 = read_marcs(get_filename1(geometry,*params[grid_key],alpha2z1))
        if alpha1z1 == alpha2z1:
          models[grid_key] = modela1.copy()
        else:
          aw2z1 = (alphafe-alpha1z1)/(alpha2z1-alpha1z1)
          models[grid_key] = interp_model2(modela1,modela2,aw2z1)
        if not a_success1:
          models[grid_key]['comment'] += 'interp_error_A '
      for grid_key in ['112','122','212','222']:
        modela1 = read_marcs(get_filename1(geometry,*params[grid_key],alpha1z2))
        modela2 = read_marcs(get_filename1(geometry,*params[grid_key],alpha2z2))
        if alpha1z2 == alpha2z2:
          models[grid_key] = modela1.copy()
        else:
          aw2z2 = (alphafe-alpha1z2)/(alpha2z2-alpha1z2)
          models[grid_key] = interp_model2(modela1,modela2,aw2z2)
        if not a_success2:
          models[grid_key]['comment'] += 'interp_error_A '
    except:
      for grid_key in params.keys():
        models[grid_key] = read_marcs(get_filename1(geometry,*params[grid_key]))
        models[grid_key]['comment'] += 'interp_error_A '
  
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
        interp_model2(models[f'{grid_key}1'],models[f'{grid_key}2'],mw2,alpha=alpha_values)
  if not m_success:
    for grid_key in ['11','12','21','22']:
      models[f'{grid_key}0']['comment'] += 'interp_error_M '
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
        interp_model2(models[f'{grid_key}10'],models[f'{grid_key}20'],gw2,alpha=alpha_values)
  if not g_success:
    for grid_key in ['1','2']:
      models[f'{grid_key}00']['comment'] += 'interp_error_g '
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
        interp_model2(models[f'100'],models[f'200'],tw2,alpha=alpha_values)
  if not t_success:
    models['000']['comment'] += 'interp_error_t '
  if check_interp:
    return models
  else:
    return models['000']

get_model = get_marcs_mod        



def write_marcs(filename, marcs_model):
  '''
  Read marcs .mod files.
  The output is a dictionary containing the model structure
  '''
  with open(filename,'w') as f:
    # line 1
    f.write(marcs_model['modelname']+'\n')
    # line 2
    f.write('{0:7.0f}      Teff [K].         Last iteration; yyyymmdd={1:s}\n'.format(\
      marcs_model['teff'],marcs_model['last_iteration']))
    # line 3
    f.write('{0:12.4E} Flux [erg/cm2/s]\n'.format(marcs_model['Flux']))
    # line 4
    f.write('{0:12.4E} Surface gravity [cm/s2]\n'.format(marcs_model['gravity']))
    # line 5
    f.write('{0:5.1f}        Microturbulence parameter [km/s]\n'.format(marcs_model['vt']))
    # line 6 
    f.write('{0:5.1f}        No mass for plane-parallel models\n'.format(marcs_model['mass']))
    # line 7
    f.write('{0:+6.2f}{1:+6.2f} Metallicity [Fe/H] and [alpha/Fe]\n'.format(marcs_model['m_h'],marcs_model['alpha_m']))
    # line 8
    f.write('{0:12.4E} 1 cm radius for plane-parallel models\n'.format(marcs_model['radius']))
    # line 9
    if marcs_model['radius']==1.0:
      f.write('{0:12.4E} Luminosity [Lsun]\n'.format(marcs_model['luminosity']))
    else:
      f.write('{0:12.5f} Luminosity [Lsun]\n'.format(marcs_model['luminosity']))
    # line 10
    f.write('{0:6.2f}{1:5.2f}{2:6.3f}{3:5.2f} are the convection parameters: alpha, nu, y and beta\n'.format(\
      marcs_model['conv_alpha'],marcs_model['conv_nu'],marcs_model['conv_y'],marcs_model['conv_beta']))
    # line 11
    f.write('{0:9.5f}{1:8.5f}{2:9.2E} are X, Y and Z, 12C/13C={3:2.0f} \n'.format(\
      marcs_model['X'],marcs_model['Y'],marcs_model['Z'],marcs_model['12C13C']))

    # line 12, skip
    f.write('Logarithmic chemical number abundances, H always 12.00\n')
    # Abundances 
    i10 = 0
    for abund in marcs_model['abundance'].values():
      f.write('{0:7.2f}'.format(abund))
      i10 += 1
      if i10==10:
        f.write('\n')
        i10 = 0
    f.write('\n')
    # Number of depth points
    f.write('{0:4d} Number of depth points\n'.format(marcs_model['ndepth']))
    f.write('Model structure\n')
    # Model structure
    f.write(' k lgTauR  lgTau5    Depth     T        Pe          Pg         Prad       Pturb\n')
    for ii in range(marcs_model['ndepth']):
      f.write('{0:3d} {1:5.2f} {2:7.4f} {3:10.3E} {4:7.1f} {5:11.4E} {6:11.4E} {7:11.4E} {8:11.4E}\n'.format(\
        ii+1,
        marcs_model['lgTauR'][ii],
        marcs_model['lgTau5'][ii],
        marcs_model['Depth'][ii],
        marcs_model['T'][ii],
        marcs_model['Pe'][ii],
        marcs_model['Pg'][ii],
        marcs_model['Prad'][ii],
        marcs_model['Pturb'][ii]))
    f.write(' k lgTauR    KappaRoss   Density   Mu      Vconv   Fconv/F      RHOX\n')
    for ii in range(marcs_model['ndepth']):
      f.write('{0:3d} {1:5.2f} {2:11.4E} {3:11.4E} {4:5.3f} {5:10.3E} {6:7.5f} {7:13.6E}\n'.format(\
        ii+1,
        marcs_model['lgTauR'][ii],
        marcs_model['KappaRoss'][ii],
        marcs_model['Density'][ii],
        marcs_model['Mu'][ii],
        marcs_model['Vconv'][ii],
        marcs_model['Fconv/F'][ii],
        marcs_model['RHOX'][ii]))
    f.write('Assorted logarithmic partial pressures\n')
    f.write(' k  lgPgas   H I    H-     H2     H2+    H2O    OH     CH     CO     CN     C2\n')
    for ii in range(marcs_model['ndepth']):
      f.write('{0:3d} {1:6.3f} {2:6.2f} {3:6.2f} {4:6.2f} {5:6.2f} {6:6.2f} {7:6.2f} {8:6.2f} {9:6.2f} {10:6.2f} {11:6.2f}\n'.format(\
        ii+1,
        marcs_model['lgPgas'][ii],
        marcs_model['H_I'][ii],
        marcs_model['H-'][ii],
        marcs_model['H2'][ii],
        marcs_model['H2+'][ii],
        marcs_model['H2O'][ii],
        marcs_model['OH'][ii],
        marcs_model['CH'][ii],
        marcs_model['CO'][ii],
        marcs_model['CN'][ii],
        marcs_model['C2'][ii]))
    f.write(' k    N2     O2     NO     NH     TiO   C2H2    HCN    C2H    HS     SiH    C3H\n')
    for ii in range(marcs_model['ndepth']):
      f.write('{0:3d} {1:6.2f} {2:6.2f} {3:6.2f} {4:6.2f} {5:6.2f} {6:6.2f} {7:6.2f} {8:6.2f} {9:6.2f} {10:6.2f} {11:6.2f}\n'.format(\
        ii+1,
        marcs_model['N2'][ii],
        marcs_model['O2'][ii],
        marcs_model['NO'][ii],
        marcs_model['NH'][ii],
        marcs_model['TiO'][ii],
        marcs_model['C2H2'][ii],
        marcs_model['HCN'][ii],
        marcs_model['C2H'][ii],
        marcs_model['HS'][ii],
        marcs_model['SiH'][ii],
        marcs_model['C3H'][ii]))
    f.write(' k    C3     CS     SiC   SiC2    NS     SiN    SiO    SO     S2     SiS   Other\n')
    for ii in range(marcs_model['ndepth']):
      f.write('{0:3d} {1:6.2f} {2:6.2f} {3:6.2f} {4:6.2f} {5:6.2f} {6:6.2f} {7:6.2f} {8:6.2f} {9:6.2f} {10:6.2f} {11:6.2f}\n'.format(\
        ii+1,
        marcs_model['C3'][ii],
        marcs_model['CS'][ii],
        marcs_model['SiC'][ii],
        marcs_model['SiC2'][ii],
        marcs_model['NS'][ii],
        marcs_model['SiN'][ii],
        marcs_model['SiO'][ii],
        marcs_model['SO'][ii],
        marcs_model['S2'][ii],
        marcs_model['SiS'][ii],
        marcs_model['Other'][ii]))



def read_marcs(filename):
  '''
  Read marcs .mod files.
  The output is a dictionary containing the model structure
  '''
  assert filename.endswith('.mod'),\
    'This function is to read .mod files from MARCS'
  marcs_model = MARCS()
  marcs_model['filename'] = filename
  marcs_model['modeltype'] = 'marcs'
  marcs_model['comment'] = ''
  with open(filename) as f:
    # line 1
    line = f.readline().rstrip()
    marcs_model['modelname'] = line
    # line 2
    line = f.readline().rstrip()
    assert 'teff' in line.lower(), \
      'line 2 should be for Teff. file format might have changed'  
    marcs_model['teff'] = float(line[0:7])
    str1 = 'Last iteration; yyyymmdd='
    idx1 = line.find(str1)+len(str1)
    marcs_model['last_iteration'] = line[idx1:idx1+8]
    # line 3
    line = f.readline().rstrip()
    assert 'flux' in line.lower(), \
      'line 3 should be for Flux. file format might have changed'  
    marcs_model['Flux'] = float(line[0:12])
    # line 4
    line = f.readline().rstrip()
    assert 'gravity' in line.lower(), \
      'line 4 should be for surface gravity. file format might have changed'  
    marcs_model['gravity'] = float(line[0:12])
    marcs_model['logg'] = np.log10(float(line[0:12]))
    # line 5
    line = f.readline().rstrip()
    assert 'microturbulence' in line.lower(), \
      'line 5 should be for microturblence. file format might have changed'  
    marcs_model['vt'] = float(line[0:5])
    # line 6 
    line = f.readline().rstrip()
    marcs_model['mass'] = float(line[0:5])
    assert 'mass' in line.lower(), \
      'line 6 should be for mass. file format might have changed'
    if marcs_model['mass'] == 0.0:
      marcs_model['geometry'] = 'plane-parallel'
    else:
      marcs_model['geometry'] = 'spherical'
    # line 7
    line = f.readline().rstrip()
    assert ('metallicity' in line.lower()) and ('alpha/fe' in line.lower()),\
      'line 7 should be for [Fe/H] and [alpha/Fe]. '+\
      'file format might have changed.'
    marcs_model['m_h']     = float(line[0:6])
    marcs_model['alpha_m'] = float(line[6:12])
    # line 8
    line = f.readline().rstrip()
    assert 'radius' in line.lower(),\
      'line 8 should be for radius'
    marcs_model['radius'] = float(line[0:12]) 
    # line 9
    line = f.readline().rstrip()
    assert 'luminosity' in line.lower(),\
      'line 9 should be for radius'
    marcs_model['luminosity'] = float(line[0:12]) 
    # line 10
    line = f.readline().rstrip()
    assert 'convection' in line.lower(),\
      'line 10 should be for convection parameters'+\
      'file format might have changed.'
    marcs_model['conv_alpha'] = float(line[0:6]) 
    marcs_model['conv_nu']    = float(line[6:11]) 
    marcs_model['conv_y']     = float(line[11:17]) 
    marcs_model['conv_beta']  = float(line[17:22]) 
    # line 11
    line = f.readline().rstrip()
    assert 'X, Y and Z' in line,\
      'line 11 should be for X, Y, Z'+\
      'file format might have changed.' 
    values = line.split()
    marcs_model['X'] = float(values[0])
    marcs_model['Y'] = float(values[1])
    marcs_model['Z'] = float(values[2])
    str12c13c = '12C/13C='
    i12c13cstart = line.find(str12c13c)+len(str12c13c)
    marcs_model['12C13C'] = int(line[i12c13cstart:i12c13cstart+3])
    # line 12, skip
    line = f.readline().rstrip()
    assert 'abundances' in line.lower(),\
      'line 12 should contains "abundances"'+\
      'file format might have changed.' 
    # Abundances 
    line = f.readline().rstrip()
    abundance = []
    while (not 'depth' in line):
      abundance += [line[7*ii:7*(ii+1)] for ii in range(10)]
      line = f.readline().rstrip()
    abundance 
    marcs_model['abundance'] = \
      dict(zip(range(1,len(abundance)+1),\
               np.array([float(ab) for ab in abundance if ab.strip()!=''])))
    # Number of depth points
    assert 'depth' in line,\
      'right after the abundance, the number of points should be described'+\
      'file format might have changed.' 
    ndepth = int(line[0:4])
    marcs_model['ndepth'] = ndepth 
    # skip one line
    line = f.readline().rstrip()
    assert 'model structure' in line.lower(),\
      'Model structure should start here'+\
      'file format might have changed.' 
    # Model structure
    def read_structure():  
      line = f.readline().rstrip().replace('H I','H_I')
      keys = [key.replace(' ','') for key in line.split()[1:]]
      # first column is always k
      for key in keys:
        if key in marcs_model.keys():
          'Key is duplicated. Only the last input will be kept'
        marcs_model[key] = np.zeros(ndepth,dtype=float)
      for ii in range(ndepth):
        line = f.readline().rstrip()
        values = line.split()
        for jj,key in enumerate(keys):
          marcs_model[key][ii] = float(values[jj+1]) # first column is always k
    read_structure() #  k lgTauR  lgTau5    Depth     T        Pe          Pg         Prad       Pturb
    read_structure() #  k lgTauR    KappaRoss   Density   Mu      Vconv   Fconv/F      RHOX
    # skip one line
    line = f.readline().rstrip()
    assert 'partial pressures' in line.lower(),\
      'Partial pressures should start here'+\
      'file format might have changed.' 
    # Partial pressure
    read_structure() #  k  lgPgas   H I    H-     H2     H2+    H2O    OH     CH     CO     CN     C2  
    read_structure() #  k    N2     O2     NO     NH     TiO   C2H2    HCN    C2H    HS     SiH    C3H
    read_structure() #  k    C3     CS     SiC   SiC2    NS     SiN    SiO    SO     S2     SiS   Other
    return marcs_model

read_model = read_marcs

class MARCS(ModelAtm):
  '''
  Class for MARCS model atmospheres
  '''
  def __init__(self, *args, **kwargs):
    super(MARCS, self).__init__(*args, **kwargs)
  def write(self,filename):
    write_marcs(filename,self)
  def resample(self,lgtauRnew):
    return resample_model(self,lgtauRnew)

