import pandas
import numpy as np
import os
from . import solarabundance as sa
from . import isotopic_ratio as ir
import collections
from typing import List,Union

'''
This file includes various input/output related functions that I have been using.
Not every function here is fully tested and verified.
'''

stdadefault = collections.OrderedDict({\
    'VEL':2.0, 'SWPRCC':1.0, 'SWPRIS':1.0, 'zlog':0.00, 'CCDATA':{},\
    'DCDATA2':{}, 'NIMAX':6000, 'EPS':0.0005, 'SWITER':1.0, 'SWPRIT':-1.0,\
    'SWFS':-1.0, 'SWCEQ':0.0, 'SWAMD':1.0, 'SWPF':0.0, 'MAXREP':200.0,\
    'SWSF':1.0, 'EPSSF' :0.01, 'SWPR':-1.0, 'SWCONT':-1.0, 'SWPUN':1.0,\
    'SWPLOT':-1.0, 'SWLPR':-1.0, 'SWIP':-1.0, 'KINDIP':3, 'DIP':10.0,\
    'IDBLD':'TEST', 'SWWNWL':-1.0, 'SWPRLD':-1.0, 'SWPRAC':-1.0, 'SWLAC':-1.0,\
    'COSTH'  :-1.0, 'swipam' :1.0, 'swhcn':-1.0, 'swhcch' :-1.0})
### See the bottom for the decription of parameters (stdahelp)



def checkfileexist(fname: str) -> None:
  '''
  This function checks if the output file exists and ask the user if the user
  wants to overwrite the file. If the user does not want to overwrite the file,
  it raises IOError.
  '''
  if os.path.exists(fname):
    ans = input('WARNING: output file exists!!\n'+
          'Do you want to continue? [y/n]\n')
    while (ans[0] != 'y'):
      if ans[0] == 'n':
        raise IOError('Output file exists')
      elif ans[0] != 'y':
        ans = input('Answer must be either of [y/n]')
  return 

def ts2moog_species(ts_species):
  species_body = ts_species[:ts_species.find(".")]
  species_isotope = ts_species[ts_species.find(".")+1:]
  atoms = []
  isotopes = []
  while(int(species_body)>100):
    atoms.append(species_body[-2:])
    isotopes.append(species_isotope[-3:])
    species_body = species_body[:-2]
    species_isotope = species_isotope[:-3]
  atoms.append(species_body)
  isotopes.append(species_isotope)

  natom = len(atoms)
  sort_idx = np.argsort([int(z) for z in atoms])
  frac_isotope = 1
  for zz,iso in zip(atoms,isotopes):
    if int(iso)>0:
      frac_isotope *= ir.get_fraction(int(zz),int(iso))

  if natom == 1:
    moog_body = atoms[0]
    moog_isotope = "{0:04d}".format(int(isotopes[0]))
  elif natom == 2:
    moog_body = "{0:2d}{1:02d}".format(
        int(atoms[sort_idx[0]]), int(atoms[sort_idx[1]])
    )
    if np.any([int(iso)>=100 for iso in isotopes]):
      moog_isotope = "0000"
    else:
      moog_isotope = "{0:02d}{1:02d}".format(
        int(isotopes[sort_idx[0]]), int(isotopes[sort_idx[1]]) 
      )        
  else: 
    print("3-atom molecules are ignored")
    moog_body = "0"
    moog_isotope = "0000"
  return moog_body,moog_isotope,frac_isotope

def readTSlinelist(filename, species_list = [], wvl_min = -1, wvl_max = -1,
                   max_ion = 2,
                   isotope_scale = True):
  '''
  species_list has to follow moog convention
  '''
  if len(species_list)==0:
    print("All species will be returned")
  if wvl_min <0:
    print("No minimum wavelength provided. No cut will be made")
  if wvl_min <0:
    print("No minimum wavelength provided. No cut will be made")

  def read_species_line(line):
    idx_1 = line.find("'")
    idx_2 = line.find("'",idx_1+1)
    ts_species = line[idx_1+1:idx_2-1].replace(" ","")
    ion, nline = line[idx_2+1:].split()
    ion = int(ion)
    nline = int(nline)
    moog_body,moog_isotope,frac_isotope = ts2moog_species(ts_species)
    moog_species = "{0:s}.{1:1d}{2:2s}".format(moog_body,ion-1,moog_isotope)
    skip_species = moog_body == "0"
    if isotope_scale:
      scaling = np.log10(frac_isotope)
    else:
      scaling = 0.
    skip_species  = ion > max_ion
    return moog_species,nline,skip_species,scaling
  
  moog_species = []
  wvls = []
  loggfs = []
  eps = []
  is_continue = True
  with open(filename,"r") as f:
    line = f.readline()
    while(is_continue):
      if len(line) == 0:
        break
      if line.lstrip()[0] != "'":
        break
      moog_species1,nline,skip_species,scaling = read_species_line(line)
      if len(species_list)>0:
        skip_species = skip_species or (not moog_species1 in species_list)
      f.readline()
      for ii in range(nline):
        line = f.readline().split()
        if skip_species:
          continue
        wvl,ep,loggf = float(line[0]),float(line[1]),float(line[2])
        if (wvl_min > 0 ) & (wvl<wvl_min):
          continue
        if (wvl_max > 0) & (wvl>wvl_max):
          continue
        moog_species.append(moog_species1)
        wvls.append(wvl)
        loggfs.append(loggf + scaling)
        eps.append(ep)
      line = f.readline()
  df_out = pandas.DataFrame.from_dict({\
    "wavelength":wvls,"loggf":loggfs,"expot":eps,"moog_species":moog_species})
  df_out = df_out.sort_values("wavelength")
  return df_out

class Spectrum(pandas.DataFrame):
  '''
  Class for spectrum data.
  In addition to pandas.DataFrame, it has header attribute
  '''
  def __init__(self,*args,**kwargs):
    super().__init__(*args,**kwargs)
    self.header = None
    return
  
  def writespip(self,fileout : str = 'sp.op', overwritecheck : bool = True) -> None:
    writespip(self['wvl'],self['flx'],fileout=fileout,\
              head=self.header,overwritecheck=overwritecheck)
    return

def readspip(filein : str = 'sp.ip', header : bool = False) -> Spectrum:
  with open(filein,'r') as f:
    try:
      dataall = f.read().split('\n') 
      iend = [line.replace(' ','') for line in dataall].index('END')
      head = dataall[:iend+1]
    except:
      iend = 0
      head = []
  
  #spec = pandas.read_csv(filein,skiprows=iend+1,header=None,\
  #                       sep='\s+',names=['wvl','flx'])
  spec = Spectrum(pandas.read_csv(filein,skiprows=iend+1,header=None,\
                           sep='\s+',names=['wvl','flx']))
  spec.header = head
#  if header:
#    return spec,head
#  else:
#    return spec 
  return spec
           


def writespip(wvl : np.ndarray, flx : np.ndarray, fileout : str = 'sp.op',\
              head : list = None, overwritecheck : bool = True) -> None:
  if (len(wvl) != len(flx)):
    raise ValueError('wvl and flx must have the same length')

  if overwritecheck:
    checkfileexist(fileout)

  with open(fileout,'w') as f:
    if head is not None:
      for h in head:
        f.write( h + '\n')
      if not 'END' in head:
        f.write('END\n')
    for ii in range(len(wvl)):
      f.write('{0:15.9f}{1:15.8f}\n'.format(wvl[ii],flx[ii]))
  return

class EWdata(pandas.DataFrame):
  '''
  Class for EW data.
  In addition to pandas.DataFrame, it has a function to write ew.op file
  '''
  def __init__(self,*args,**kwargs) -> None:
    super().__init__(*args,**kwargs)
    return
  def writeewop(self,fileout : str = 'ew.op',overwritecheck : bool = True) -> None:
    writeewop_pandas(self,fileout=fileout)
    return
  

def readewop(filein : str = 'ew.op') -> pandas.DataFrame:
  ftmp = open(filein,'r')
  if 'err_EW' in ftmp.readlines()[1]:
    data = pandas.read_fwf(filein,\
      colspecs = [[0,5],[5,9],[9,16],[16,26],[26,36],[36,44],[44,54],\
                 [54,62],[62,70],[70,78],[78,86],[86,96],[96,106],\
                 [106,116],[116,126],[126,136],[136,146],[146,156],[156,168]],\
      skiprows = 2,\
      names=['No','nelem','ELEMENT','WAVELENGTH','log(GF)','EXP(eV)',\
             'WL(cent)','DWL(o-c)','DEPTH','DDEP','FWHM','EW(mA)',\
             'log(EW/WL)','EWinteg(mA)','Fact','xmin','xmax','err_EW','Ref.'],header=None)
  else:
    data = pandas.read_fwf(filein,\
      colspecs = [[0,5],[5,9],[9,16],[16,26],[26,36],[36,44],[44,54],\
                 [54,62],[62,70],[70,78],[78,86],[86,96],[96,106],\
                 [106,116],[116,126],[126,136],[136,146],[146,158]],\
      skiprows = 2,\
      names=['No','nelem','ELEMENT','WAVELENGTH','log(GF)','EXP(eV)',\
             'WL(cent)','DWL(o-c)','DEPTH','DDEP','FWHM','EW(mA)',\
             'log(EW/WL)','EWinteg(mA)','Fact','xmin','xmax','Ref.'],header=None)
    data.loc[:,'err_EW'] = 0.0
  return data 

def writeewop_pandas(ewdata : Union[pandas.DataFrame,EWdata],\
                      fileout : str = 'ew.op',overwritecheck : bool = True) -> None:
  with open(fileout,'w') as f:
    ## Header
    f.write('{0:10s}{1:10.7f}'.format('DOPL=',0.0)+\
            ' '*10+\
            '{0:10s}\'{1:18s}\'\n'.format('OBJECT  = ',fileout[0:-5]))
    f.write(' No.    ELEMENT WAVELENGTH log(GF) EXP(eV)'+\
            '   WL(cent) DWL(o-c)  DEPTH   DDEP    FWHM'+\
            '     EW(mA) log(EW/WL)EWinteg(mA)     Fact'+\
            '      xmin      xmax    err_EW Ref.\n')
    if 'err_EW' not in ewdata.columns:
      ewdata.loc[:,'err_EW'] = 0.0
    
    for idx in ewdata.index:
      out = '{0:5d}'.format(ewdata.loc[idx,'No'])+\
            '{0:4d}'.format(ewdata.loc[idx,'nelem'])+\
            '{0:>7s}'.format(ewdata.loc[idx,'ELEMENT'])+\
            '{0:10.3f}'.format(ewdata.loc[idx,'WAVELENGTH'])+\
            '{0:10.3f}'.format(ewdata.loc[idx,'log(GF)'])+\
            '{0:8.3f}'.format(ewdata.loc[idx,'EXP(eV)'])+\
            '{0:10.3f}'.format(ewdata.loc[idx,'WL(cent)'])+\
            '{0:8.3f}'.format(ewdata.loc[idx,'DWL(o-c)'])+\
            '{0:8.3f}'.format(ewdata.loc[idx,'DEPTH'])+\
            '{0:8.3f}'.format(ewdata.loc[idx,'DDEP'])+\
            '{0:8.3f}'.format(ewdata.loc[idx,'FWHM'])+\
            '{0:10.3f}'.format(ewdata.loc[idx,'EW(mA)'])+\
            '{0:10.3f}'.format(ewdata.loc[idx,'log(EW/WL)'])+\
            '{0:10.3f}'.format(ewdata.loc[idx,'EWinteg(mA)'])+\
            '{0:10.3f}'.format(ewdata.loc[idx,'Fact'])+\
            '{0:10.3f}'.format(ewdata.loc[idx,'xmin'])+\
            '{0:10.3f}'.format(ewdata.loc[idx,'xmax'])+\
            '{0:10.3f}'.format(ewdata.loc[idx,'err_EW'])+\
            '{0:12}\n'.format(ewdata.loc[idx,'Ref.'])
      f.write(out)

def writeewop(elem : Union[List[int],int], 
              wvl : List[float], ew : List[float],\
              gf : List[float], ev : List[float], 
              fwhm : Union[List[float],float],\
              err_ew : Union[List[float],float,None] = None, 
              refs : Union[List[str], None] = None,
              fileout : str = 'ew.op',\
              dopl : float = 1.0, objname : str = 'Object',\
              overwritecheck : bool = True) -> None:

  if len(wvl) != len(ew):
    raise ValueError('wvl and ew must have the same length')
  if len(wvl) != len(gf):
    raise ValueError('wvl and gf must have the same length')
  if len(wvl) != len(ev):
    raise ValueError('wvl and ev must have the same length')

  try:
    if len(elem)!= len(wvl):
      raise ValueError('wvl and elem must have the same length')
    singleelem = False
  except:
    elem = np.array([elem]*len(wvl))
    singleelem = True

  try:
    if len(fwhm)!= len(wvl):
      raise ValueError('wvl and fwhm must have the same length')
    singlefwhm = False
  except:
    fwhm = np.array([fwhm]*len(wvl))
    singlefwhm = True

  if err_ew is not None:
    if len(err_ew) != len(wvl):
      raise ValueError('wvl and err_ew must have the same length')

  if refs is not None:
    if len(refs)!= len(wvl):
      raise ValueError('wvl and refs must have the same length')
    isref = True 
  else:
    isref = False

  if overwritecheck:
    checkfileexist(fileout)

  depth = ew/fwhm * 2.0 * np.sqrt(np.log(2.0)/np.pi)*1.0e-3
  logew = np.log10(ew/wvl)-3.0

  with open(fileout,'w') as f:
     ## Header
     f.write('{0:10s}{1:10.7f}'.format('DOPL=',dopl)+\
             ' '*10+\
             '{0:10s}\'{1:18s}\'\n'.format('OBJECT  = ',objname))
     f.write(' No.    ELEMENT WAVELENGTH log(GF) EXP(eV)'+\
             '   WL(cent) DWL(o-c)  DEPTH   DDEP    FWHM'+\
             '     EW(mA) log(EW/WL)EWinteg(mA)     Fact'+\
             '      xmin      xmax    err_EW Ref.\n')

     ## Write table
     for ii in range(len(wvl)):
       pnum = elem[ii]%100
       ion = elem[ii]//100
       if ion == 0:
         strelem = '  ????'
       else:
         strelem = ' '*2+sa.get_elemname(pnum)+' '+'I'*ion
         
       out = '{0:5d}'.format(ii)+\
             '{0:4d}'.format(elem[ii])+\
             '{0:7s}'.format(strelem)+\
             '{0:10.3f}'.format(wvl[ii])+\
             '{0:10.3f}'.format(gf[ii])+\
             '{0:8.3f}'.format(ev[ii])+\
             '{0:10.3f}'.format(wvl[ii])+\
             '{0:8.3f}'.format(0.0)+\
             '{0:8.3f}'.format(depth[ii])+\
             '{0:8.3f}'.format(0.0)+\
             '{0:8.3f}'.format(fwhm[ii])+\
             '{0:10.3f}'.format(ew[ii])+\
             '{0:10.3f}'.format(logew[ii])+\
             '{0:10.3f}'.format(ew[ii])+\
             '{0:10.3f}'.format(1.0)+\
             '{0:10.3f}'.format(wvl[ii]-fwhm[ii])+\
             '{0:10.3f}'.format(wvl[ii]+fwhm[ii])+\
             '{0:10.3f}'.format(0.0 if err_ew is None else err_ew[ii])
       if isref:
         out = out + '{0:12s}\n'.format(refs[ii])
       else:
         out = out + ' '*12 + '\n'
       f.write(out)
  return    

   
def readamdb(filein : str = 'amdb.ip', \
    isev : Union[str,bool] = True) -> pandas.DataFrame:
  #  For backward compatibility, isev can be 'y' or 'n'
  if isinstance(isev,str):
    if isev.lower() == 'y':
      isev = True
    elif isev.lower() == 'n':
      isev = False
    else:
      raise ValueError('isev must be y/n')

  # The order of columns are different for isev = True and False
  if isev:
    names = ['nelem','WAVELENGTH','EXP(eV)','log(GF)']
  else:
    names = ['nelem','WAVELENGTH','log(GF)','EXP(eV)']
  data =  pandas.read_fwf(filein,\
                 colspecs=[[0,10],[10,20],[20,30],[30,40]],
                 names = names)
  if isev == 'n':
    data.loc[:,'EXP(eV)'] = data.loc[:,'EXP(eV)']/8065.73
  return data 


def readref(filein='amdb.ip',isev='y'):
  ''' 
  Same as readamdb but reference given in 50..70 can be read 
  '''
  if isev == 'y':
    names = ['nelem','WAVELENGTH','EXP(eV)','log(GF)','ELEM','REF']
  elif isev == 'n':
    names = ['nelem','WAVELENGTH','log(GF)','EXP(eV)','ELEM','REF']
  else:
    raise ValueError('isev must be y/n')
  data =  pandas.read_fwf(filein,\
                 colspecs=[[0,10],[10,20],[20,30],[30,40],[40,50],[50,70]],
                 names = names)
  if isev == 'n':
    data.loc[:,'EXP(eV)'] = data.loc[:,'EXP(eV)']/8065.73
  return data 


def readipew(filein='ipew.dat'):
  namesmaster = ['isatom','lineno','swvl','fwvl','dwvl','awidth',\
           'nstep','dstep','rew','lineid']
  
  masterdata = pandas.DataFrame([],columns=namesmaster)

  nameslines = ['ielm','wvl','gf','ep','lineno','lineid']  
  linesdata = pandas.DataFrame([],columns=nameslines)

  with open(filein,'r') as f:
    state = 0
    for line in f:
      if (state==0):
        state = 1
        continue
      elif (state == 1):
        isatom = line[0:4] 
        lineno = int(line[4:10])
        swvl = float(line[10:20])
        fwvl = float(line[20:30])
        dwvl = float(line[30:40])
        awidth = float(line[40:50])
        nstep = int(line[50:52])
        dstep = float(line[52:60])
        rew = float(line[60:70])
        lineid = int(line[75:80]) 
        ## WRITE TO DATABASE
        masterdataline = pandas.Series([isatom,lineno,swvl,fwvl,dwvl,\
                                        awidth,nstep,dstep,rew,lineid],\
                                        index=namesmaster)
        masterdata = masterdata.append(masterdataline,ignore_index=True)
        state = 2
      elif (state ==2):
        try:
          ielem = int(line[0:10])
        except:
          state = 1
          continue
        wvl  = float(line[10:20])
        gf = float(line[20:30])
        ep = float(line[30:40])
        linesdataline = pandas.Series([ielem,wvl,gf,ep,lineno,lineid],\
                                       index=nameslines)
        linesdata = linesdata.append(linesdataline,ignore_index=True)
  return masterdata,linesdata


def writeipew(masterdata,linedata,fileout='ipew.dat',dopl=1.0,\
              objname='Object',overwritecheck=True,isendzero=False):
  if overwritecheck:
    checkfileexist(fileout)
  with open(fileout,'w') as f:
    f.write('{0:10s}{1:10.7f}'.format('DOPL=',dopl)+\
             ' '*10+\
             '{0:10s}\'{1:18s}\'\n'.format('OBJECT  = ',objname))
    for idx in masterdata.index:
       f.write('{0:4s}'.format(masterdata.loc[idx,'isatom'])+\
               '{0:6d}'.format(int(round(masterdata.loc[idx,'lineno'])))+\
               '{0:10.3f}'.format(masterdata.loc[idx,'swvl'])+\
               '{0:10.3f}'.format(masterdata.loc[idx,'fwvl'])+\
               '{0:10.3f}'.format(masterdata.loc[idx,'dwvl'])+\
               '{0:10.3f}'.format(masterdata.loc[idx,'awidth'])+\
               '{0:2d}'.format(int(round(masterdata.loc[idx,'nstep'])))+\
               '{0:8.2f}'.format(masterdata.loc[idx,'dstep'])+\
               '{0:10.3f}'.format(masterdata.loc[idx,'rew'])+\
               '{0:10d}\n'.format(int(round(masterdata.loc[idx,'lineid']))))
       linehere = linedata[linedata['lineid']==masterdata.loc[idx,'lineid']]
       for jdx in linehere.index:
          f.write('{0:10d}'.format(int(round(linehere.loc[jdx,'ielm'])))+\
                  '{0:10.3f}'.format(linehere.loc[jdx,'wvl'])+\
                  '{0:10.3f}'.format(linehere.loc[jdx,'gf'])+\
                  '{0:10.3f}\n'.format(linehere.loc[jdx,'ep']))
       f.write(' '*10+'{0:10.3f}\n'.format(0.0))
   
    if isendzero:
      f.write('{0:10d}\n'.format(0))
  return
               

def readopew(filein='opew.dat'):
  with open(filein) as f:
    fall = [line.replace(' ','') for line in \
               f.read().split('\n')]
    try:
      iend = fall.index('OPEWRESULT')
    except:
      iend = 0
    try:
      fend = len(fall)-1-fall.index('0')
    except:
      fend = 0
  return pandas.read_fwf(filein,skiprows=iend+1,skipfooter=fend,\
    colspecs = [[0,5],[5,10],[10,15],[15,25],[25,35],[35,45],[45,55],\
               [55,65],[65,75],[75,85],[85,95]],\
    names=['lineno','lineid','ELEMENT','swvl','fwvl','FCLOG','EWDWLL',\
           'REW','dGF','logE','EP'])


def readstda(filein='stda.ip'):
  outdict = {} 
  ccmode = False
  dcmode = False
  with open(filein) as f:
    for line in f:
      if ccmode:
        if line[0:10].replace(' ','').isnumeric():
          elem = int(line[0:10])
          abu  = float(line[10:20])
          modabun[elem] = abu
          continue
        else:
          outdict['CCDATA'] = modabun
          ccmode =False
      if dcmode:
        if line[0:10].replace(' ','').isnumeric():
          elem = int(line[0:10])
          abu  = float(line[10:])
          modiso[elem] = abu
          continue
        else:
          outdict['DCDATA2'] = modiso
          dcmode =False
      if line[0:10] ==  'CCDATA   1':
        ccmode = True
        modabun = {}
      elif line[0:10] ==  'DCDATA2  2':
        dcmode = True
        modiso = {}
      elif line[0:10] ==  '          ':
        pass
      else:
        key = line[0:10].split('=')[0]
        value = line[10:20]
        if value[-1] == '\n':
           value = value[:-1]
        value = value.replace(' ','')
        try:
           value = float(value)
        except:
           pass
        outdict[key] = value
  return outdict


def writestda(paramdict,fileout='stda.ip',overwritecheck=True):
  if overwritecheck: 
    checkfileexist(fileout)
  with open(fileout,'w') as f:
    for stk in stdadefault.keys():
      stkwrite = '{0:s}='.format(stk)
      if not stk in paramdict.keys():
        paramdict[stk] = stdadefault[stk]

      if stk in ['CCDATA','DCDATA2']:
        atommod = paramdict[stk]
        nmod = len(atommod)
        if stk == 'DCDATA2':
          f.write('DCDATA2  2     MOD{0:2d}'.format(nmod)+\
                 '               \n')
        else:
          f.write('CCDATA   1     MOD{0:2d}'.format(nmod)+\
                 '   IPCC=std1999\n')
        for atom in atommod.keys():
           f.write('{0:10d}{1:10.2f}\n'.format(atom,atommod[atom]))
      elif stk == 'EPS':
        f.write('{0:*<10s}'.format(stkwrite)+\
                '{0:7.4f}\n'.format(paramdict[stk]))

      elif stk in ['NIMAX','MAXREP','KINDIP']:
        f.write('{0:*<10s}'.format(stkwrite)+\
                  '{0:8d}\n'.format(int(paramdict[stk])))
      else:
        try:
          f.write('{0:*<10s}'.format(stkwrite)+\
                  '{0:>8s}\n'.format(paramdict[stk]))
          continue
        except:
          pass
        try:
          f.write('{0:*<10s}'.format(stkwrite)+\
                  '{0:8.2f}\n'.format(paramdict[stk]))
          continue
        except:
          pass
  return       


def readopma(filein = 'opma.ipt'):
  ### structure
  ## i  log(tau)  log(ak)  t  log(pg)  prad  pturb  vturb
  ## row_num opt.depth massabs.coef. temp. gas_pres. rad.pres. 
  ## (continued) turb.pres. microturb._vel.
  with open(filein,'r') as f:

    line1 = f.readline()
    params = {'teff':float(line1[25:31]),
              'logg': float(line1[31:37]),
              'feh':float(line1[37:43])}

    line1 = f.readline()
    model = {'tau':[],'logak':[],'t':[],\
             'logPg':[],'Prad':[],'Pturb':[],'vturb':[]}

    for line1 in f:
      model['tau'].append(float(line1[3:10]))
      model['logak'].append(float(line1[10:20]))
      model['t'].append(float(line1[20:30]))
      model['logPg'].append(float(line1[30:40]))
      model['Prad'].append(float(line1[40:50]))
      model['Pturb'].append(float(line1[50:60]))
      model['vturb'].append(float(line1[60:70]))
  return model,params


stdahelp = collections.OrderedDict({\
  'VEL': 'Microturblence', \
  'SWPRCC': 'if >0: Log contents of ipcc.dat (atomic inputs)',\
  'SWPRIS': 'if >0: Log contents of ipdc.dat (molecules inputs)',\
  'zlog': '[M/H]',\
  'CCDATA':'dict type. input abundance change in the following format      \n'+\
           '  pnum: abundance                                              \n'+\
           'In stda.ip, it is written in the following format              \n'+\
           '>>CCDATA   1     MOD j  IPCC=std1999                           \n'+\
           '>>        pp      abun                                         \n'+\
           'the second line repeats j times                                \n'+\
           'where   j: number of changes                                   \n'+\
           '       pp: proton number of the atom                           \n'+\
           '     abun: revised abundance in log(X/H) (i.e. A(X)-12)',\
  'DCDATA2':'dict type. input isotpe ratio change in the following format  \n'+\
            '  molcule id : ratio (see ipdc.dat)                           \n'+\
            'In stda.ip, it is written in the following format             \n'+\
            '>>DCDATA2  2     MOD j                                        \n'+\
            '>>        pp     ratio                                        \n'+\
            'the second line repeats j times                               \n'+\
            'where   j: number of changes                                  \n'+\
            '       pp: molecule id                                        \n'+\
            '     abun: molecule abundance ratio.                          \n'+\
            '           molecule abundance will be multiplied by this ratio',\
  'NIMAX': 'parameter for chemical equilibrium                             \n'+\
           'Maximum number of iteration in newton-rapson method',\
  'EPS':   'parameter for chemical equilibrium                             \n'+\
           'Relative precision criteria to stop newton-rapson method',\
  'SWITER':'parameter for chemical equilibrium                             \n'+\
           'If >0, change in each step will be moderated ',\
  'SWPRIT':'parameter for chemical equilibrium                               '+\
           'If >0, log intermediate results (probably not work; no file unit)',\
  'SWFS':'parameter for chemical equilibrium                               \n'+\
         'If >0, subroutine diec will be used for fast computation           '+\
         'TM could not figure out how this influence the results             ',\
  'SWCEQ':'parameter for chemical equilibrium                              \n'+\
          'If <0, No results will be logged                                \n'+\
          'If =0, log chemical equilibrium results (incl. atomic pressure) \n'+\
          'If >0, detailed results (might not work; no file unit)',\
 'SWAMD': 'If >0, print molecules considered in chemical equilibrium',\
 'SWPF':  'If >0, only deal with molecules                                 \n'+\
          'If =0, deal with molecules and atoms                            \n'+\
          'if <0, only deal with atoms',\
 'MAXREP':'parameter for source function calculation                       \n'+\
          'Maxmum number of iterations',\
 'SWSF':'parameter for source function calculation                         \n'+\
        'If >0, log when source function does not converge',\
 'EPSSF' :'parameter for source function calculation                       \n'+\
          'only consider region scattering/absorption < epssf and          \n'+\
          'iteration is stopped when relative change is < epssf',\
 'SWPR':'If >0, log radiation information',\
 'SWCONT':'(maybe not actually used) consider pesudo-continuum',\
 'SWPUN':'If >0, print out line profiles to op.dat',\
 'SWPLOT':'(maybe not actually used) plot line profile',\
 'SWLPR': 'If >0, convolution with instrumental profile',\
 'SWIP': 'If <0, skip convolution but log normalized continuum             \n'+\
         'If =0, skip convolution                                          \n'+\
         'If >0, do convolution ',\
 'KINDIP':'Instrumental profile type 1:rectangulra, 2:triangle, 3:gaussian',\
 'DIP':'Characteristic instrumental profile scale (sigma if sqrt(2)*gaussian)',\
 'IDBLD':'Some string for the log',\
 'SWWNWL':'If <=0, wavelength unit is Angstrom                             \n'+\
          'If >0,  wavelength unit is cm^-1',\
 'SWPRLD':'If >0, log details about line list',\
 'SWPRAC':'If >0, log line opacties',\
 'SWLAC': 'If <0, do not log line absorption coefficients calculation      \n'+\
          'If =0, partially log line absorption coefficients calculation   \n'+\
          'If >0, fully log line absorption coefficients calculation',\
 'COSTH'  :'If <0, get flux, >=0, get intensity',\
 'swipam' :'Not used in this version',\
 'swhcn':'Not used in this version',\
 'swhcch' :'Not used in this version'})
