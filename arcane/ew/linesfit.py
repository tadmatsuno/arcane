import numpy as np
import numpy as np
import matplotlib.pyplot as plt
for key in plt.rcParams.keys():
  if key.startswith('keymap'):
    [plt.rcParams[key].remove(ss) for ss in plt.rcParams[key]]
import matplotlib
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.Qt import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from scipy.special import voigt_profile
from scipy.stats import linregress
from scipy.optimize import minimize,LinearConstraint
from .pyqtlinesfit import Ui_Dialog
from pycont import myspec_utils as utils
import os 
import iofiles
from ..utils import utils


gfactfwhm = 2.0*np.sqrt(2.0*np.log(2.0))
lfactfwhm = 2.0
vfwhm = lambda fg,fl:0.5346*fl + np.sqrt(0.2166*fl*fl + fg*fg)

with open(os.path.join(__file__[:__file__.rfind('/')],'pnum_symbol.csv'),'r') as f:
  ptable = {}
  for line in f.readlines():
    pnum,symb = line.strip().split(',')
    ptable[int(pnum)] = symb


class LoadResults:
  def __init__(self,ewop,ewlog,wvl=[],flux=[]):
     self.ewop = ewop
     self.ewlog = ewlog 
     self.data = iofiles.readewop(self.ewop)
     with open(self.ewlog) as flog:
       self.details = flog.readlines()
     self.wvl = wvl
     self.flux = flux
  def list_species(self,idspecies):
     print('species = {0:d}'.format(idspecies))
     print('{0:>5s} {1:>10s} {2:>8s} {3:>8s} {4:>8s} {5:>8s} {6:>8s} {7:>8s} {8:>8s}'.format(\
           'No','wvl','gf','ep','depth','fwhm','ew','rew','cont'))
     for idx in self.data[self.data['nelem']==idspecies].index:
       print('{0:5d} {1:10.3f} {2:8.3f} {3:8.3f} {4:8.3f} {5:8.3f} {6:8.3f} {7:8.3f} {8:8.3f}'.format(\
           self.data.loc[idx,'No'],
           self.data.loc[idx,'WAVELENGTH'],
           self.data.loc[idx,'log(GF)'],
           self.data.loc[idx,'EXP(eV)'],
           self.data.loc[idx,'DEPTH'],
           self.data.loc[idx,'FWHM'],
           self.data.loc[idx,'EW(mA)'],
           self.data.loc[idx,'log(EW/WL)'],
           self.data.loc[idx,'Fact']))
     return self.data[self.data['nelem']==idspecies]['No'].values

  def read_one(self,lineno):
    d = self.data[np.abs(self.data['No'])==lineno]
    if len(d)==0:
      print('No match')
      return
    elif len(d)>1:
      print('Too many match')
      return
    idx = d.index[0]
    line = self.details[idx+1]
    a0 = float(line[0:10])
    a1 = float(line[10:20])
    wc = float(line[20:30])
    center = []
    dwvl = []
    ews = []
    fwhm = []
    gfwhm = []
    lfwhm = []
    depths = []
    gews = []
    nline = (len(line)-20)//80
    self.nline = nline
    for ii in range(nline):
      line1 = line[20+80*ii:20+80*ii+80]
      center.append(float(line1[0:10]))
      dwvl.append(float(line1[10:20])-float(line1[0:10]))
      ews.append(float(line1[20:30]))
      fwhm.append(float(line1[30:40]))
      gfwhm.append(float(line1[40:50]))
      lfwhm.append(float(line1[50:60]))
      depths.append(float(line1[60:70]))
      gews.append(float(line1[70:80]))
    flfwhm = np.array(lfwhm)/np.array(fwhm)      
    print('wc = {0:.3f}'.format(wc))
    print('continuum = {0:.3f} + {1:.3f}*(x-{2:.3f})'.format(a0,a1,wc))
    print('Lines')
    print('{0:>10s}{1:>10s}{2:>10s}{3:>10s}{4:>10s}{5:>10s}'.format(\
      'center','ew','fwhm','depth','lfwhm/fwhm','REW-REW_g'))
    for ii in range(nline):
      print(\
        '{0:10.3f}'.format(center[ii])+\
        '{0:10.3f}'.format(ews[ii])+\
        '{0:10.3f}'.format(fwhm[ii])+\
        '{0:10.3f}'.format(depths[ii])+\
        '{0:10.3f}'.format(flfwhm[ii])+\
        '{0:10.3f}'.format(np.log10(ews[ii]/gews[ii]))
        )
    self.cont = Continuum(wc,a0,a1)
    self.voigt = Voigt(center,fwhm=fwhm,depth=depths)
    self.voigt.dwvl = np.array(dwvl)
    self.voigt.flfwhm = flfwhm

  def plot(self,xr=None,figname=None):
    xbins = np.arange(self.voigt.center[0]-20,self.voigt.center[0]+20,0.01)
    fig,axs = plt.subplots(1,1,figsize=(5,5))
    axs = np.atleast_2d(axs).ravel()
    ax = axs[0]
    yy = self.cont.get_flux(xbins)*(1.0-self.voigt.get_flux(xbins))
    ax.plot(xbins,self.cont.get_flux(xbins),'C1--') ## Continuum
    ax.plot(xbins,self.cont.get_flux(xbins)*(1.0-self.voigt.get_flux(xbins)),'C1-') ## All lines
    for ii in range(self.nline): ## Each line
      ax.plot(xbins,\
        self.cont.get_flux(xbins)*(1.0-\
          utils.voigts_multi2(\
          self.voigt.center[ii]+self.voigt.dwvl[ii],\
          self.voigt.depth[ii],\
          self.voigt.fwhm[ii],\
          self.voigt.flfwhm[ii])(xbins)),'C1-',lw=0.5)
      ax.plot(xbins,\
        self.cont.get_flux(xbins)*(1.0-\
          utils.voigts_multi2(\
          self.voigt.center[ii]+self.voigt.dwvl[ii],\
          self.voigt.depth[ii],\
          self.voigt.fwhm[ii],\
          0.0)(xbins)),'C1--',lw=0.5)
    if self.wvl is not None:
      ax.plot(self.wvl,self.flux,'C7-',lw=0.5)
      ax.plot(self.wvl,self.flux,'C7o',ms=1.0)
    print(xr)
    if xr is None:
      xr = [np.min(xbins[yy/self.cont.a0<0.999])-1.,np.max(xbins[yy/self.cont.a0<0.999])+1.]
    yr = [(1.0-1.1*np.max(self.voigt.depth))*self.cont.a0,(1.0+0.1*np.max(self.voigt.depth))*self.cont.a0]
    ax.set_xlim(*xr)
    ax.set_ylim(*yr)
    fig.tight_layout()
    if figname is None:
      fig.show()
    else:
      fig.savefig(figname)
    plt.close(fig)

class Continuum:
  '''
  Class for continuum
  '''
  def __init__(self,wc,a0=1.0,a1=0.0,fix_a0=False,fix_a1=True):
    '''
    Parameters
    ----------
    wc : float
      central wavelength

    a0 : float
      intercept (default : 1)

    a1 : float
      slope (default : 0)

    fix_a0 : bool
      if True, a0 will be fixed during the fitting (dafault : False)

    fix_a1 : bool
      if True, a1 will be fixed during the fitting. (dafault : True)
    '''
    self.wc = wc
    self.a0 = a0
    self.a1 = a1
    self.fix_a0 = fix_a0
    self.fix_a1 = fix_a1

  def base_func(self,a0,a1):
    return lambda x:a0 + a1*x

  def get_flux(self,wavelength):
    '''
    Parameters
    ----------
    wavelength : array
      
    Returns
    -------
    continuum flux : array

    '''
    return self.base_func(self.a0,self.a1)(wavelength-self.wc)


  def fit(self,wavelength,flux,lowrej,highrej,niter,samples,grow=0.00):
    '''
    Fits local continuum 

    Parameters
    ----------
    wavelength : array

    flux : array

    lowrej : float
      lower rejection criteria for sigma-clipping

    highrej : float
      higher rejection criteria for sigma-clipping

    niter : int
      number of iterations for sigma-clipping

    samples : 2 x N list
      ranges that define local continuum

    grow : float
      grow radius in sigma-clipping

    '''

    self.flux_cont = self.get_flux(wavelength)
    self.use_flag = utils.get_region_mask(wavelength,samples)
    self.std = np.std((flux-self.flux_cont)[self.use_flag])
    if not ((self.fix_a0 and self.fix_a1) or (np.sum(self.use_flag)==0)):
      istep = 1
      use_flag = utils.get_region_mask(wavelength,samples)
      removemask =  np.array([False]*len(use_flag))
      while (istep <= niter):
        use_flag = use_flag & (~removemask)
        if np.sum(use_flag)==0:
          break          
        f_residual = lambda xx:\
          np.sum( (flux[use_flag]-\
          self.base_func(xx[0],xx[1])(wavelength[use_flag]-self.wc))**2.0)
        ## Linear Constraint depending on the fix options
        linconst = []
        linconst_low = []
        linconst_high = [] 
        if self.fix_a0:
          linconst.append([1.0,0.0])
          linconst_low.append(self.a0)
          linconst_high.append(self.a0)
        if self.fix_a1:
          linconst.append([0.0,1.0])
          linconst_low.append(self.a1)
          linconst_high.append(self.a1)
        if len(linconst)==0:
          constraints = None
        else:
          constraints = LinearConstraint(linconst,linconst_low,linconst_high)
        res = minimize(f_residual,x0=[self.a0,self.a1],constraints=constraints)
        self.a0,self.a1 = res.x
        flx_cont = self.get_flux(wavelength)
        self.std = np.std((flux-flx_cont)[use_flag],\
          ddof=np.sum(~self.fix_a0)+np.sum(~self.fix_a1))
        removemask = utils.sigmaclip(wavelength,flux,use_flag,flx_cont,grow,lowrej,highrej)     
        istep += 1
      self.flux_cont = self.get_flux(wavelength)
      self.use_flag = use_flag
      self.std = np.std((flux-self.flux_cont)[self.use_flag],\
        ddof=np.sum(~self.fix_a0)+np.sum(~self.fix_a1))
  
class Voigt:
  ''' 
  Class for line profile
  '''

  def __init__(self,wc,fwhm=0.1,depth=0.3,share_dwvl=False,\
    share_fwhm=False,share_gfwhm=False):
    '''
    a Voigt profile is controlled by four parameters
    - central wavelength
    - FWHM
    - depth
    - FWHM_L / FWHM = 2*gamma / FWHM

    Parameters
    ----------
    wc : float or 1d-array
      central wavelengths of lines 
    
    fwhm : float or 1d-array
      fwhm of the lines
    
    depth : float or 1d-array
      depths of the lines

    share_dwvl : bool
      if True, the shift in wavelengths will be shared for all the components during the fitting

    share_fwhm : bool
      if True, fwhm will be shared for all the components during the fitting
    
    share_gfwhm : bool
      if True, fwhm of the Gaussian components will be shared for all the components during the fitting


    '''

    wc = np.atleast_1d(wc)
    fwhm = np.atleast_1d(fwhm)
    depth = np.atleast_1d(depth)
    self.nline = len(wc)

    def input_value(xx):
      if (len(xx)==1):
        return xx.repeat(self.nline)
      elif (len(xx)==self.nline):
        return xx
      else:
        print(xx)
        print(len(xx),print(self.nline))
        raise ValueError('Voigt initialization error [length mismatch]')
    
    self.center = wc
    self.fwhm = input_value(fwhm)
    self.dwvl = np.zeros(self.nline)
    self.depth = input_value(depth)
    self.flfwhm = np.array([0.0]).repeat(self.nline)  # starts from a pure gaussian
    self.fix_fwhm = np.array([False]).repeat(self.nline) # If fwhm is fixed
    self.fix_dwvl = np.array([False]).repeat(self.nline) # If wavelength shift is set to 0
    self.fix_gfwhm = np.array([False]).repeat(self.nline) # If fwhm of the gaussian component is fixed
    self.fix_lfwhm = np.array([True]).repeat(self.nline) # If pure gaussian is assumed
    self.share_dwvl = share_dwvl
    self.share_fwhm = share_fwhm
    self.share_gfwhm = share_gfwhm

  def add_line(self,wc,fwhm=0.1,depth=0.3):
    '''
    This function add a new line at x = wc


    Parameters
    ----------
    wc : float
      the central wavelength of the new line
    
    fwhm : float
      fwhm of the new line

    depth : float
      depth of the new line

    '''
    self.center = np.append(self.center,wc)
    self.fwhm = np.append(self.fwhm,fwhm)
    self.dwvl = np.append(self.dwvl,0.0)
    self.depth = np.append(self.depth,depth)
    self.flfwhm = np.append(self.flfwhm,0.0)
    self.fix_dwvl = np.append(self.fix_dwvl,False)
    self.fix_fwhm = np.append(self.fix_fwhm,False)
    self.fix_gfwhm = np.append(self.fix_gfwhm,False)
    self.fix_lfwhm = np.append(self.fix_lfwhm,True)
    self.nline += 1
 
  def get_flux(self,wavelength):
    '''
    Get line profiles
    
    Parameters
    ----------
    wavelength : float or array
      wavelength of the points where flux is to be evaluated
    
    '''
    return utils.voigts_multi2(self.center+self.dwvl,self.depth,self.fwhm,self.flfwhm)(wavelength)

  def get_ew(self):
    '''
    Returns EW of each line
    '''
    gfwhm,lfwhm = utils.get_glFWHM(self.fwhm,self.flfwhm)
    gamma = lfwhm/2.0
    sigma = gfwhm/2.3548200450309493
    return [dep/voigt_profile(0,ss,ll)\
      for dep,ss,ll in zip(self.depth,sigma,gamma)]

  def fit(self,wavelength,flux_norm,lowrej,highrej,niter,samples,max_dwvl,grow=0.0):
    '''
    Fit the observed spectra with Voigt profiles

    Parameters
    ----------
    wavelength : array
      wavelengths of the observed spectrum 

    flux_norm : array
      normalized flux of the observed spectrum 
    
    lowrej : float
      lower rejection criteria for sigma-clipping

    highrej : float
      higher rejection criteria for sigma-clipping

    niter : int
      number of iterations for sigma-clipping

    samples : 2 x N list
      fitting regions

    max_dwvl : float
      allowed shift in central wavelengths of the lines

    grow : float
      grow radius in sigma-clipping


    '''

    flux = 1.0 - flux_norm
    istep = 1
    use_flag = utils.get_region_mask(wavelength,samples)
    removemask = np.array([False]*len(use_flag))
    while (istep <= niter):
      use_flag = use_flag & (~removemask)
      if np.sum(use_flag)==0:
        break
      f_residual = lambda xx:\
        np.sum((\
          utils.voigts_multi2(xx[0:self.nline]+self.center,xx[self.nline:2*self.nline],\
          xx[2*self.nline:3*self.nline],xx[3*self.nline:4*self.nline])(wavelength[use_flag]) \
          - flux[use_flag])**2.)
      linconst = []
      linconst_low = []
      linconst_high = []      
      # Constrains on shifts
      if self.share_dwvl: # same shift will be applied to all the lines
        # Constraints on the first line
        tmp = [0.]*self.nline*4
        tmp[0] = 1.0
        linconst.append(tmp)
        if self.fix_dwvl[0]: 
          linconst_low.append(self.dwvl[0])
          linconst_high.append(self.dwvl[0])
        else:
          linconst_low.append(-max_dwvl)
          linconst_high.append(max_dwvl)
        # Constraints on the other lines
        for ii in range(self.nline-1):
          tmp = [0.]*self.nline*4
          tmp[0] = 1.0
          tmp[ii+1] = -1.0 ## So that the shift in this line can be the same as the first line
          linconst.append(tmp)
          linconst_low.append(0.0)
          linconst_high.append(0.0)
      else:
        for ii in range(self.nline):
          tmp = [0.]*self.nline*4
          tmp[ii] = 1.0
          if self.fix_dwvl[ii]:
            linconst.append(tmp)
            linconst_low.append(self.dwvl[ii])
            linconst_high.append(self.dwvl[ii])
          else:
            linconst.append(tmp)
            linconst_low.append(-max_dwvl)
            linconst_high.append(max_dwvl)
      # Constrains on depths (between 0 and 1)
      for ii in range(self.nline):
        tmp = [0.]*self.nline*4
        tmp[ii+self.nline] = 1.
        linconst.append(tmp)
        linconst_low.append(0.0)
        linconst_high.append(1.0)
      # Constrains on fwhms
      if self.share_fwhm:
        # Constraints on the first line
        tmp = [0]*self.nline*4
        tmp[0+2*self.nline] = 1.0
        linconst.append(tmp)
        if self.fix_fwhm[0]:
          linconst_low.append(self.fwhm[0])
          linconst_high.append(self.fwhm[0])
        else:
          linconst_low.append(np.maximum(np.median(wavelength[1:]-wavelength[:-1]),0.01)) # has to be greater than 0.01 A or pixel size
          linconst_high.append(10.) # has to be smaller than some reasonable value (10.)
        for ii in range(self.nline-1):
          tmp = [0.]*self.nline*4
          tmp[0+2*self.nline] = 1.0
          tmp[ii+1+2*self.nline] = -1.0 # The fwhm of this line has to be equal to the first line
          linconst.append(tmp)
          linconst_low.append(0.0)
          linconst_high.append(0.0)
      else:
        for ii in range(self.nline):
          tmp = [0.]*self.nline*4
          tmp[ii+2*self.nline] = 1.0
          if self.fix_fwhm[ii]:
            linconst.append(tmp)
            linconst_low.append(self.fwhm[ii])
            linconst_high.append(self.fwhm[ii])
          else:
            linconst.append(tmp)
            linconst_low.append(np.maximum(np.median(wavelength[1:]-wavelength[:-1]),0.01))
            linconst_high.append(10.0)
      # Constraint on flfwhm = FWHM_L/FWHM. has to be between 0 and 1
      for ii in range(self.nline):
        tmp = [0.]*self.nline*4
        tmp[ii+3*self.nline] = 1.
        linconst.append(tmp)
        if self.fix_lfwhm[ii]:
          linconst_low.append(0.0)
          linconst_high.append(0.0)
        else:
          linconst_low.append(0.0)
          linconst_high.append(1.0)

      res = minimize(f_residual,\
        x0=np.hstack([self.dwvl,self.depth,self.fwhm,self.flfwhm]),\
        constraints=LinearConstraint(linconst,linconst_low,linconst_high))
      self.dwvl = res.x[0:self.nline]
      self.depth = res.x[self.nline:2*self.nline]
      self.fwhm = res.x[2*self.nline:3*self.nline]
      self.flfwhm = res.x[3*self.nline:4*self.nline]
      flx_fit = self.get_flux(wavelength)
      removemask = utils.sigmaclip(wavelength,flux,use_flag,flx_fit,grow,lowrej,highrej)
      istep +=1


    self.flux_fit = 1.0-self.get_flux(wavelength)
#    sigma,gamma = utils.get_glFWHM(self.fwhm,self.flfwhm)
    self.use_flag = use_flag    
    self.std = np.std(self.get_flux(wavelength[use_flag]) - flux[use_flag])

class OneLine:
  '''
  Class to conduct analysis for a single line
  '''
  def __init__(self,max_dwvl = 0.1, resolution = 60000.,n_average = 1,\
    lowrej_continuum=3.,highrej_continuum=3.,niter_continuum=5,\
    grow_continuum = 0.0, 
    lowrej_line = 3.,highrej_line=3.,niter_line=5, grow_line = 0.0):
    '''
    Parameters
    ----------
    max_dwvl : float
      allowed maximum wavelength shift in line fitting
    
    resolution : float
      resolution of the spectrum. Used to provid an initial guess for FWHM

    n_average : float
      binning

    lowrej_continuum, highrej_continuum : float, float
      lower/upper rejection threshold for sigma-clipping in local continuum fitting

    niter_continuum : int
      number of iterations for sigma-clipping in local continuum fitting
    
    grow_continuum : float
      grow radius for sigma-clipping in local continuum fitting

    lowrej_line, highrej_line : float, float
      lower/upper rejection threshold for sigma-clipping in line fitting

    niter_line : int
      number of iterations for sigma-clipping in line fitting
    
    grow_line : float
      grow radius for sigma-clipping in line fitting
    '''

    self.max_dwvl = max_dwvl
    self.resolution = resolution
    self.n_average = 1
    self.lowrej_continuum = lowrej_continuum
    self.highrej_continuum = highrej_continuum
    self.niter_continuum = niter_continuum
    self.grow_continuum = grow_continuum
    self.samples_continuum = []
    self.lowrej_line = lowrej_line
    self.highrej_line = highrej_line
    self.niter_line = niter_line
    self.grow_line = grow_line
    self.samples_line = []
    
  def setup(self,wavelength,flux,wc):
    '''
    This function is used to initialize analysis for a line
    
    Parameters
    ----------
    wavelength : 1d-array
    
    flux : 1d-array

    wc : the input line position
    '''
    self.wavelength_obs,self.flux_obs = wavelength,flux
    self.wc = wc
    fwhmA = wc/self.resolution
    self.samples_continuum = [[wc-fwhmA*5.,wc-fwhmA*3.],[wc+fwhmA*3.,wc+fwhmA*5.]]
    self.samples_line = [[wc-fwhmA*2.,wc+fwhmA*2.]]
    self.absorptions = []
    if hasattr(self,'continuum'):
      self.continuum = Continuum(wc,fix_a0=self.continuum.fix_a0,\
        fix_a1=self.continuum.fix_a1) ### Adopt the same parameter as previous line
    else:
      self.continuum = Continuum(wc)
    self.voigt = Voigt(wc)
    
  def fit(self,adjust_range=False,iter_cont_line=1):
    '''
    Parameters
    ----------
    adjust_range : bool
      If iter_cont_line = 0, this parameter is ignored. 
      If True, fitting range will be adjusted at each step
      (default : False)

    iter_cont_line : int
      The number of iterations between local continuum fitting and line fitting to 
      consider the cases where local continuum overlaps with line wings
      (default : 1)
    '''
    self.wavelength,self.flux = \
      utils.average_nbins(self.n_average,self.wavelength_obs,self.flux_obs)
    print('Continuum')
    self.continuum.fit(self.wavelength,self.flux,\
      self.lowrej_continuum,self.highrej_continuum,self.niter_continuum,
      self.samples_continuum,grow=self.grow_continuum)
    self.continuum.flux_norm = self.flux/self.continuum.flux_cont
    print('Line')
    self.voigt.fit(self.wavelength,self.continuum.flux_norm,\
      self.lowrej_line,self.highrej_line,self.niter_line,\
      self.samples_line,self.max_dwvl,
      grow=self.grow_line) 
    for ii in range(iter_cont_line):## Repeat n times in case the continuum overlaps with line wings
      if adjust_range:
        wc0 = self.voigt.center[0]+self.voigt.dwvl[0]
        fwhm0 = self.voigt.fwhm[0]
        self.samples_continuum = \
          [[wc0-fwhm0*5.,wc0-fwhm0*3.0],[wc0+fwhm0*5.,wc0+fwhm0*3.0]]
        self.samples_line = [[wc0-fwhm0*2.0,wc0+fwhm0*2.0]]
      print('Continuum')
      self.continuum.fit(self.wavelength,self.flux/self.voigt.flux_fit,\
        self.lowrej_continuum,self.highrej_continuum,self.niter_continuum,
        self.samples_continuum,
        grow = self.grow_continuum)
      self.continuum.flux_norm = self.flux/self.continuum.flux_cont
      print('Line')
      self.voigt.fit(self.wavelength,self.continuum.flux_norm,\
        self.lowrej_line,self.highrej_line,self.niter_line,\
        self.samples_line,self.max_dwvl,
        grow = self.grow_line) 

class PlotCanvas(FigureCanvas):
  '''
  Class for main window
  '''

  def __init__(self,parent,layout):
    self.fig,self.axes = plt.subplots(2,1,\
      gridspec_kw={'hspace':0.,'height_ratios':[3.,1.]},\
      sharex=True)
    FigureCanvas.__init__(self,self.fig)

    layout.addWidget(self)
    self.axes[1].set_xlabel('wavelength')
    self.axes[0].set_ylabel('flux')
    self.axes[1].set_ylabel('norm - fit')

    self.line_obs_axis0, = self.axes[0].plot([0],[0],'C7-',lw=0.5) # observed spectrum
    self.pt_use_axis0, = self.axes[0].plot([0],[0],'C0o',ms=2.0) # data points used 
    self.line_cont_axis0, = self.axes[0].plot([0],[0],'C1--',lw=1.0) # local continuum 
    self.line_allline_axis0, = self.axes[0].plot([0],[0],'C1-',lw=1.0) # the result of fit
    self.line_eachline_axis0, = self.axes[0].plot([0],[0],'C1-',lw=0.5) # every line in the fit
    self.line_geachline_axis0, = self.axes[0].plot([0],[0],'C1--',lw=0.5) # gaussian component of the every line
    self.region_cont_2sigma_axis0 = \
      self.axes[0].fill_between([0.0],[0.0],[0.0],facecolor='C7',alpha=0.3) # continuum uncertainty
    self.region_cont_2sigma_axis1 = \
      self.axes[1].fill_between([0.0],[0.0],[0.0],facecolor='C7',alpha=0.3) # continuum uncertainty
    self.region_cont_2sigma_label_axis0 = \
      self.axes[0].text(0.,0.,'cont (2sig)',\
        verticalalignment='top',horizontalalignment='left',
        color='C7')  # continuum uncertainty text
    self.line_obs_axis1, = self.axes[1].plot([0],[0],'C7-',lw=0.5)  # residual plot
    self.pt_use_axis1, = self.axes[1].plot([0],[0],'C0o',ms=2.0) # data points used 
    self.zero_axis1, = self.axes[1].plot([0],[0],'C1--',ms=1.0) # zero in the residual plot

    self.cursorx = self.axes[0].axvline(x=0,\
      linestyle='-',color='C7',lw=0.5) # mouse cursor x
    self.cursory = self.axes[0].axhline(y=0,\
      linestyle='-',color='C7',lw=0.5) # mouse curor y


    self.fig.canvas.mpl_connect('motion_notify_event',self.mouse_move)
    self.txt = self.axes[0].text(0.0,0.0,'',\
      transform=self.fig.transFigure,
      horizontalalignment='left',verticalalignment='bottom') # Mouse cursor position text
    self.mode_txt = self.axes[0].text(1.0,1.0,'Normal',\
      transform=self.fig.transFigure,
      fontsize='x-large',color='k',
      horizontalalignment='right',verticalalignment='top') # Display current mode

    self.setFocusPolicy(Qt.ClickFocus)
    self.setFocus()
    self.toolbar = NavigationToolbar(self ,parent)
    layout.addWidget(self.toolbar)

    self.fig.tight_layout()
    self.updateGeometry()

  def mouse_move(self,event): # capture movement of mouse cursor
    x,y = event.xdata,event.ydata
    self.cursorx.set_xdata(x)
    if event.inaxes == self.axes[0]:
      self.cursory.set_ydata(y)
    else:
      self.cursory.set_ydata(None)
    if not ((x is None)|(y is None)):
      self.txt.set_text('x={0:10.3f}    y={1:10.5f}'.format(x,y))
    self.draw()



class MainWindow(QWidget,Ui_Dialog):
  '''
  Class for main window
  '''
  def __init__(self,parent=None,resolution = 60000.):
    super(MainWindow,self).__init__(parent)
    self.ui = Ui_Dialog()
    self.ui.setupUi(self)

    self.oneline = OneLine(resolution=resolution)
    self.canvas  =  PlotCanvas(self.ui.left_grid,self.ui.main_figure)
    
    self.ui.inwvls = [self.ui.inwvl1,self.ui.inwvl2,self.ui.inwvl3,self.ui.inwvl4,self.ui.inwvl5] # input wavelengths
    self.ui.fix_dwvls = [self.ui.fix_dwvl1,self.ui.fix_dwvl2,self.ui.fix_dwvl3,self.ui.fix_dwvl4,self.ui.fix_dwvl5] # check box if wavelength shift is fixed to zero
    self.ui.edit_dwvls = [self.ui.edit_dwvl1,self.ui.edit_dwvl2,self.ui.edit_dwvl3,self.ui.edit_dwvl4,self.ui.edit_dwvl5] # wavelength shift 
    self.ui.fix_fwhms = [self.ui.fix_fwhm1,self.ui.fix_fwhm2,self.ui.fix_fwhm3,self.ui.fix_fwhm4,self.ui.fix_fwhm5] # check box if fwhm is fixed
    self.ui.edit_fwhms = [self.ui.edit_fwhm1,self.ui.edit_fwhm2,self.ui.edit_fwhm3,self.ui.edit_fwhm4,self.ui.edit_fwhm5] # fwhms
    self.ui.fix_lfwhms = [self.ui.fix_lfwhm1,self.ui.fix_lfwhm2,self.ui.fix_lfwhm3,self.ui.fix_lfwhm4,self.ui.fix_lfwhm5] # check box if FWHM_L/FWHM is fixed to zero
    
    self.ui.button_fit.installEventFilter(self) # button for refit 
    self.ui.button_draw.installEventFilter(self) # button for draw
    self.ui.edit_naverage.installEventFilter(self) # textbox for binning
    self.ui.edit_lowrej_cont.installEventFilter(self) # textbox for lowrej threshold in continuum fitting
    self.ui.edit_highrej_cont.installEventFilter(self) # textbox for highrej threshold in continuum fitting
    self.ui.edit_niter_cont.installEventFilter(self) # text box for number of sigma-clipping in continuum fitting
    self.ui.edit_samples_cont.installEventFilter(self) # text box for sampling range in continuum fitting
    self.ui.edit_lowrej_line.installEventFilter(self) # textbox for lowrej threshold in line fitting
    self.ui.edit_highrej_line.installEventFilter(self) # textbox for highrej threshold in line fitting
    self.ui.edit_niter_line.installEventFilter(self) # textbox for number of sigma-clipping in line fitting
    self.ui.edit_samples_line.installEventFilter(self) # text box for sampling range in line fitting
    self.ui.fix_a0.installEventFilter(self) # check box if continuum level is fixed
    self.ui.fix_a1.installEventFilter(self) # check box if continuum slope is fixed
    self.ui.edit_a0.installEventFilter(self) # text box for continuum level
    self.ui.edit_a1.installEventFilter(self) # text box for continuum slope
    self.ui.share_dwvl.installEventFilter(self) # check box if wavelength shift is shared for all the lines
    self.ui.share_fwhm.installEventFilter(self) # check box if fwhm is shared for all the lines
    [ui1.installEventFilter(self) for ui1 in self.ui.inwvls]
    [ui1.installEventFilter(self) for ui1 in self.ui.fix_dwvls]
    [ui1.installEventFilter(self) for ui1 in self.ui.edit_dwvls]
    [ui1.installEventFilter(self) for ui1 in self.ui.fix_fwhms]
    [ui1.installEventFilter(self) for ui1 in self.ui.edit_fwhms]
    [ui1.installEventFilter(self) for ui1 in self.ui.fix_lfwhms]

    self.reflect_ui(fit_result=False)

    self.mpl_status = None
    self.canvas.mpl_connect('key_press_event',self.on_press)


  def reflect_ui(self,fit_result=True):
    '''
    This is to reflect internally-stored value to textboxes and checkboxes

    Parameters
    ----------
    fit_result : bool
      If fitting result should be reflected to ui
    '''
    self.ui.edit_naverage.setText('{0:d}'.format(self.oneline.n_average))
    self.ui.edit_lowrej_cont.setText('{0:.2f}'.format(self.oneline.lowrej_continuum))
    self.ui.edit_highrej_cont.setText('{0:.2f}'.format(self.oneline.highrej_continuum))
    self.ui.edit_niter_cont.setText('{0:d}'.format(self.oneline.niter_continuum))
    self.ui.edit_samples_cont.setPlainText(utils.textsamples(self.oneline.samples_continuum))
    self.ui.edit_lowrej_line.setText('{0:.2f}'.format(self.oneline.lowrej_line))
    self.ui.edit_highrej_line.setText('{0:.2f}'.format(self.oneline.highrej_line))
    self.ui.edit_niter_line.setText('{0:d}'.format(self.oneline.niter_line))
    self.ui.edit_samples_line.setPlainText(utils.textsamples(self.oneline.samples_line))

    if fit_result:
      self.ui.edit_a0.setText('{0:.3f}'.format(self.oneline.continuum.a0))
      self.ui.edit_a1.setText('{0:.3f}'.format(self.oneline.continuum.a1))
      gfwhm,lfwhm = utils.get_glFWHM(self.oneline.voigt.fwhm,self.oneline.voigt.flfwhm)
      ews = self.oneline.voigt.get_ew()
      for ii in range(5):
        if ii < self.oneline.voigt.nline:
          getattr(self.ui,'inwvl{0:1d}'.format(ii+1)).setText(\
            '{0:.3f}'.format(self.oneline.voigt.center[ii]))
          getattr(self.ui,'edit_dwvl{0:1d}'.format(ii+1)).setText(\
            '{0:.3f}'.format(self.oneline.voigt.dwvl[ii]))
          getattr(self.ui,'edit_fwhm{0:1d}'.format(ii+1)).setText(\
            '{0:.3f}'.format(self.oneline.voigt.fwhm[ii]))
          getattr(self.ui,'edit_gfwhm{0:1d}'.format(ii+1)).setText(\
            '{0:.3f}'.format(gfwhm[ii]))
          getattr(self.ui,'edit_lfwhm{0:1d}'.format(ii+1)).setText(\
            '{0:.3f}'.format(lfwhm[ii]))
          getattr(self.ui,'edit_EW{0:1d}'.format(ii+1)).setText(\
            '{0:.3f}'.format(ews[ii]*1.0e3))
          getattr(self.ui,'fix_fwhm{0:1d}'.format(ii+1)).setChecked(\
            self.oneline.voigt.fix_fwhm[ii])
          getattr(self.ui,'fix_lfwhm{0:1d}'.format(ii+1)).setChecked(\
            self.oneline.voigt.fix_lfwhm[ii])
          getattr(self.ui,'fix_dwvl{0:1d}'.format(ii+1)).setChecked(\
            self.oneline.voigt.fix_dwvl[ii])
        else:
          getattr(self.ui,'inwvl{0:1d}'.format(ii+1)).setText('')
          getattr(self.ui,'edit_dwvl{0:1d}'.format(ii+1)).setText('')
          getattr(self.ui,'edit_fwhm{0:1d}'.format(ii+1)).setText('')
          getattr(self.ui,'edit_gfwhm{0:1d}'.format(ii+1)).setText('')
          getattr(self.ui,'edit_lfwhm{0:1d}'.format(ii+1)).setText('')
          getattr(self.ui,'edit_EW{0:1d}'.format(ii+1)).setText('')
          getattr(self.ui,'fix_fwhm{0:1d}'.format(ii+1)).setChecked(False)
          getattr(self.ui,'fix_lfwhm{0:1d}'.format(ii+1)).setChecked(False)
          getattr(self.ui,'fix_dwvl{0:1d}'.format(ii+1)).setChecked(False)


      self.ui.fix_a0.setChecked(self.oneline.continuum.fix_a0)
      self.ui.fix_a1.setChecked(self.oneline.continuum.fix_a1)
      self.ui.share_dwvl.setChecked(self.oneline.voigt.share_dwvl)
      self.ui.share_fwhm.setChecked(self.oneline.voigt.share_fwhm)

  def draw_fig(self):
    '''
    Draw fitting results

    '''
    if not hasattr(self.oneline,'wavelength'):
      raise AttributeError('setup Oneline first!')
    ## observed
    self.canvas.line_obs_axis0.set_xdata(self.oneline.wavelength)
    self.canvas.line_obs_axis0.set_ydata(self.oneline.flux)
    self.canvas.pt_use_axis0.set_xdata(\
      self.oneline.wavelength[self.oneline.voigt.use_flag])
    self.canvas.pt_use_axis0.set_ydata(\
      self.oneline.flux[self.oneline.voigt.use_flag])
    ## continuum
    self.canvas.line_cont_axis0.set_xdata(self.oneline.wavelength)
    self.canvas.line_cont_axis0.set_ydata(self.oneline.continuum.flux_cont)
    ## fitting results
    self.canvas.line_allline_axis0.set_xdata(self.oneline.wavelength)
    self.canvas.line_allline_axis0.set_ydata(\
      self.oneline.continuum.flux_cont*self.oneline.voigt.flux_fit)
    ## show each component
    self.canvas.line_eachline_axis0.set_xdata(\
      np.hstack([self.oneline.wavelength]*self.oneline.voigt.nline))
    self.canvas.line_eachline_axis0.set_ydata(\
      np.hstack([\
      self.oneline.continuum.flux_cont*(1.0-\
      utils.voigts_multi2(\
      self.oneline.voigt.center[ii]+self.oneline.voigt.dwvl[ii],
      self.oneline.voigt.depth[ii],\
      self.oneline.voigt.fwhm[ii],\
      self.oneline.voigt.flfwhm[ii])(self.oneline.wavelength))\
      for ii in range(self.oneline.voigt.nline)]))
    ## show each component neglecting their lorenzian components
    self.canvas.line_geachline_axis0.set_xdata(\
      np.hstack([self.oneline.wavelength]*self.oneline.voigt.nline))
    self.canvas.line_geachline_axis0.set_ydata(\
      np.hstack([\
      self.oneline.continuum.flux_cont*(1.0-\
      utils.voigts_multi2(\
      self.oneline.voigt.center[ii]+self.oneline.voigt.dwvl[ii],
      self.oneline.voigt.depth[ii],\
      self.oneline.voigt.fwhm[ii],\
      0.0)(self.oneline.wavelength))\
      for ii in range(self.oneline.voigt.nline)]))
    ## second axis, residual plot
    self.canvas.line_obs_axis1.set_xdata(self.oneline.wavelength)
    self.canvas.pt_use_axis1.set_xdata(\
      self.oneline.wavelength[self.oneline.voigt.use_flag])
    yresid = self.oneline.flux/self.oneline.continuum.flux_cont - \
      self.oneline.voigt.flux_fit
    self.canvas.line_obs_axis1.set_ydata(yresid)
    self.canvas.pt_use_axis1.set_ydata(\
      yresid[self.oneline.voigt.use_flag])
    self.canvas.zero_axis1.set_xdata(self.oneline.wavelength)
    self.canvas.zero_axis1.set_ydata(0.0)

    ## Show continuum
    self.canvas.region_cont_2sigma_axis0.remove()
    self.canvas.region_cont_2sigma_axis1.remove()
    self.canvas.region_cont_2sigma_axis0 = \
      self.canvas.axes[0].fill_between(\
        self.oneline.wavelength,\
        self.oneline.continuum.flux_cont-2.0*self.oneline.continuum.std,\
        self.oneline.continuum.flux_cont+2.0*self.oneline.continuum.std,\
        facecolor='C7',alpha=0.3)
    self.canvas.region_cont_2sigma_axis1 = \
      self.canvas.axes[1].fill_between(\
        self.oneline.wavelength,\
        -2.0*self.oneline.continuum.std/self.oneline.continuum.flux_cont,\
        2.0*self.oneline.continuum.std/self.oneline.continuum.flux_cont,\
        facecolor='C7',alpha=0.3)
    ## define xy limits
    wc0 = self.oneline.voigt.center[0]+self.oneline.voigt.dwvl[0]
    fwhm0 = np.maximum(self.oneline.voigt.fwhm[0],0.01)
    if len(self.oneline.samples_continuum)>0:
      self.canvas.axes[0].set_xlim(\
         np.minimum(wc0-5.*fwhm0,np.min(np.array(self.oneline.samples_continuum)))-fwhm0,\
         np.maximum(wc0+5.*fwhm0,np.max(np.array(self.oneline.samples_continuum)))+fwhm0)
    else:
      self.canvas.axes[0].set_xlim(wc0-5.*fwhm0,wc0+5.*fwhm0)
    mask1 = ((wc0-2.*fwhm0)<self.oneline.wavelength)&(self.oneline.wavelength<(wc0+2.*fwhm0))
    if np.sum(mask1)==0:
      ymin0,ymax0 = 0.,1.5
    else:
      ymin0,ymax0 = np.nanmin(self.oneline.flux[mask1]),np.maximum(1.0,np.nanmax(self.oneline.flux[mask1]))
    self.canvas.axes[0].set_ylim(\
      ymin0-(ymax0-ymin0)*0.05,ymax0+(ymax0-ymin0)*0.05)
    ax1ylim = 5.0*self.oneline.continuum.std/np.median(self.oneline.continuum.flux_cont)
    if ~np.isfinite(ax1ylim):
      self.canvas.axes[1].set_ylim(-0.5,0.5)
    else:
      self.canvas.axes[1].set_ylim(-ax1ylim,ax1ylim)
    ## move a label
    self.canvas.region_cont_2sigma_label_axis0.set_x(\
      wc0-6.*fwhm0)
    self.canvas.region_cont_2sigma_label_axis0.set_y(\
      (self.oneline.continuum.flux_cont-3.0*self.oneline.continuum.std)[0])


    ## Show regions
    _ = self.show_selected_region(self.oneline.samples_continuum,continuum=True)
    _ = self.show_selected_region(self.oneline.samples_line,continuum=False)
    self.canvas.draw()

  def done(self):
    '''
    This function shows a textbox that indicates that all the lines have been measured
    '''
    self.canvas.axes[0].text(0.5,0.5,'Done! \n Close the window',
      bbox=dict(facecolor='white', alpha=0.5),
      transform=self.canvas.fig.transFigure,
      fontsize='xx-large',color='k',
      horizontalalignment='center',verticalalignment='center')
  
  def input_data(self,wavelength,flux,wavelength_lines,\
    fout=None,flog=None,elemid=None,gf=None,ep=None):
    self.wavelength = wavelength 
    self.flux = flux
    self.wavelength_lines = wavelength_lines
    self.nlines = len(wavelength_lines)
    self.iline = 0
    self.fout = fout
    self.flog = flog
    if self.fout is not None:
      self.fout.write('\n')
      self.fout.write(' No.    ELEMENT WAVELENGTH log(GF) EXP(eV)   WL(cent) DWL(o-c)  DEPTH   DDEP    FWHM     EW(mA) log(EW/WL)EWinteg(mA)     Fact      xmin      xmax Ref.\n')
    if self.flog is not None:
      self.flog.write('a0 a1 wc wobs ew fwhm gfwhm lfwhm depth gew\n')
    if elemid is not None:
      self.elemid = elemid
    if gf is not None:
      self.gf = gf
    if ep is not None:
      self.ep = ep


  def show_selected_region(self,samples,continuum=True):
    if continuum:
      self.canvas.samples_cont = samples
      color = 'C0'
      vspan_label = 'vspan_samples_cont'
    else:
      self.canvas.samples_line = samples
      color = 'yellow'
      vspan_label = 'vspan_samples_line'
    if hasattr(self.canvas,vspan_label):
        _ = [vss.remove() for vss in getattr(self.canvas,vspan_label)]
    if len(samples)==0:
      setattr(self.canvas,vspan_label,[])
      self.canvas.draw()
      return samples
    else:
      ss = np.sort(samples)
      idx_ss = np.argsort(ss[:,0])
      vspan_samples = []
      x1,x2 = ss[idx_ss[0]]
      for idx in idx_ss[1:]:
        x1n,x2n = ss[idx]
        if x2 < x1n:
          vspan_samples.append(\
            self.canvas.axes[0].axvspan(x1,x2,facecolor=color,alpha=0.3,))
          x1 = x1n
          x2 = x2n
        else:
          x2 = x2n
          continue
      vspan_samples.append(\
        self.canvas.axes[0].axvspan(x1,x2,facecolor=color,alpha=0.3,))
      setattr(self.canvas,vspan_label,vspan_samples)
      self.canvas.draw()
      return ss[idx_ss].tolist()

  def fit_line(self,wc):
    mask = ((wc-20.)<self.wavelength)&(self.wavelength<(wc+20.))
    mask2 = ((wc-1.)<self.wavelength)&(self.wavelength<(wc+1.))
    if (np.sum(mask2)==0)|(all(self.flux[mask2]==0.0)):
      self.moveon_done(use=False,skip=True)
      print('{0:.3f} line is skipped'.format(wc))
    else:
      self.oneline.setup(self.wavelength[mask],self.flux[mask],wc)
      self.oneline.fit(iter_cont_line=0)
      self.reset_sample_regions()
      self.oneline.fit()
      self.reflect_ui()
      self.draw_fig()

  def moveon_done(self,use=True,skip=False):
    if hasattr(self,'fout') & (not skip):
      lineid = self.iline + 1
      if not use:
        lineid = -lineid
      if hasattr(self,'elemid'):
        elid = self.elemid[self.iline]
        elname = ptable[elid%100] + ' ' + 'I'*(elid//100)
      else:
        elid = 0
        elname = ''
      if hasattr(self,'gf'):
        gf = self.gf[self.iline]
      else:
        gf = 0.0
      if hasattr(self,'ep'):
        ep = self.ep[self.iline]
      else:
        ep = 0.0
      gfwhm,lfwhm = utils.get_glFWHM(self.oneline.voigt.fwhm,self.oneline.voigt.flfwhm)
      ews = self.oneline.voigt.get_ew()
      self.fout.write(\
        ' {0:4d}'.format(lineid)+\
        ' {0:3d}'.format(elid)+\
        '  {0:5s}'.format(elname)+\
        '{0:10.3f}'.format(self.oneline.voigt.center[0])+\
        '{0:10.3f}'.format(gf)+\
        '{0:8.3f}'.format(ep)+\
        '{0:10.3f}'.format(self.oneline.voigt.center[0]+self.oneline.voigt.dwvl[0])+\
        '{0:8.3f}'.format(self.oneline.voigt.dwvl[0])+\
        '{0:8.3f}'.format(self.oneline.voigt.depth[0])+\
        '{0:8.3f}'.format(0.0)+\
        '{0:8.3f}'.format(self.oneline.voigt.fwhm[0])+\
        '{0:10.2f}'.format(ews[0]*1.0e3)+\
        '{0:10.3f}'.format(np.log10(ews[0]/(self.oneline.voigt.center[0]+self.oneline.voigt.dwvl[0])))+\
        '{0:10.2f}'.format(ews[0]*1.0e3)+\
        '{0:10.5f}'.format(self.oneline.continuum.a0)+\
        '{0:10.3f}'.format(np.min(np.array(self.oneline.samples_line)))+\
        '{0:10.3f}\n'.format(np.max(np.array(self.oneline.samples_line)))\
          )
      if hasattr(self,'flog'):
        flogout = '{0:10.5f}{1:10.5f}'.format(self.oneline.continuum.a0,self.oneline.continuum.a1)
        for ii in range(self.oneline.voigt.nline):
          flogout += '{0:10.3f}{1:10.3f}{2:10.3f}{3:10.3f}{4:10.3f}{5:10.3f}{6:10.3f}{7:10.3f}'.format(\
            self.oneline.voigt.center[ii],
            self.oneline.voigt.center[ii]+self.oneline.voigt.dwvl[ii],\
            ews[ii]*1.0e3,
            self.oneline.voigt.fwhm[ii],gfwhm[ii],lfwhm[ii],
            self.oneline.voigt.depth[ii],
            1.0e3*self.oneline.voigt.depth[ii]*np.sqrt(2.0*np.pi)*self.oneline.voigt.fwhm[ii]/2.3548200450309493)
        self.flog.write(flogout+'\n')
    self.iline += 1
    if self.iline == self.nlines:
      self.done()
    else:
      self.fit_line(self.wavelength_lines[self.iline])


  def eventFilter(self,source,event):
    if event.type() == QEvent.FocusIn:
      if source in [self.ui.edit_samples_line,self.ui.edit_samples_cont]:
        self.temp_text = source.toPlainText()
      else:
        self.temp_text = source.text()
    elif event.type() == QEvent.FocusOut:
      if source in [self.ui.edit_samples_line,self.ui.edit_samples_cont]:
        new_text = source.toPlainText()
      else:
        new_text = source.text()
      print(new_text)
      if source is self.ui.edit_naverage:
        try:
          self.oneline.n_average = int(new_text)
        except:
          self.ui.edit_naverage.setText(self.temp_text)
      elif source == self.ui.edit_lowrej_cont:
        try:
          self.oneline.lowrej_continuum = float(new_text)
        except:
          self.ui.edit_lowrej_cont.setText(self.temp_text)
      elif source is self.ui.edit_highrej_cont:
        try:
          self.oneline.highrej_continuum = float(new_text)
        except:
          self.ui.edit_highrej_cont.setText(self.temp_text)
      elif source is self.ui.edit_niter_cont:
        try:
          self.oneline.niter_continuum = int(new_text)
        except:
          self.ui.edit_niter_cont.setText(self.temp_text)
      elif source is self.ui.edit_lowrej_line:
        try:
          self.oneline.lowrej_line = float(new_text)
        except:
          self.ui.edit_lowrej_line.setText(self.temp_text)
      elif source is self.ui.edit_highrej_line:
        try:
          self.oneline.highrej_line = float(new_text)
        except:
          self.ui.edit_highrej_line.setText(self.temp_text)
      elif source is self.ui.edit_niter_line:
        try:
          self.oneline.niter_line = int(new_text)
        except:
          self.ui.edit_niter_line.setText(self.temp_text)
      elif source is self.ui.edit_a0:
        try:
          self.oneline.continuum.a0 = float(new_text)
        except:
          self.ui.edit_a0.setText(self.temp_text)
      elif source is self.ui.edit_a1:
        try:
          self.oneline.continuum.a1 = float(new_text)
        except:
          self.ui.edit_a1.setText(self.temp_text)
      elif source in self.ui.inwvls:
        idx = self.ui.inwvls.index(source)
        if  idx < self.oneline.voigt.nline:
          if new_text == '':
            self.oneline.voigt.center = np.delete(self.oneline.voigt.center,idx)
            self.oneline.voigt.fwhm = np.delete(self.oneline.voigt.fwhm,idx)
            self.oneline.voigt.dwvl = np.delete(self.oneline.voigt.dwvl,idx)
            self.oneline.voigt.depth = np.delete(self.oneline.voigt.depth,idx)
            self.oneline.voigt.flfwhm = np.delete(self.oneline.voigt.flfwhm,idx)
            self.oneline.voigt.fix_fwhm = np.delete(self.oneline.voigt.fix_fwhm,idx)
            self.oneline.voigt.fix_dwvl = np.delete(self.oneline.voigt.fix_dwvl,idx)
            self.oneline.voigt.fix_gfwhm = np.delete(self.oneline.voigt.fix_gfwhm,idx)
            self.oneline.voigt.fix_lfwhm = np.delete(self.oneline.voigt.fix_lfwhm,idx)
            self.oneline.voigt.nline = self.oneline.voigt.nline -1
            self.reflect_ui()
          else:
            self.oneline.voigt.center[idx] = float(new_text)
        else:
          self.ui.inwvls[idx].setText(self.temp_text)
      elif source in self.ui.edit_dwvls:
        idx = self.ui.edit_dwvls.index(source)
        if  idx < self.oneline.voigt.nline:
          self.oneline.voigt.dwvl[idx] = float(new_text)
        else:
          self.ui.edit_dwvls[idx].setText(self.temp_text)
      elif source in self.ui.edit_fwhms:
        idx = self.ui.edit_fwhms.index(source)
        if  idx < self.oneline.voigt.nline:
          self.oneline.voigt.fwhm[idx] = float(new_text)
        else:
          self.ui.edit_fwhms[idx].setText(self.temp_text)     
      elif source is self.ui.edit_samples_cont:
        try:
          ss = utils.textsamples(new_text,reverse=True)
          ss_sorted = self.show_selected_region(ss,continuum=True)
          self.oneline.samples_continuum = ss_sorted
          self.ui.edit_samples_cont.setPlainText(\
            utils.textsamples(ss_sorted))
        except:
          print('Input error')
          self.ui.edit_samples_cont.setPlainText(self.temp_text)
      elif source is self.ui.edit_samples_line:
        try:
          ss = utils.textsamples(new_text,reverse=True)
          ss_sorted = self.show_selected_region(ss,continuum=False)
          self.oneline.samples_line = ss_sorted
          self.ui.edit_samples_line.setPlainText(\
            utils.textsamples(ss_sorted))
        except:
          print('Input error')
          self.ui.edit_samples_line.setPlainText(self.temp_text)
      else:
        try:
          source.setText(self.temp_text)
        except:
          source.setPlainText(self.temp_text)
    elif event.type() == QEvent.MouseButtonPress:
      if source is self.ui.button_fit:
        print('Refit')
        self.oneline.fit()
        self.reflect_ui()
        self.draw_fig()
      elif source is self.ui.button_draw:
        self.draw_fig()
    elif event.type() == QEvent.Paint:
      if source is self.ui.share_dwvl:
        self.oneline.voigt.share_dwvl = \
          self.ui.share_dwvl.isChecked()
      elif source is self.ui.share_fwhm:
        self.oneline.voigt.share_fwhm = \
          self.ui.share_fwhm.isChecked()
      elif source is self.ui.fix_a0:
        self.oneline.continuum.fix_a0 = \
          self.ui.fix_a0.isChecked()
      elif source is self.ui.fix_a1:
        self.oneline.continuum.fix_a1 = \
          self.ui.fix_a1.isChecked()
      elif source in self.ui.fix_dwvls:
        idx = self.ui.fix_dwvls.index(source)
        if  idx < self.oneline.voigt.nline:
          self.oneline.voigt.fix_dwvl[idx] = \
             self.ui.fix_dwvls[idx].isChecked()
        else:
          self.ui.fix_dwvls[idx].setChecked(False)
      elif source in self.ui.fix_fwhms:
        idx = self.ui.fix_fwhms.index(source)
        if  idx < self.oneline.voigt.nline:
          self.oneline.voigt.fix_fwhm[idx] = \
             self.ui.fix_fwhms[idx].isChecked()
        else:
          self.ui.fix_fwhms[idx].setChecked(False)
      elif source in self.ui.fix_lfwhms:
        idx = self.ui.fix_lfwhms.index(source)
        if  idx < self.oneline.voigt.nline:
          self.oneline.voigt.fix_lfwhm[idx] = \
             self.ui.fix_lfwhms[idx].isChecked()
        else:
          self.ui.fix_lfwhms[idx].setChecked(False)


  #  print(source,event,self.ui.share_dwvl.isChecked())
    return QWidget.eventFilter(self, source, event)
  
  def _clear_state(self):
    if hasattr(self,'tmp_data'):
      delattr(self,'tmp_data')
    self.canvas.mode_txt.set_text('Normal')
    self.mpl_status = None
    self.canvas.draw()

  def reset_sample_regions(self):
    fwhm0 = self.oneline.voigt.fwhm[0]
    wc0 = self.oneline.voigt.center[0]+self.oneline.voigt.dwvl[0]
    ## continuum
    self.oneline.samples_continuum = \
      [[wc0-fwhm0*5.,wc0-fwhm0*3.],[wc0+fwhm0*3.,wc0+fwhm0*5.]]
    self.oneline.samples_line = [[wc0-fwhm0*2.,wc0+fwhm0*2.]]
    _ = self.show_selected_region(\
      self.oneline.samples_continuum,continuum=True)
    self.ui.edit_samples_cont.setPlainText(\
        utils.textsamples(self.oneline.samples_continuum))
    _ = self.show_selected_region(\
      self.oneline.samples_line,continuum=False)
    self.ui.edit_samples_line.setPlainText(\
        utils.textsamples(self.oneline.samples_line))

  def on_press(self,event):
    print(event.key)
    if self.mpl_status is None:
      if (event.key=='s') and (event.xdata is not None):
        self.mpl_status = 's'
        self.tmp_data = {'x':event.xdata,
          'lvx1':self.canvas.axes[0].axvline(event.xdata,color='r',lw=2.)}
        self.canvas.mode_txt.set_text('Sample: waitning for [c] or [l]')
        self.canvas.draw()
      elif event.key == 't':
        self.mpl_status = 't'
        self.canvas.mode_txt.set_text('Clean: waitning for [c] [l] [b,r]')
        self.canvas.draw()
      elif event.key == 'y':
        print('Use this line')
        self.mpl_status = 'y'
        self.canvas.mode_txt.set_text('Use this line? [y] or [n]')
        self.oneline.fit()
        self.reflect_ui()
        self.draw_fig()
      elif event.key == 'u':
        pass
        #self.mpl_status = 'u'
        #self.canvas.mode_txt.set_text('Un-sample')
        #self.canvas.draw()
      elif event.key == 'r':
        print('Reset sample region \n')
        self.reset_sample_regions()
      elif event.key == 'n':
        print('Not use this line')
        self.moveon_done(use=False)
      elif event.key == 'f':
        print('Refit')
        self.oneline.fit()
        self.reflect_ui()
        self.draw_fig()
      elif event.key == 'a':
        print('Add line')
        self.oneline.voigt.add_line(event.xdata,\
          fwhm=self.oneline.voigt.fwhm[0],depth=self.oneline.voigt.depth[0])
        self.reflect_ui()
      elif event.key == 'd':
        print('Default FWHM and dwvl')
        self.oneline.voigt.center = self.oneline.voigt.center[0:1]
        self.oneline.voigt.fwhm = self.oneline.voigt.center/self.oneline.resolution
        self.oneline.voigt.dwvl = np.array([0.0])
        self.oneline.voigt.depth = np.array([0.1])
        self.oneline.voigt.flfwhm = np.array([0.0])
        self.oneline.voigt.fix_fwhm = np.array([False])
        self.oneline.voigt.fix_dwvl = np.array([True])
        self.oneline.voigt.fix_gfwhm = np.array([False])
        self.oneline.voigt.fix_lfwhm = np.array([False])
        self.oneline.voigt.nline = 1
        self.reset_sample_regions()
        self.reflect_ui()
        self.oneline.fit()
        self.reflect_ui()
        self.draw_fig()
    else:
      if event.key=='q':
        if self.mpl_status=='s':
          self.tmp_data['lvx1'].remove()
        self._clear_state()
      if self.mpl_status == 't':
        if event.key =='l':
          self.oneline.samples_line = \
            self.show_selected_region([],continuum=False)
          self.ui.edit_samples_line.setPlainText(\
            utils.textsamples(self.oneline.samples_line))
        elif event.key =='c':
          self.oneline.samples_continuum = \
            self.show_selected_region([],continuum=True)
          self.ui.edit_samples_cont.setPlainText(\
            utils.textsamples(self.oneline.samples_continuum))
        elif event.key == 'r':
          self.oneline.samples_continuum = \
          self.show_selected_region(\
             self.oneline.samples_continuum[:-1],continuum=True)
          self.ui.edit_samples_cont.setPlainText(\
            utils.textsamples(self.oneline.samples_continuum))
        elif event.key == 'b':
          self.oneline.samples_continuum = \
          self.show_selected_region(\
             self.oneline.samples_continuum[1:],continuum=True)
          self.ui.edit_samples_cont.setPlainText(\
            utils.textsamples(self.oneline.samples_continuum))
        self._clear_state()
      elif self.mpl_status == 'y':
        if event.key == 'y':
          self.moveon_done(use=True)
        self._clear_state()
      elif self.mpl_status == 's':
        x1 = self.tmp_data['x']
        x2 = event.xdata
        if (event.key=='l') and (x2 is not None):
          self.oneline.samples_line.append([x1,x2])
          self.oneline.samples_line = \
            self.show_selected_region(self.oneline.samples_line,continuum=False)
          self.ui.edit_samples_line.setPlainText(\
            utils.textsamples(self.oneline.samples_line))
        elif (event.key=='c') and (x2 is not None):
          self.oneline.samples_continuum.append([x1,x2])
          self.oneline.samples_continuum = \
            self.show_selected_region(self.oneline.samples_continuum,continuum=True)
          self.ui.edit_samples_cont.setPlainText(\
            utils.textsamples(self.oneline.samples_continuum))
        self.tmp_data['lvx1'].remove()
        self._clear_state()



def startgui(wavelength,flux,\
    wavelength_lines,
    fout=None,flog=None,elemid=None,gf=None,ep=None, resolution = 60000.):
  app = QApplication(sys.argv)
  window = MainWindow(resolution = resolution)
  window.input_data(wavelength,flux,wavelength_lines,
    fout=fout,flog=flog,elemid=elemid,gf=gf,ep=ep)
  window.fit_line(window.wavelength_lines[window.iline])
  window.show()
  sys.exit(app.exec_())

def test_main():
  xbin = np.linspace(-50.0,50.0,3000)
  yy = 1.5*(\
    1.0 - \
    0.050*voigt_profile(xbin+20.0,0.06,0.05)-\
    0.125*voigt_profile(xbin,0.06,0.05) - \
    0.030*voigt_profile(xbin+0.4,0.06,0.05))+\
    0.03*np.random.randn(len(xbin))+0.0*xbin
  #yy = 1.0-0.5*np.exp(-(xbin)**2./(2.0*0.1**2.))
  xbin = xbin+6000.
  with open('testew.op','w') as fout, open('testew.log','w') as flog:
    startgui(xbin,yy,[5980,6000.05],fout=fout,flog=flog)

if __name__ == '__main__':
  test_main()
